#include "gpuSpiralDeblurGadget.h"
#include "GenericReconJob.h"
#include "cuNDArray_utils.h"
#include "cuNDArray_reductions.h"
#include "vector_td_utilities.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray_fileio.h"
#include "vector_td.h"
#include "vector_td_operators.h"
#include "check_CUDA.h"
#include "b1_map.h"
#include "GPUTimer.h"
#include "vds.h"
#include "ismrmrd/xml.h"
#include "GPUTimer.h"

#include <algorithm>
#include <vector>

#define ARMA_64BIT_WORD

#ifdef USE_ARMADILLO
	#include <armadillo>
#endif

#ifdef USE_OMP
    #include <omp.h>
#endif

using namespace std;
// using namespace Gadgetron;

namespace Gadgetron {

// Define desired precision
typedef float _real;
typedef complext<_real> _complext;
typedef reald<_real,2>::Type _reald2;
typedef cuNFFT_plan<_real,2> plan_type;

  gpuSpiralDeblurGadget::gpuSpiralDeblurGadget()
    : samples_to_skip_start_(0)
    , samples_to_skip_end_(0)
    , samples_per_interleave_(0)
    , prepared_(false)
    , use_multiframe_grouping_(false)
    , acceleration_factor_(0)
  {
  }

  gpuSpiralDeblurGadget::~gpuSpiralDeblurGadget() {}

  int gpuSpiralDeblurGadget::process_config(ACE_Message_Block* mb)
  {

    int number_of_devices = 0;
    if (cudaGetDeviceCount(&number_of_devices)!= cudaSuccess) {
      GDEBUG( "Error: unable to query number of CUDA devices.\n" );
      return GADGET_FAIL;
    }

    if (number_of_devices == 0) {
      GDEBUG( "Error: No available CUDA devices.\n" );
      return GADGET_FAIL;
    }

    device_number_ = deviceno.value();

    if (device_number_ >= number_of_devices) {
      GDEBUG("Adjusting device number from %d to %d\n", device_number_,  (device_number_%number_of_devices));
      device_number_ = (device_number_%number_of_devices);
    }

    if (cudaSetDevice(device_number_)!= cudaSuccess) {
      GDEBUG( "Error: unable to set CUDA device.\n" );
      return GADGET_FAIL;
    }

    cudaDeviceProp deviceProp;
    if( cudaGetDeviceProperties( &deviceProp, device_number_ ) != cudaSuccess) {
      GDEBUG( "Error: unable to query device properties.\n" );
      return GADGET_FAIL;
    }
    
    unsigned int warp_size = deviceProp.warpSize;

    propagate_csm_from_set_ = propagate_csm_from_set.value();

    if( propagate_csm_from_set_ > 0 ){
      GDEBUG("Currently, only set 0 can propagate coil sensitivity maps. Set %d was specified.\n", propagate_csm_from_set_ );
      return GADGET_FAIL;
    }

    if( propagate_csm_from_set_ >= 0 ){
      GDEBUG("Propagating csm from set %d to all sets\n", propagate_csm_from_set_ );
    }

    buffer_using_solver_ = buffer_using_solver.value();
    use_multiframe_grouping_ = use_multiframe_grouping.value();

    if( buffer_using_solver_ && !use_multiframe_grouping_ ){
      GDEBUG("Enabling 'buffer_using_solver' requires also enabling 'use_multiframe_grouping'.\n" );
      return GADGET_FAIL;
    }

    // Start parsing the ISMRMRD XML header
    //

    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);
    
    
    if (h.encoding.size() != 1) {
      GDEBUG("This Gadget only supports one encoding space\n");
      return GADGET_FAIL;
    }
    
    // Get the encoding space and trajectory description
    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;

    // Determine reconstruction matrix sizes
    //

    kernel_width_ = buffer_convolution_kernel_width.value();
    oversampling_factor_ = buffer_convolution_oversampling_factor.value();
    
    image_dimensions_recon_.push_back(((static_cast<unsigned int>(std::ceil(e_space.matrixSize.x*reconstruction_os_factor_x.value()))+warp_size-1)/warp_size)*warp_size);  
    image_dimensions_recon_.push_back(((static_cast<unsigned int>(std::ceil(e_space.matrixSize.y*reconstruction_os_factor_y.value()))+warp_size-1)/warp_size)*warp_size);
      
    image_dimensions_recon_os_ = uint64d2
      (((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[0]*oversampling_factor_))+warp_size-1)/warp_size)*warp_size,
       ((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[1]*oversampling_factor_))+warp_size-1)/warp_size)*warp_size);
    
    // In case the warp_size constraint kicked in
    oversampling_factor_ = float(image_dimensions_recon_os_[0])/float(image_dimensions_recon_[0]);


    if (h.encoding[0].trajectoryDescription) {
      traj_desc = *h.encoding[0].trajectoryDescription;
    } else {
      GDEBUG("Trajectory description missing");
      return GADGET_FAIL;
    }
    
    if (traj_desc.identifier != "HargreavesVDS2000") {
      GDEBUG("Expected trajectory description identifier 'HargreavesVDS2000', not found.");
      return GADGET_FAIL;
    }
    
    
    long interleaves = -1;
    long fov_coefficients = -1;
    long sampling_time_ns = -1;
    double max_grad = -1.0;
    double max_slew = -1.0;
    double fov_coeff = -1.0;
    double kr_max = -1.0;
    
    
    for (std::vector<ISMRMRD::UserParameterLong>::iterator i (traj_desc.userParameterLong.begin()); i != traj_desc.userParameterLong.end(); ++i) {
      if (i->name == "interleaves") {
        interleaves = i->value;
      } else if (i->name == "fov_coefficients") {
        fov_coefficients = i->value;
      } else if (i->name == "SamplingTime_ns") {
        sampling_time_ns = i->value;
      } else {
        GDEBUG("WARNING: unused trajectory parameter %s found\n", i->name.c_str());
      }
    }

    for (std::vector<ISMRMRD::UserParameterDouble>::iterator i (traj_desc.userParameterDouble.begin()); i != traj_desc.userParameterDouble.end(); ++i) {
      if (i->name == "MaxGradient_G_per_cm") {
	max_grad = i->value;
      } else if (i->name == "MaxSlewRate_G_per_cm_per_s") {
	max_slew = i->value;
      } else if (i->name == "FOVCoeff_1_cm") {
	fov_coeff = i->value;
      } else if (i->name == "krmax_per_cm") {
	kr_max= i->value;
      } else {
	GDEBUG("WARNING: unused trajectory parameter %s found\n", i->name.c_str());
      }
    }
    
    if ((interleaves < 0) || (fov_coefficients < 0) || (sampling_time_ns < 0) || (max_grad < 0) || (max_slew < 0) || (fov_coeff < 0) || (kr_max < 0)) {
      GDEBUG("Appropriate parameters for calculating spiral trajectory not found in XML configuration\n");
      return GADGET_FAIL;
    }
    
    
    Tsamp_ns_ = sampling_time_ns;
    Nints_ = interleaves;
    interleaves_ = static_cast<int>(Nints_);

    gmax_ = max_grad;
    smax_ = max_slew;
    krmax_ = kr_max;
    fov_ = fov_coeff;

    samples_to_skip_start_  = 0; //n.get<int>(std::string("samplestoskipstart.value"))[0];
    samples_to_skip_end_    = -1; //n.get<int>(std::string("samplestoskipend.value"))[0];

    fov_vec_.push_back(r_space.fieldOfView_mm.x);
    fov_vec_.push_back(r_space.fieldOfView_mm.y);
    fov_vec_.push_back(r_space.fieldOfView_mm.z);

    slices_ = e_limits.slice ? e_limits.slice->maximum + 1 : 1;
    sets_ = e_limits.set ? e_limits.set->maximum + 1 : 1;

    buffer_ = boost::shared_array< ACE_Message_Queue<ACE_MT_SYNCH> >(new ACE_Message_Queue<ACE_MT_SYNCH>[slices_*sets_]);

    image_headers_queue_ = 
      boost::shared_array< ACE_Message_Queue<ACE_MT_SYNCH> >(new ACE_Message_Queue<ACE_MT_SYNCH>[slices_*sets_]);

    size_t bsize = sizeof(GadgetContainerMessage<ISMRMRD::ImageHeader>)*100*Nints_;

    for( unsigned int i=0; i<slices_*sets_; i++ ){
      image_headers_queue_[i].high_water_mark(bsize);
      image_headers_queue_[i].low_water_mark(bsize);
    }

    GDEBUG("smax:                    %f\n", smax_);
    GDEBUG("gmax:                    %f\n", gmax_);
    GDEBUG("Tsamp_ns:                %d\n", Tsamp_ns_);
    GDEBUG("Nints:                   %d\n", Nints_);
    GDEBUG("fov:                     %f\n", fov_);
    GDEBUG("krmax:                   %f\n", krmax_);
    GDEBUG("samples_to_skip_start_ : %d\n", samples_to_skip_start_);
    GDEBUG("samples_to_skip_end_   : %d\n", samples_to_skip_end_);
    GDEBUG("recon matrix_size_x    : %d\n", image_dimensions_recon_[0]);
    GDEBUG("recon matrix_size_y    : %d\n", image_dimensions_recon_[1]);

    return GADGET_OK;
  }

  int gpuSpiralDeblurGadget::
  process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
	  GadgetContainerMessage< hoNDArray< std::complex<float> > > *m2,
	  GadgetContainerMessage< hoNDArray<float> > *m3)
  {
    // Gadget process begin
    
	//Initialize timer for testing purposes (commented out)
	GPUTimer *timer;
	timer = new GPUTimer("Process time");

	//Check user_int[0], 1 & 2 = map acquisition, 0 = image data
	int flag = m1->getObjectPtr()->user_int[0];
	if(flag > 0){
		Nints = 1; //map is always single shot, variable nints_ is defined for image data
		interleaves_ = static_cast<int>(Nints);
		if(flag_old == 0){ prepared_ = false;} //If the flag changes, then the NFFT needs to be re-prepared
		flag_old = flag;
	}
	if(flag == 0){
		Nints = Nints_;
		interleaves_ = static_cast<int>(Nints);
		if(flag_old != flag){ prepared_ = false;} //If the flag changes, then the NFFT needs to be re-prepared
		flag_old = flag;
	}

	//Initialize and check if acquisition is noise and if it is the last in the slice
    bool is_noise = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
    if (is_noise) {
      m1->release();
      return GADGET_OK;
    }
	bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);

	//Prepare the NFFT on the GPU, if not previously done
    if (!prepared_) {

		//Initialize traj and weight parameters if it is the first interleave or a map acquisiton & allocate data buffer
		if(m1->getObjectPtr()->idx.kspace_encode_step_1 == 0 || flag > 0){
			
			traj_attached = 0; //1 = use attached trajectory, 0 = calc trajectory using calc_vds
			sample_time = (1.0*Tsamp_ns_) * 1e-9;
			samples_per_interleave_ = m1->getObjectPtr()->number_of_samples;

			host_traj_ = boost::shared_ptr< hoNDArray<floatd2> >(new hoNDArray<floatd2>);
			host_weights_ = boost::shared_ptr< hoNDArray<float> >(new hoNDArray<float>);

			std::vector<size_t> trajectory_dimensions;
			trajectory_dimensions.push_back(samples_per_interleave_*Nints);

			host_traj_->create(&trajectory_dimensions);
			host_weights_->create(&trajectory_dimensions);
			
			//Setup data buffer
			std::vector<size_t> data_dimensions;
			data_dimensions.push_back(samples_per_interleave_*interleaves_);
			data_dimensions.push_back(m1->getObjectPtr()->active_channels);
			host_data_buffer_ = boost::shared_array< hoNDArray<float_complext> > (new hoNDArray<float_complext>[slices_*sets_]);
			if (!host_data_buffer_.get()) {
				GDEBUG("Unable to allocate array for host data buffer\n");
				return GADGET_FAIL;
			}
			for (unsigned int i = 0; i < slices_*sets_; i++) {
			host_data_buffer_[i].create(&data_dimensions);
			host_data_buffer_[i].fill(0.0f);
      		}
		}

		//Read in traj and weight if attached (in m3)
		if(traj_attached) {
			float *p3 = m3->getObjectPtr()->get_data_ptr();
			float* co_ptr = reinterpret_cast<float*>(host_traj_->get_data_ptr());
			float* we_ptr =  reinterpret_cast<float*>(host_weights_->get_data_ptr());
			int index;
			float krmax = 0;
			int i;
			if(flag > 0){ index = 0; } //map interleave is always 0
			else{ index = m1->getObjectPtr()->idx.kspace_encode_step_1*samples_per_interleave_; }
			//Find krmax for nomalization, note that not always the last sample because of girf correction
			for (i = samples_per_interleave_-20; i < (samples_per_interleave_); i++) {
				  if(p3[i*3]*p3[i*3]+p3[i*3+1]*p3[i*3+1] > krmax){
						krmax = p3[i*3]*p3[i*3]+p3[i*3+1]*p3[i*3+1];
				  }
			}
			krmax = 2.0*std::sqrt(krmax);
			#ifdef USE_OMP
			#pragma omp parallel for default(none) private(i) shared(co_ptr, we_ptr, p3, krmax, index)
			#endif
			for (i = 0; i < (samples_per_interleave_); i++) {	
				  co_ptr[2*index+i*2]   = p3[i*3]/krmax;
				  co_ptr[2*index+i*2+1] = p3[i*3+1]/krmax;
				  we_ptr[index+i] = p3[i*3+2]/krmax;
			}

		}

		//Calcualte the trajectory and weights if not attached
		if (!traj_attached) {
			//Setup calc_vds parameters
			int     nfov   = 1;
			int     ngmax  = 1e5;       /*  maximum number of gradient samples      */
			double  *xgrad;             /*  x-component of gradient.                */
			double  *ygrad;             /*  y-component of gradient.                */
			double  *x_trajectory;
			double  *y_trajectory;
			double  *weighting;
			int     ngrad;
			//Map trajecotry is different, parameters defined in user_floats
			if (flag > 0) {
				double fov2_ = (m1->getObjectPtr()->user_float[5]);
				calc_vds(3*((m1->getObjectPtr()->user_float[3])/10.),(m1->getObjectPtr()->user_float[1])/10.,sample_time,sample_time,Nints,&fov2_,nfov,2*((m1->getObjectPtr()->user_float[4])/10000.),ngmax,&xgrad,&ygrad,&ngrad);
			}
			//Otherwise use calc_vds parameters from header (set in process_config_
			else{
				calc_vds(smax_,gmax_,sample_time,sample_time,Nints,&fov_,nfov,krmax_,ngmax,&xgrad,&ygrad,&ngrad);
			}
			calc_traj(xgrad, ygrad, samples_per_interleave_, Nints, sample_time, krmax_, &x_trajectory, &y_trajectory, &weighting);
			{
				float* co_ptr = reinterpret_cast<float*>(host_traj_->get_data_ptr());
				float* we_ptr =  reinterpret_cast<float*>(host_weights_->get_data_ptr());
				for (int i = 0; i < (samples_per_interleave_*Nints); i++) {
				  co_ptr[i*2]   = -x_trajectory[i]/(2);
				  co_ptr[i*2+1] = -y_trajectory[i]/(2);
				  we_ptr[i] = weighting[i];
				}
			}
			delete [] xgrad;
			delete [] ygrad;
			delete [] x_trajectory;
			delete [] y_trajectory;
			delete [] weighting;
		}

		//After traj/weight data is loaded in buffers, upload to GPU, setup, and prepocess NFFT
		if( is_last_scan_in_slice || Nints == 1){

			cuNDArray<floatd2> traj(*host_traj_);
			dcw_buffer_ = boost::shared_ptr< cuNDArray<float> >( new cuNDArray<float>(*host_weights_) );

			nfft_plan_.setup( from_std_vector<size_t,2>(image_dimensions_recon_), image_dimensions_recon_os_, kernel_width_ );
			nfft_plan_.preprocess(&traj, cuNFFT_plan<float,2>::NFFT_PREP_NC2C);

			prepared_ = true;
		}
    }


    // Allocate various counters if they are NULL
    if( !image_counter_.get() ){
      image_counter_ = boost::shared_array<long>(new long[slices_*sets_]);
      for( unsigned int i=0; i<slices_*sets_; i++ )
	image_counter_[i] = 0;
    }

    if( !interleaves_counter_singleframe_.get() ){
      interleaves_counter_singleframe_ = boost::shared_array<long>(new long[slices_*sets_]);
      for( unsigned int i=0; i<slices_*sets_; i++ )
	interleaves_counter_singleframe_[i] = 0;
    }

    if( !interleaves_counter_multiframe_.get() ){
      interleaves_counter_multiframe_ = boost::shared_array<long>(new long[slices_*sets_]);
      for( unsigned int i=0; i<slices_*sets_; i++ )
	interleaves_counter_multiframe_[i] = 0;
    }

    // Define some utility variables
    unsigned int samples_to_copy = m1->getObjectPtr()->number_of_samples-samples_to_skip_end_;
    unsigned int interleave = m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice = m1->getObjectPtr()->idx.slice;
    unsigned int set = m1->getObjectPtr()->idx.set;
    unsigned int samples_per_channel =  samples_per_interleave_*interleaves_;

    // Some book-keeping to keep track of the frame count
    interleaves_counter_singleframe_[set*slices_+slice]++;
    interleaves_counter_multiframe_[set*slices_+slice]++;

	//Copy interleave data into buffer
    ISMRMRD::AcquisitionHeader base_head = *m1->getObjectPtr();

    if (samples_to_skip_end_ == -1) {
      samples_to_skip_end_ = m1->getObjectPtr()->number_of_samples-samples_per_interleave_;
      GDEBUG("Adjusting samples_to_skip_end_ = %d\n", samples_to_skip_end_);
    }

    std::complex<float>* data_ptr = reinterpret_cast< std::complex<float>* > (host_data_buffer_[set*slices_+slice].get_data_ptr());
	std::complex<float>* profile_ptr = m2->getObjectPtr()->get_data_ptr();

    for (unsigned int c = 0; c < m1->getObjectPtr()->active_channels; c++) {
    	memcpy(data_ptr+c*samples_per_channel+interleave*samples_to_copy,profile_ptr+c*m1->getObjectPtr()->number_of_samples, samples_to_copy*sizeof(std::complex<float>));
    }

    // Have we received sufficient data for a new frame?
    if (is_last_scan_in_slice || Nints == 1) {

		// This was the final profile of a frame (check)
		if( Nints%interleaves_counter_singleframe_[set*slices_+slice] ){
			GDEBUG("Unexpected number of interleaves encountered in frame\n");
			return GADGET_FAIL;
		}	

		if( acceleration_factor_ != Nints/interleaves_counter_singleframe_[set*slices_+slice] ){
			GDEBUG("Change of acceleration factor detected\n");
			acceleration_factor_ =  Nints/interleaves_counter_singleframe_[set*slices_+slice];
      }

		// Prepare an image header for this frame
		GadgetContainerMessage<ISMRMRD::ImageHeader> *header = new GadgetContainerMessage<ISMRMRD::ImageHeader>();
		ISMRMRD::AcquisitionHeader *base_head = m1->getObjectPtr();

		  {
		// Initialize header to all zeroes (there is a few fields we do not set yet)
		ISMRMRD::ImageHeader tmp;
		*(header->getObjectPtr()) = tmp;
		  }

		header->getObjectPtr()->version = base_head->version;

		header->getObjectPtr()->matrix_size[0] = image_dimensions_recon_[0];
		header->getObjectPtr()->matrix_size[1] = image_dimensions_recon_[1];
		header->getObjectPtr()->matrix_size[2] = acceleration_factor_;

		header->getObjectPtr()->field_of_view[0] = fov_vec_[0];
		header->getObjectPtr()->field_of_view[1] = fov_vec_[1];
		header->getObjectPtr()->field_of_view[2] = fov_vec_[2];

		header->getObjectPtr()->channels = 1;//base_head->active_channels;
		header->getObjectPtr()->slice = base_head->idx.slice;
		header->getObjectPtr()->set = base_head->idx.set;

		header->getObjectPtr()->acquisition_time_stamp = base_head->acquisition_time_stamp;
		memcpy(header->getObjectPtr()->physiology_time_stamp, base_head->physiology_time_stamp, sizeof(uint32_t)*ISMRMRD::ISMRMRD_PHYS_STAMPS);

		memcpy(header->getObjectPtr()->position, base_head->position, sizeof(float)*3);
		memcpy(header->getObjectPtr()->read_dir, base_head->read_dir, sizeof(float)*3);
		memcpy(header->getObjectPtr()->phase_dir, base_head->phase_dir, sizeof(float)*3);
		memcpy(header->getObjectPtr()->slice_dir, base_head->slice_dir, sizeof(float)*3);
		memcpy(header->getObjectPtr()->patient_table_position, base_head->patient_table_position, sizeof(float)*3);

		header->getObjectPtr()->data_type = ISMRMRD::ISMRMRD_CXFLOAT;
		header->getObjectPtr()->image_index = image_counter_[set*slices_+slice]++; 
		header->getObjectPtr()->image_series_index = set*slices_+slice;

		if( !use_multiframe_grouping_ || 
		(use_multiframe_grouping_ && interleaves_counter_multiframe_[set*slices_+slice] == Nints) ){

			unsigned int num_coils = m1->getObjectPtr()->active_channels;

			// Compute coil images from the fully sampled data buffer

			//Setup image arrays
			std::vector<size_t> image_dims;
			image_dims.push_back(image_dimensions_recon_[0]);
			image_dims.push_back(image_dimensions_recon_[1]);
			image_dims.push_back(num_coils);
			cuNDArray<float_complext> image(&image_dims);

			// Setup output image array
			image_dims.pop_back();
			cuNDArray<_complext> reg_image(&image_dims);
			hoNDArray<_complext> output_image(&image_dims);

			float sample_time = (1.0*Tsamp_ns_) * 1e-9;

			//If data is for single shot off-res map			
			if(flag > 0){
				//filter data to ignore high-freq image components
				hoNDArray<_complext> map_samples0_filt(&host_data_buffer_[set*slices_+slice]);
				for(int i =0; i < samples_per_interleave_; i++){
					map_samples0_filt[i] *= exp(-.5*pow(i/200.,2.) );
				}

				// Upload map data to device
				cuNDArray<_complext> samples(map_samples0_filt);

				// Gridder first map data
				nfft_plan_.compute( &samples, &image, dcw_buffer_.get(), cuNFFT_plan<float,2>::NFFT_BACKWARDS_NC2C );

				// Setup Sense Operator
				boost::shared_ptr< cuNonCartesianSenseOperator<_real,2,false> > E ( new cuNonCartesianSenseOperator<_real,2,false>() );
				if( flag == 1){ //compute and save first map image
					csm_ = estimate_b1_map<float,2>( &image );
					E->setup( from_std_vector<size_t,2>(image_dimensions_recon_), image_dimensions_recon_os_, kernel_width_ );
					E->set_csm(csm_);
					E->mult_csm_conj_sum( &image, &reg_image );
					reg_image0 = reg_image;
				}
				else if( flag == 2){ //compute 2nd using existing csm
					cuNDArray<_complext> reg_image1(&image_dims);
					E->setup( from_std_vector<size_t,2>(image_dimensions_recon_), image_dimensions_recon_os_, kernel_width_ );
					E->set_csm(csm_);
					E->mult_csm_conj_sum( &image, &reg_image1 );	

					//Get map images from GPU
					boost::shared_ptr< hoNDArray<_complext> > host_image0 = reg_image0.to_host();
					boost::shared_ptr< hoNDArray<_complext> > host_image1 = reg_image1.to_host();

					//Comput B0 Map on host
					hoNDArray<_complext> temp0 = *host_image0;
					hoNDArray<_complext> temp1 = *host_image1;
					output_map = *(new hoNDArray<_real>(&image_dims));
					//hoNDArray<_real> output_map(&image_dims);
					for (int i = 0; i < image_dims[0]*image_dims[1]; i++) {
						output_map[i] = _real(arg(temp0[i]*conj(temp1[i]))/( 2*M_PI*.001 ));
						//std::cout << output_map[i] << std::endl;
					}
				}
			} //END MAP SCOPE

			//Compute deblurred image from map
			else{

				// Upload data to GPU
				cuNDArray<_complext> samples(&host_data_buffer_[set*slices_+slice]);

				// Reconstuct on-resonant image
				nfft_plan_.compute( &samples, &image, dcw_buffer_.get(), plan_type::NFFT_BACKWARDS_NC2C );

				// Setup SENSE opertor and compute CSM
				csm_ = estimate_b1_map<float,2>( &image );
				boost::shared_ptr< cuNonCartesianSenseOperator<_real,2,false> > E ( new cuNonCartesianSenseOperator<_real,2,false>() );
				E->setup( from_std_vector<size_t,2>(image_dimensions_recon_), image_dimensions_recon_os_, kernel_width_ );
				E->set_csm(csm_);

				//Deblur using Multi-frequency Interpolation
				int fmax = 600;
				int L = std::ceil(3*fmax*samples_per_interleave_*sample_time);
				std::complex<float> om (0.0,2*M_PI);
				std::cout << "L = " << L << std::endl;

				//Compute MFI coefficients if not they do not already exist
				if( MFI_C.get_number_of_elements() == 0 ){

					//Setup some arma matrices
					arma::cx_fmat demod( samples_per_interleave_ , L );
					MFI_C = hoNDArray<_complext>( fmax*2+1 , L );
					arma::cx_fvec b( samples_per_interleave_ );
					arma::cx_fvec x( L );
					//Setup and solve least-squares probelm
					int j = 0;
					for(float f = -fmax; f <= fmax; f += fmax*2./(L-1)){
						for(float i = 0; i < samples_per_interleave_; i++) {
							demod(i,j) = exp(om*i*sample_time*f);
						}
						j++;
					}
					j = 0;
					for(float f = -fmax; f <= fmax; f++){
						for(float i = 0; i < samples_per_interleave_; i++) {
							b(i) = exp(om*i*sample_time*f);
						}
						x = arma::solve(demod, b);
						memcpy(MFI_C.get_data_ptr()+j*L, x.memptr(), L*sizeof(std::complex<float>));		
						j++;
					}
				}

				//Initialize/Reset output image
				for (int i = 0; i < image_dims[0]*image_dims[1]; i++) {
					output_image[i] = 0;
				}
				hoNDArray<_complext> temp_image(&image_dims);
				hoNDArray<_complext> samples_demod( new hoNDArray<_complext> );
				int j = 0;

				//Interate over numer of base images
				for(float f = -fmax; f <= fmax; f += fmax*2./(L-1)){

					samples_demod = host_data_buffer_[set*slices_+slice];
					_complext omega = _complext(0,2*M_PI*f*sample_time);
					int i;

					//demodulate data at base frequency f
					#ifdef USE_OMP
					#pragma omp parallel for default(none) private(i) shared(num_coils, samples_demod, f, sample_time, omega)
					#endif
					for(i = 0; i < samples_per_interleave_*Nints*num_coils; i++) {
						samples_demod[i] *= exp(omega*(i%samples_per_interleave_));
					}

					//Upload samples and compute base image
					samples = samples_demod;
					nfft_plan_.compute( &samples, &image, dcw_buffer_.get(), plan_type::NFFT_BACKWARDS_NC2C );
					E->mult_csm_conj_sum( &image, &reg_image );
					temp_image = *(reg_image.to_host());
		
					//Update output image
					#ifdef USE_OMP
					#pragma omp parallel for default(none) private(i) shared(output_image, temp_image, image_dims, L, j, fmax)
					#endif
					for (i = 0; i < image_dims[0]*image_dims[1]; i++) {
						output_image[i] += (MFI_C[int(output_map[i]+fmax)*L+j]*temp_image[i]);
					}
					j++;
				}
				write_nd_array<_complext>( &output_image, "deblurred_im.cplx" );
				//Package image into message and pass on to next gadget
				GadgetContainerMessage< hoNDArray< std::complex<float> > >* cm2 = new GadgetContainerMessage<hoNDArray< std::complex<float> > >();
				cm2->getObjectPtr()->create(output_image.get_dimensions());
				memcpy(cm2->getObjectPtr()->get_data_ptr(), output_image.get_data_ptr(), output_image.get_number_of_elements()*sizeof(std::complex<float>));
				header->cont(cm2);

				if (this->next()->putq(header) < 0) {
				  GDEBUG("Failed to put job on queue.\n");
				  header->release();
				  return GADGET_FAIL;
				}
			}
			interleaves_counter_multiframe_[set*slices_+slice] = 0;
		}
		interleaves_counter_singleframe_[set*slices_+slice] = 0;
	}
    m1->release();
	delete timer;
    return GADGET_OK;
	
  }

  GADGET_FACTORY_DECLARE(gpuSpiralDeblurGadget)
}
