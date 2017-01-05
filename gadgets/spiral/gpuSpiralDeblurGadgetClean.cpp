#include "gpuSpiralDeblurGadgetClean.h"
#include "GenericReconJob.h"
#include "cuNDArray_utils.h"
#include "cuNDArray_reductions.h"
#include "vector_td_utilities.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray_fileio.h"
#include "vector_td.h"
#include "vector_td_operators.h"
#include "sense_utilities.h"
#include "check_CUDA.h"
#include "b1_map.h"
#include "GPUTimer.h"
#include "vds.h"
#include "ismrmrd/xml.h"
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

  gpuSpiralDeblurGadgetClean::gpuSpiralDeblurGadgetClean()
    : samples_to_skip_start_(0)
    , samples_to_skip_end_(0)
    , samples_per_interleave_(0)
    , prepared_(false)
  {
  }

  gpuSpiralDeblurGadgetClean::~gpuSpiralDeblurGadgetClean() {}

  int gpuSpiralDeblurGadgetClean::process_config(ACE_Message_Block* mb)
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

    // Start parsing the ISMRMRD XML header
    //

    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

	// Get the encoding space and trajectory description
    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;

    // Determine reconstruction matrix sizes
    //
	fov_vec_.push_back(r_space.fieldOfView_mm.x);
    fov_vec_.push_back(r_space.fieldOfView_mm.y);
    fov_vec_.push_back(r_space.fieldOfView_mm.z);

	kernel_width_ = buffer_convolution_kernel_width.value();
    oversampling_factor_ = buffer_convolution_oversampling_factor.value();
    
    image_dimensions_recon_.push_back(((static_cast<unsigned int>(std::ceil(e_space.matrixSize.x*reconstruction_os_factor_x.value()))+warp_size-1)/warp_size)*warp_size);  
    image_dimensions_recon_.push_back(((static_cast<unsigned int>(std::ceil(e_space.matrixSize.y*reconstruction_os_factor_y.value()))+warp_size-1)/warp_size)*warp_size);
      
    image_dimensions_recon_os_ = uint64d2
      (((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[0]*oversampling_factor_))+warp_size-1)/warp_size)*warp_size,
       ((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[1]*oversampling_factor_))+warp_size-1)/warp_size)*warp_size);

	return GADGET_OK;
  }



    int gpuSpiralDeblurGadgetClean::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {

		IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
		
		
		// Allocate various counters if they are NULL
		if( !image_counter_.get() ){
			image_counter_ = boost::shared_array<long>(new long[1]);
			image_counter_[0] = 0;
		}

        size_t R0 = recon_bit_->rbit_[0].data_.data_.get_size(0);
        size_t E1 = recon_bit_->rbit_[0].data_.data_.get_size(1);
        size_t E2 = recon_bit_->rbit_[0].data_.data_.get_size(2);
		size_t CHA = recon_bit_->rbit_[0].data_.data_.get_size(3);
		size_t N = recon_bit_->rbit_[0].data_.data_.get_size(4);
		size_t S = recon_bit_->rbit_[0].data_.data_.get_size(5);
		size_t SLC = recon_bit_->rbit_[0].data_.data_.get_size(6);

		hoNDArray<complext<float>> host_data(R0,E1,E2,CHA,N,S,SLC);	
		//std::cout << recon_bit_->rbit_[0].data_.data_.get_number_of_elements() << std::endl;
		//std::cout << host_data.get_size(0) << std::endl;
		//host_data = recon_bit_->rbit_[0].data_.data_;
					//write_nd_array<float>( host_traj, "hotraj.real" );
		memcpy(host_data.get_data_ptr(), recon_bit_->rbit_[0].data_.data_.get_data_ptr(), recon_bit_->rbit_[0].data_.data_.get_number_of_elements()*sizeof(complext<float>));
		write_nd_array<complext<float>>( &host_data, "hostdata.cplx");

		
		ISMRMRD::AcquisitionHeader& curr_header = recon_bit_->rbit_[0].data_.headers_(0, 0, 0);
		std::cout << curr_header.user_int[0] << std::endl;
		std::cout << curr_header.scan_counter << std::endl;
/*
		if(curr_header.user_int[0] == 0){
			if(!prepared_){
				//Setup image arrays
				std::vector<size_t> image_dims;
				image_dims.push_back(image_dimensions_recon_[0]);
				image_dims.push_back(image_dimensions_recon_[1]);
				image_dims.push_back(CHA);
				image.create(&image_dims);
				host_image.create(&image_dims);	
				host_image.fill(0.0f);	

				// Setup output image array
				image_dims.pop_back();
				reg_image.create(&image_dims);
				host_image.fill(0.0f);

				samples_per_interleave_ = recon_bit_->rbit_[0].data_.data_.get_size(0);
				Nints_ = recon_bit_->rbit_[0].data_.data_.get_size(1);

				std::cout << samples_per_interleave_ << std::endl;
				std::cout << Nints_ << std::endl;

				hoNDArray<float> trajectory = *(recon_bit_->rbit_[0].data_.trajectory_);
				write_nd_array<float>( &trajectory, "traj.real" );
				hoNDArray<floatd2> host_traj(new hoNDArray<floatd2>(R0*E1));
				hoNDArray<float> host_weights(new hoNDArray<float>(R0*E1));
				for (int i = 0; i < (samples_per_interleave_*Nints_); i++) {
					//std::cout << i << std::endl;
					host_traj[i]   = floatd2(trajectory[i*3],trajectory[i*3+1]);
					host_weights[i] = trajectory[i*3+2];
				}		

				//upload to gpu
				gpu_traj = host_traj;
				gpu_weights = host_weights;

				//write_nd_array<complext<float>>( &gpu_data, "gpudata.cplx");
				write_nd_array<floatd2>( &gpu_traj, "gputraj.real");

				//pre-process
				nfft_plan_.setup( from_std_vector<size_t,2>(image_dimensions_recon_), image_dimensions_recon_os_, kernel_width_ );
				nfft_plan_.preprocess(&gpu_traj, cuNFFT_plan<float,2>::NFFT_PREP_NC2C);
				prepared_ = true;
			}

			cuNDArray<complext<float>> gpu_data(host_data);
			nfft_plan_.compute( &gpu_data, &image, &gpu_weights, plan_type::NFFT_BACKWARDS_NC2C );

			csm_ = estimate_b1_map<float,2>( &image );	
			cuNDArray<complext<float>> deref_csm = *csm_;	
			csm_mult_MH<float,2>(&image, &reg_image, &deref_csm);
			host_image = *(reg_image.to_host());
			//write_nd_array<complext<float>>( &reg_image, "spiral_img.cplx" );

			// Prepare an image header for this frame
			GadgetContainerMessage<ISMRMRD::ImageHeader> *header = new GadgetContainerMessage<ISMRMRD::ImageHeader>();

			  {
			// Initialize header to all zeroes (there are a few fields we do not set yet)
			ISMRMRD::ImageHeader tmp;
			*(header->getObjectPtr()) = tmp;
			  }

			header->getObjectPtr()->version = curr_header.version;

			header->getObjectPtr()->matrix_size[0] = image_dimensions_recon_[0];
			header->getObjectPtr()->matrix_size[1] = image_dimensions_recon_[1];
			header->getObjectPtr()->matrix_size[2] = 1;

			header->getObjectPtr()->field_of_view[0] = fov_vec_[0];
			header->getObjectPtr()->field_of_view[1] = fov_vec_[1];
			header->getObjectPtr()->field_of_view[2] = fov_vec_[2];
			header->getObjectPtr()->channels = 1;//base_head->active_channels;

			header->getObjectPtr()->slice = curr_header.idx.slice;
			header->getObjectPtr()->set = curr_header.idx.set;

			header->getObjectPtr()->acquisition_time_stamp = curr_header.acquisition_time_stamp;
			memcpy(header->getObjectPtr()->physiology_time_stamp, curr_header.physiology_time_stamp, sizeof(uint32_t)*ISMRMRD::ISMRMRD_PHYS_STAMPS);

			memcpy(header->getObjectPtr()->position, curr_header.position, sizeof(float)*3);
			memcpy(header->getObjectPtr()->read_dir, curr_header.read_dir, sizeof(float)*3);
			memcpy(header->getObjectPtr()->phase_dir, curr_header.phase_dir, sizeof(float)*3);
			memcpy(header->getObjectPtr()->slice_dir, curr_header.slice_dir, sizeof(float)*3);
			memcpy(header->getObjectPtr()->patient_table_position, curr_header.patient_table_position, sizeof(float)*3);

			header->getObjectPtr()->data_type = ISMRMRD::ISMRMRD_CXFLOAT;
			header->getObjectPtr()->image_index = image_counter_[0]++; 
			header->getObjectPtr()->image_series_index = 1;

			GadgetContainerMessage< hoNDArray< std::complex<float> > >* cm2 = new GadgetContainerMessage<hoNDArray< std::complex<float> > >();
			cm2->getObjectPtr()->create(host_image.get_dimensions());
			memcpy(cm2->getObjectPtr()->get_data_ptr(), host_image.get_data_ptr(), host_image.get_number_of_elements()*sizeof(std::complex<float>));
			header->cont(cm2);

			if (this->next()->putq(header) < 0) {
			  GDEBUG("Failed to put job on queue.\n");
			  header->release();
			  return GADGET_FAIL;
			}

		}
		*/
		m1->release();
		return GADGET_OK;
	}

  GADGET_FACTORY_DECLARE(gpuSpiralDeblurGadgetClean)
}






























