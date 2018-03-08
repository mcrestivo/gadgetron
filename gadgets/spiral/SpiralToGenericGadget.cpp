#include "SpiralToGenericGadget.h"
#include "ismrmrd/xml.h"
#include "vds.h"
#include "radial_utilities.h"
#include "cuNDArray_fileio.h"

#include <algorithm>
#include <vector>

namespace Gadgetron{

  SpiralToGenericGadget::SpiralToGenericGadget()
    : samples_to_skip_start_(0)
    , samples_to_skip_end_(0)
    , samples_per_interleave_(0)
    , prepared_(false)
    , rep_(0)
  {
  }

  SpiralToGenericGadget::~SpiralToGenericGadget() {}

  int SpiralToGenericGadget::process_config(ACE_Message_Block* mb)
  {
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
  
  if(h.encoding[0].trajectory == "spiral"){
	  golden_angle = goldenAngle.value();
	  radial = false;
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
	  fov_[0] = fov_coeff;
	  fov_[1] = -fov_[0]*(1-vdsFactor.value())/kr_max;  

	  samples_to_skip_start_  =  0; //n.get<int>(std::string("samplestoskipstart.value"))[0];
	  samples_to_skip_end_    = -1; //n.get<int>(std::string("samplestoskipend.value"))[0];
	  
	  GDEBUG("smax:                    %f\n", smax_);
	  GDEBUG("gmax:                    %f\n", gmax_);
	  GDEBUG("Tsamp_ns:                %d\n", Tsamp_ns_);
	  GDEBUG("Nints:                   %d\n", Nints_);
	  GDEBUG("fov:                     %f\n", fov_[0]);
	  GDEBUG("krmax:                   %f\n", krmax_);
	  GDEBUG("samples_to_skip_start_ : %d\n", samples_to_skip_start_);
	  GDEBUG("samples_to_skip_end_   : %d\n", samples_to_skip_end_);
	  
  }
  else if(h.encoding[0].trajectory == "radial"){
	  //num_samples_per_profile
	  radial = true;
	  num_profiles_per_frame_ = radialProfiles.value();
	  num_frames_ = 1;
	  golden_angle = goldenAngle.value();
  }
  else{
	  GDEBUG("Trajectory in XML header is not set to spiral or radial\n");
	  return GADGET_FAIL;
  }
  
  return GADGET_OK;
  }
  
  int SpiralToGenericGadget::
  process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
	  GadgetContainerMessage< hoNDArray< std::complex<float> > > *m2)
  {
    // Noise should have been consumed by the noise adjust, but just in case...
    //

    bool is_noise = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
    if (is_noise) {
      m1->release();
      return GADGET_OK;
    }

    // Check attached trajectories. Delete previously attached trajectories if not correct. 
    if (m2->cont()) {
		GadgetContainerMessage< hoNDArray<float> >* m_traj = AsContainerMessage< hoNDArray<float> >(m2->cont());
		if(!(checkAttachedTrajectories(m_traj->getObjectPtr()))){
			m2->cont()->release();
		}
		else{
			if (this->next()->putq(m1) < 0) {
			  GDEBUG("Failed to put job on queue.\n");
			  return GADGET_FAIL;
			}
			return GADGET_OK;
		}
    }
    
    // Compute hoNDArray of trajectory and weights at first pass
    //
	int rep_last = rep_;
	rep_ = m1->getObjectPtr()->idx.repetition;
	
    if (!prepared_ || rep_ != rep_last) {
		
		samples_per_interleave_ = m1->getObjectPtr()->number_of_samples;//std::min(ngrad,static_cast<int>(m1->getObjectPtr()->number_of_samples));
		GDEBUG("Using %d samples per interleave\n", samples_per_interleave_);
		
		std::vector<size_t> trajectory_dimensions;
		trajectory_dimensions.push_back(3);
		if(radial){Nints_ = num_profiles_per_frame_;}
		if(!radial && golden_angle){
				Nints_ = 13;
		}
		trajectory_dimensions.push_back(samples_per_interleave_*Nints_);

		host_traj_ = boost::shared_ptr< hoNDArray<float> >(new hoNDArray<float>(&trajectory_dimensions));


		if(!radial){

			int     nfov   = 2;         /*  number of fov coefficients.             */
			int     ngmax  = 1e5;       /*  maximum number of gradient samples      */
			double  *xgrad;             /*  x-component of gradient.                */
			double  *ygrad;             /*  y-component of gradient.                */
			double  *x_trajectory;
			double  *y_trajectory;
			double  *weighting;
			int     ngrad;
			double sample_time = (1.0*Tsamp_ns_) * 1e-9;

			// Calculate gradients 
			calc_vds(smax_,gmax_,sample_time,sample_time,interleaves_,&fov_[0],nfov,krmax_,ngmax,&xgrad,&ygrad,&ngrad);

			// Calculate the trajectory and weights
			calc_traj(xgrad, ygrad, samples_per_interleave_, Nints_, sample_time, krmax_, &x_trajectory, &y_trajectory, &weighting);

			{
				float* co_ptr = reinterpret_cast<float*>(host_traj_->get_data_ptr());

				for (int i = 0; i < (samples_per_interleave_*Nints_); i++) {
				co_ptr[i*3+0] = -x_trajectory[i]/2;
				co_ptr[i*3+1] = -y_trajectory[i]/2;
				co_ptr[i*3+2] = weighting[i];
				}
			}

			delete [] xgrad;
			delete [] ygrad;
			delete [] x_trajectory;
			delete [] y_trajectory;
			delete [] weighting;
		}
		  
		if(radial){
			hoNDArray<floatd2> radial_traj;
			hoNDArray<float> radial_dcw; 
			if(!golden_angle){
				radial_traj = *compute_radial_trajectory_fixed_angle_2d<float>(samples_per_interleave_, 
																		num_profiles_per_frame_, 
																		num_frames_)->to_host();
				radial_dcw = *compute_radial_dcw_fixed_angle_2d<float>( samples_per_interleave_, 
																	num_profiles_per_frame_, 
																	1.25, 
																	1.0/(samples_per_interleave_/192))->to_host();
			}
			else{
				radial_traj = *compute_radial_trajectory_golden_ratio_2d<float>(samples_per_interleave_, 
																	num_profiles_per_frame_, 
																	num_frames_,
																	m1->getObjectPtr()->idx.repetition*num_profiles_per_frame_)->to_host();
				radial_dcw = *compute_radial_dcw_golden_ratio_2d<float>( samples_per_interleave_, 
																	num_profiles_per_frame_, 
																	1.25, 
																	1.0/(samples_per_interleave_/192))->to_host();
				
			}				
			
			{
				float* co_ptr = reinterpret_cast<float*>(host_traj_->get_data_ptr());

				for (int i = 0; i < (samples_per_interleave_*Nints_); i++) {
				co_ptr[i*3+0] = radial_traj[i][0];
				co_ptr[i*3+1] = radial_traj[i][1];
				co_ptr[i*3+2] = radial_dcw[i];
				}
			}
		}	

      prepared_ = true;
    }

    // Adjustments based in the incoming data
    //

    if (samples_to_skip_end_ == -1) {
      samples_to_skip_end_ = m1->getObjectPtr()->number_of_samples-samples_per_interleave_;
      GDEBUG("Adjusting samples_to_skip_end_ = %d\n", samples_to_skip_end_);
    }

    // Define some utility variables
    //

    unsigned int samples_to_copy = m1->getObjectPtr()->number_of_samples-samples_to_skip_end_;
    unsigned int interleave = m1->getObjectPtr()->idx.kspace_encode_step_1;

    // Prepare for a new array continuation for the trajectory/weights of the incoming profile
    //

    std::vector<size_t> trajectory_dimensions;
    trajectory_dimensions.push_back(3);
    trajectory_dimensions.push_back(samples_per_interleave_);
    
    hoNDArray<float> *traj_source = new hoNDArray<float>
      (&trajectory_dimensions, host_traj_->get_data_ptr()+3*samples_per_interleave_*m1->getObjectPtr()->idx.kspace_encode_step_1);
    
    // Make a new array as continuation of m1, and pass along
    //

    GadgetContainerMessage< hoNDArray<float> > *cont = new GadgetContainerMessage< hoNDArray<float> >();
    *(cont->getObjectPtr()) = *traj_source;
    m2->cont(cont);

    //We need to make sure that the trajectory dimensions are attached. 
    m1->getObjectPtr()->trajectory_dimensions = 3;
    
    if (this->next()->putq(m1) < 0) {
      GDEBUG("Failed to put job on queue.\n");
      return GADGET_FAIL;
    }
    
    return GADGET_OK;
  }
  
  bool SpiralToGenericGadget::checkAttachedTrajectories(hoNDArray<float> *traj){
		auto dim = *traj->get_dimensions();
		bool valid = false;
		if(dim.size() == 2){
			if(dim[0] == samples_per_interleave_){
				if(dim[1] == 3){
					valid = true;
				}
			}
		}
		return valid; //MCR always calculate new trajectories
		//return false;
   }
  
  GADGET_FACTORY_DECLARE(SpiralToGenericGadget)
}
