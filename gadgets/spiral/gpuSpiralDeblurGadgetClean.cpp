#include "gpuSpiralDeblurGadgetClean.h"
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
		/*//GDEBUG("Size R0 = %d \n", recon_bit_->rbit_[0].data_.data_.get_size(0));
		//GDEBUG("Size E1 = %d \n", recon_bit_->rbit_[0].data_.data_.get_size(1));	
		if ( recon_bit_->rbit_[0].data_.data_.get_size(0) > 1788){
			hoNDArray<complex<float>> data = (recon_bit_->rbit_[0].data_.data_);
			hoNDArray<float> traj = *(recon_bit_->rbit_[0].data_.trajectory_);
			//GDEBUG("Size Traj = %d \n", traj.get_size(0));
			//GDEBUG("Size Traj = %d \n", traj.get_size(1));
			//GDEBUG("Size Traj = %d \n", traj.get_size(2));
			write_nd_array<complex<float>>( &data, "data.cplx" );
			write_nd_array<float>( &traj, "traj.cplx" );
		}*/

		//Setup image arrays
		std::vector<size_t> image_dims;
		image_dims.push_back(image_dimensions_recon_[0]);
		image_dims.push_back(image_dimensions_recon_[1]);
		image_dims.push_back(recon_bit_->rbit_[0].data_.data_.get_size(3));
		cuNDArray<complex<float>> image(&image_dims);
		hoNDArray<complex<float>> host_image(&image_dims);

		// Setup output image array
		image_dims.pop_back();
		cuNDArray<complex<float>> reg_image(&image_dims);

		dcw_buffer_ = boost::shared_ptr< cuNDArray<float> >( new cuNDArray<float>(*(recon_bit_->rbit_[0].data_.trajectory_[2])) );
		gpu_data_buffer_ = boost::shared_ptr< cuNDArray<complex<float>> >( new cuNDArray<complex<float>>(recon_bit_->rbit_[0].data_.data_) );

		nfft_plan_.setup( from_std_vector<size_t,2>(image_dimensions_recon_), image_dimensions_recon_os_, kernel_width_ );
		nfft_plan_.preprocess(recon_bit_->rbit_[0].data_.trajectory_[0:1], cuNFFT_plan<float,2>::NFFT_PREP_NC2C);
		
		nfft_plan_.compute( &gpu_data_buffer_, &image, dcw_buffer_.get(), plan_type::NFFT_BACKWARDS_NC2C );

		csm_ = estimate_b1_map<float,2>( &image );		
		csm_mult_MH(&image, &reg_image, csm_);
		host_image = *(reg_image.to_host());
		write_nd_array<complex<float>>( &host_image, "spiral_img.cplx" );

		m1->release();
		return GADGET_OK;
	}

  GADGET_FACTORY_DECLARE(gpuSpiralDeblurGadgetClean)
}






























