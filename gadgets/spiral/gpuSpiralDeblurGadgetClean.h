#ifndef gpuSpiralDeblurGadgetClean_H
#define gpuSpiralDeblurGadgetClean_H
#pragma once

#include "gadgetron_spiral_export.h"
#include "Gadget.h"
#include "GadgetMRIHeaders.h"
#include "cuNFFT.h"
#include "hoNDArray.h"
#include "vector_td.h"
#include "mri_core_data.h"
#include <ismrmrd/ismrmrd.h>
#include <complex>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <ismrmrd/xml.h>
using namespace std;

namespace Gadgetron{

  class EXPORTGADGETS_SPIRAL gpuSpiralDeblurGadgetClean :
    public Gadget1< IsmrmrdReconData >
  {

  public:
    GADGET_DECLARE(gpuSpiralDeblurGadgetClean);

    gpuSpiralDeblurGadgetClean();
    virtual ~gpuSpiralDeblurGadgetClean();

  protected:
    GADGET_PROPERTY(deviceno, int, "GPU device number", 0);
    GADGET_PROPERTY(buffer_convolution_kernel_width, float, "Convolution kernel width for buffer", 5.5);
    GADGET_PROPERTY(buffer_convolution_oversampling_factor, float, "Oversampling used in buffer convolution", 1.25);
    GADGET_PROPERTY(reconstruction_os_factor_x, float, "Oversampling for reconstruction in x-direction", 1.0);
    GADGET_PROPERTY(reconstruction_os_factor_y, float, "Oversampling for reconstruction in y-direction", 1.0);
	cuNDArray<float_complext> reg_image0;

    virtual int process_config(ACE_Message_Block* mb);
    
    virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
    
    //virtual GadgetContainerMessage<ISMRMRD::AcquisitionHeader>*
    //  duplicate_profile( GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *profile );
    
  private:
    int samples_to_skip_start_;
    int samples_to_skip_end_;
    int slices_;
    int sets_;
    int device_number_;
	int flag_old = 0;
	int traj_attached;

    long    Tsamp_ns_;
    long    Nints_;
	//long image_counter_;
	boost::shared_array<long> image_counter_;
    //boost::shared_array<long> interleaves_counter_singleframe_;
    //boost::shared_array<long> interleaves_counter_multiframe_;
    //long    acceleration_factor_;
    double  gmax_;
    double  smax_;
    double  krmax_;
    double  fov_;
	double sample_time;

    bool prepared_;
	bool prepared_B0_;

    float kernel_width_;
    float oversampling_factor_;

    //hoNDArray<floatd2> host_traj;
    //hoNDArray<float> host_weights;
    
    //boost::shared_ptr< cuNDArray<complex<float>> > gpu_data;
    //boost::shared_ptr< cuNDArray<float> > gpu_weights;

	cuNDArray<floatd2> gpu_traj;
	cuNDArray<float> gpu_weights;
	cuNDArray<float> gpu_weights_B0;

	hoNDArray<float> host_weights;
	hoNDArray<floatd2> host_traj;
	hoNDArray<floatd2> B0_traj;
	hoNDArray<float> B0_weights;

	cuNDArray<complext<float>> image;
	cuNDArray<complext<float>> reg_image;
	hoNDArray<complext<float>> host_image;	
	hoNDArray<complext<float>> output_image;
	hoNDArray<float> B0_map;

    std::vector<size_t> fov_vec_;
    std::vector<size_t> image_dimensions_recon_;
    uint64d2 image_dimensions_recon_os_;

    cuNFFT_plan<float,2> nfft_plan_;
	cuNFFT_plan<float,2> nfft_plan_B0_;
    boost::shared_ptr< cuNDArray<complext<float>> > csm_;


	hoNDArray<float_complext> MFI_C;
	hoNDArray<float_complext> phase_mask;
  };
}
#endif //gpuSpiralDeblurGadget_H
