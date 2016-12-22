// Deblurring stand alone Gadget

// Read in three scans (2 field map scans, 1 full resolution image)
// Reconstruct Field map
// Save Fied map
// Reconstruct MFI base images
// Read-in or calculate MFI coefficients
// Reconstruct final image

#include "cuNFFT.h"
#include "radial_utilities.h"
#include "vector_td_utilities.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray_fileio.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_blas.h" 
#include "cuNDArray_utils.h" 
#include "cuNDArray_operators.h"
#include "cuNDArray_reductions.h" 
#include "GPUTimer.h"
#include "parameterparser.h"
#include "complext.h"
#include "vds.h"
#include "b1_map.h"
#include "cuNonCartesianSenseOperator.h"
#ifdef USE_OMP
    #include <omp.h>
#endif

#include <iostream>

using namespace std;
using namespace Gadgetron;

// Define desired precision
typedef float _real;
typedef complext<_real> _complext;
typedef reald<_real,2>::Type _reald2;
typedef cuNFFT_plan<_real,2> plan_type;

int main( int argc, char** argv)
{

  //
  // Parse command line
  //

  ParameterParser parms;
  parms.add_parameter( 'd', COMMAND_LINE_STRING, 1, "Input samples file name (.cplx)", true );
  parms.add_parameter( '0', COMMAND_LINE_STRING, 1, "Input samples 1st shot file name (.cplx)", true );
  parms.add_parameter( '1', COMMAND_LINE_STRING, 1, "Input samples 2nd shot file name (.cplx)", true );
  parms.add_parameter( 'r', COMMAND_LINE_STRING, 1, "Output image file name (.cplx)", true, "result.cplx" );
  parms.add_parameter( 'm', COMMAND_LINE_INT,    1, "Output field map file name (.real)", true, "OffResMap.real" );
  parms.add_parameter( 's', COMMAND_LINE_INT,    1, "Matrix size", true );
  parms.add_parameter( 'o', COMMAND_LINE_INT,    1, "Oversampled matrix size", true );
  parms.add_parameter( 'k', COMMAND_LINE_FLOAT,  1, "Kernel width", true, "5.5" );

  parms.parse_parameter_list(argc, argv);
  if( parms.all_required_parameters_set() ){
    cout << " Running reconstruction with the following parameters: " << endl;
    parms.print_parameter_list();
  }
  else{
    cout << " Some required parameters are missing: " << endl;
    parms.print_parameter_list();
    parms.print_usage();
    return 1;
  }

  GPUTimer *timer;
  GPUTimer *timer2;

  // Load samples from disk

  boost::shared_ptr< hoNDArray<_complext> > data_samples = read_nd_array<_complext>((char*)parms.get_parameter('d')->get_string_value());
  boost::shared_ptr< hoNDArray<_complext> > map_samples0 = read_nd_array<_complext>((char*)parms.get_parameter('0')->get_string_value());
  boost::shared_ptr< hoNDArray<_complext> > map_samples1 = read_nd_array<_complext>((char*)parms.get_parameter('1')->get_string_value());

  // Setup OffResMap parameters
  unsigned int num_profiles = map_samples0->get_size(2);
  unsigned int samples_per_profile = map_samples0->get_size(0);
  unsigned int num_channels = map_samples0->get_size(1);
  uint64d2 matrix_size = uint64d2(parms.get_parameter('s')->get_int_value(), parms.get_parameter('s')->get_int_value());
  uint64d2 matrix_size_os = uint64d2(parms.get_parameter('o')->get_int_value(), parms.get_parameter('o')->get_int_value());
  _real kernel_width = parms.get_parameter('k')->get_float_value();

  double smax_ =14414.4;
  double gmax_ =2.4;
  double interleaves = double(num_profiles);
  std::cout << interleaves << " interleaves" << std::endl;
  std::cout << samples_per_profile << " samples" << std::endl;
  double dt = 3.9e-6;
  double fov_ = 40;
  double krmax_ = 1.6;
  double sample_time = dt;

  // Compute trajectories
  timer = new GPUTimer("Computing sprial trajectories");
  /*	call c-function here to calculate gradients */
  int     nfov   = 1;         /*  number of fov coefficients.             */
  int     ngmax  = 1e5;       /*  maximum number of gradient samples      */
  double  *xgrad;             /*  x-component of gradient.                */
  double  *ygrad;             /*  y-component of gradient.                */
  double  *x_trajectory;
  double  *y_trajectory;
  double  *weighting;
  int     ngrad;
  //int     count;
  calc_vds(smax_,gmax_,sample_time,sample_time,interleaves,&fov_,nfov,krmax_,ngmax,&xgrad,&ygrad,&ngrad);
  std::cout << ngrad << std::endl;
  /* Calcualte the trajectory and weights*/
  calc_traj(xgrad, ygrad, samples_per_profile, interleaves, sample_time, krmax_, &x_trajectory, &y_trajectory, &weighting);
  boost::shared_ptr< hoNDArray<floatd2> > host_traj_ = boost::shared_ptr< hoNDArray<floatd2> >(new hoNDArray<floatd2>);
  boost::shared_ptr< hoNDArray<float> > host_weights_ = boost::shared_ptr< hoNDArray<float> >(new hoNDArray<float>);
  std::vector<size_t> trajectory_dimensions;
  trajectory_dimensions.push_back(samples_per_profile*interleaves);
  host_traj_->create(&trajectory_dimensions);
  host_weights_->create(&trajectory_dimensions);
  {
    float* co_ptr = reinterpret_cast<float*>(host_traj_->get_data_ptr());
    float* we_ptr =  reinterpret_cast<float*>(host_weights_->get_data_ptr());
    for (int i = 0; i < (samples_per_profile*interleaves); i++) {
      //std::cout << "data=" << map_samples0[i] << std::endl;
      //co_ptr[i*2]   = -x_trajectory[i]/(2*M_PI);
      //co_ptr[i*2+1] = -y_trajectory[i]/(2*M_PI);
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
  delete timer;

  //filter data
  hoNDArray<_complext> map_samples0_filt = *(map_samples0.get());
  hoNDArray<_complext> map_samples1_filt = *(map_samples1.get());
  for(int i =0; i < samples_per_profile; i++){
    map_samples0_filt[i] *= exp(-.5*pow((i)/400.,2.));
    map_samples1_filt[i] *= exp(-.5*pow((i)/400.,2.));
  }
  //write_nd_array( &filt , "filt.real");

  // Initialize plan
  timer = new GPUTimer("Initializing plan");
  plan_type plan( matrix_size, matrix_size_os, kernel_width );
  delete timer;

  // Preprocess
  timer = new GPUTimer("NFFT preprocessing");
  boost::shared_ptr< cuNDArray<_reald2> > traj = boost::shared_ptr< cuNDArray<_reald2> >( new cuNDArray<_reald2>(*host_traj_) );
  boost::shared_ptr< cuNDArray<_real> > dcw_buffer_ = boost::shared_ptr< cuNDArray<_real> >( new cuNDArray<_real>(*host_weights_) );
  write_nd_array( traj.get() , "traj.real" );
  write_nd_array(dcw_buffer_.get(),"dcw.real");
  plan.preprocess( traj.get() , plan_type::NFFT_PREP_NC2C );
  delete timer;

  // Setup resulting image array
  vector<size_t> image_dims = to_std_vector(matrix_size);
  image_dims.push_back(num_channels);
  std::cout << image_dims[0] << " " << image_dims[1] << " " << image_dims[2] <<" image dimensions" << std::endl;
  cuNDArray<_complext> image(&image_dims);

  // Upload map data to device
  timer = new GPUTimer("Uploading samples to device");
  cuNDArray<_complext> samples(map_samples0_filt);
  delete timer;

  // Gridder first map data
  timer = new GPUTimer("Computing adjoint nfft (gridding)");
  plan.compute( &samples, &image, dcw_buffer_.get(), plan_type::NFFT_BACKWARDS_NC2C );
  delete timer;

  // Setup output image array
  image_dims.pop_back();
  cuNDArray<_complext> map_image0(&image_dims);
  cuNDArray<_complext> map_image1(&image_dims);
  //cuNDArray<_complext> output_map(&image_dims);

  // Setup SENSE opertor and compute CSM
  boost::shared_ptr< cuNDArray<float_complext> > csm = estimate_b1_map<float,2>( &image );
  boost::shared_ptr< cuNonCartesianSenseOperator<_real,2,false> > E ( new cuNonCartesianSenseOperator<_real,2,false>() ); 
  E->setup( matrix_size, matrix_size_os, kernel_width );
  E->set_csm(csm);
  E->mult_csm_conj_sum( &image, &map_image0 );	

  // Upload map data to device
  timer = new GPUTimer("Uploading samples to device");
  samples = map_samples1_filt;
  std::cout << "smaples uploaded" << std::endl;
  delete timer;

  // Gridder second map data
  timer = new GPUTimer("Computing adjoint nfft (gridding)");
  plan.compute( &samples, &image, dcw_buffer_.get(), plan_type::NFFT_BACKWARDS_NC2C );
  delete timer;
  E->mult_csm_conj_sum( &image, &map_image1 );
  
  //Comput B0 Map on host
  boost::shared_ptr< hoNDArray<_complext> > host_image0 = map_image0.to_host();
  boost::shared_ptr< hoNDArray<_complext> > host_image1 = map_image1.to_host();
  hoNDArray<_complext> temp0 = *host_image0;
  hoNDArray<_complext> temp1 = *host_image1;
  hoNDArray<_real> output_map(&image_dims);
  for (int i = 0; i < matrix_size[0]*matrix_size[1]; i++) {
    output_map[i] = _real(arg(temp0[i]*conj(temp1[i]))/( 2*M_PI*.001 ));
    //std::cout << output_map[i] << std::endl;
  }
  write_nd_array<_real>( &output_map, "map.real" );





/////// Setup NUFFT for image data
  // Setup OffResMap parameters
  num_profiles = data_samples->get_size(2);
  num_channels = data_samples->get_size(1);
  //interleaves = double(num_profiles);
  interleaves = 16;
  samples_per_profile = data_samples->get_size(0)/interleaves;
  std::cout << interleaves << " interleaves" << std::endl;
  std::cout << samples_per_profile << " samples" << std::endl;
  // Compute trajectories
  timer = new GPUTimer("Computing sprial trajectories");
  /*	call c-function here to calculate gradients */
  calc_vds(smax_,gmax_,sample_time,sample_time,interleaves,&fov_,nfov,krmax_,ngmax,&xgrad,&ygrad,&ngrad);
  /* Calcualte the trajectory and weights*/
  calc_traj(xgrad, ygrad, samples_per_profile, interleaves, sample_time, krmax_, &x_trajectory, &y_trajectory, &weighting);
  host_traj_ = boost::shared_ptr< hoNDArray<floatd2> >(new hoNDArray<floatd2>);
  host_weights_ = boost::shared_ptr< hoNDArray<float> >(new hoNDArray<float>);
  trajectory_dimensions.pop_back();
  trajectory_dimensions.push_back(samples_per_profile*interleaves);
  host_traj_->create(&trajectory_dimensions);
  host_weights_->create(&trajectory_dimensions);
  {
    float* co_ptr = reinterpret_cast<float*>(host_traj_->get_data_ptr());
    float* we_ptr =  reinterpret_cast<float*>(host_weights_->get_data_ptr());
    for (int i = 0; i < (samples_per_profile*interleaves); i++) {
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
  delete timer;
  // Preprocess
  timer = new GPUTimer("NFFT preprocessing");
  traj = boost::shared_ptr< cuNDArray<_reald2> >( new cuNDArray<_reald2>(*host_traj_) );
  dcw_buffer_ = boost::shared_ptr< cuNDArray<_real> >( new cuNDArray<_real>(*host_weights_) );
  write_nd_array( traj.get() , "traj.real" );
  write_nd_array(dcw_buffer_.get(),"dcw.real");
  plan.preprocess( traj.get() , plan_type::NFFT_PREP_NC2C );
  delete timer;
  // Upload map data to device
  timer = new GPUTimer("Uploading samples to device");
  samples = *(data_samples.get());
  delete timer;
  // Gridder first map data
  timer = new GPUTimer("Computing adjoint nfft (gridding)");
  plan.compute( &samples, &image, dcw_buffer_.get(), plan_type::NFFT_BACKWARDS_NC2C );
  delete timer;
  // Setup output image array
  cuNDArray<_complext> base_image(&image_dims);
  // Setup SENSE opertor and compute CSM
  csm = estimate_b1_map<float,2>( &image );
  E->setup( matrix_size, matrix_size_os, kernel_width );
  E->set_csm(csm);
  E->mult_csm_conj_sum( &image, &base_image );	
  // Output result
  timer = new GPUTimer("Output result to disk");
  boost::shared_ptr< hoNDArray<_complext> > host_image = base_image.to_host();
  write_nd_array<_complext>( host_image.get(), (char*)parms.get_parameter('r')->get_string_value());
  delete timer;

  //Compute MFI Coeffs
  int fmax = 600;
  int L = ceil(3*fmax*samples_per_profile*sample_time);
  std::cout << L << std::endl;
  hoNDArray<_complext> MFI_C((fmax*2+1)*L);
  streampos size;
  ifstream mfifile ("MFI_Coeff_L7.cplx", ios::in|ios::binary|ios::ate);
  size = mfifile.tellg()/sizeof(double);
  double * memblock = new double [size];
  mfifile.seekg( 0, ios::beg );
  mfifile.read((char *)memblock, sizeof(double)*size);
  mfifile.close();
  for( int i = 0; i < size; i+=2){
    MFI_C[i/2] = _complext(memblock[i],memblock[i+1]);
  }

  //Compute Base Images
  timer = new GPUTimer("MFI Loop");
  hoNDArray<_complext> output_image(&image_dims);
  for (int i = 0; i < matrix_size[0]*matrix_size[1]; i++) {
    output_image[i] = 0;
  }
  hoNDArray<_complext> temp_image(&image_dims);
  _complext I = _complext(0,1);
  //hoNDArray<_complext> demod(num_channels*samples_per_profile*interleaves);
  hoNDArray<_complext> samples_demod( new hoNDArray<_complext> );
  //float F[L];
  int j = 0;
  int indx = 0;
  for(float f = -fmax; f <= fmax; f += fmax*2./L){
	timer2 = new GPUTimer("get samples and demodulate");
    samples_demod = *(data_samples.get());
    std:cout << f <<std::endl;
	_complext omega = _complext(0,2*M_PI*f*sample_time);
	int i;
	int NS = samples_per_profile*interleaves*num_channels;
	//timer2 = new GPUTimer("Demodulate");
	#ifdef USE_OMP
	#pragma omp parallel for default(none) private(i) shared(NS, samples_demod, omega, samples_per_profile)
	#endif
    for(i = 0; i < NS; i++) {
		samples_demod[i] *= exp(omega*(i%samples_per_profile));
	}
	delete timer2;
	timer2 = new GPUTimer("Upload, NFFT Compute, recall");
    samples = samples_demod;
    plan.compute( &samples, &image, dcw_buffer_.get(), plan_type::NFFT_BACKWARDS_NC2C );
    E->mult_csm_conj_sum( &image, &base_image );
	temp_image = *(base_image.to_host());
	delete timer2;
/*    if( j == 4){  
      write_nd_array<_complext>( &temp_image, "base_im" );
    }*/
	timer2 = new GPUTimer("MFI");
	#ifdef USE_OMP
	#pragma omp parallel for default(none) private(i) shared(output_image, output_map, temp_image, matrix_size, L, j, fmax, MFI_C)
	#endif
    for (i = 0; i < matrix_size[0]*matrix_size[1]; i++) {
      output_image[i] += MFI_C[output_map[i]+fmax*L+j]*temp_image[i];
    }
	delete timer2;
    j++;
  }
  delete timer;
  //write_nd_array<_complext>( &output_image, "deblurred_im.cplx" );
  //output_image.clear();



  return 0;
}
