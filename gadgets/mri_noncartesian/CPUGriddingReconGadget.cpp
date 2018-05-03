#include "CPUGriddingReconGadget.h"
#include "mri_core_grappa.h"
#include "mri_core_coil_map_estimation.h"
#include "vector_td_utilities.h"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "hoNFFT.h"
#include "hoNDArray.h"
#include "hoNDArray_fileio.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_math.h"
#include "hoCgSolver.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_utils.h"
#include "cuNDArray_reductions.h"
#include "cuNonCartesianSenseOperator.h"
#include "cuCgSolver.h"
#include "cuCgPreconditioner.h"
#include "cuNFFT.h"
#include "cuNDArray_fileio.h"
#include "cuImageOperator.h"
#include "sense_utilities.h"
#include "b1_map.h"
#include <time.h>
#include <numeric>
#include "cudaDeviceManager.h"

namespace Gadgetron{
	CPUGriddingReconGadget::CPUGriddingReconGadget(){}

	CPUGriddingReconGadget::~CPUGriddingReconGadget(){}

	int CPUGriddingReconGadget::process_config(ACE_Message_Block *mb){
		GADGET_CHECK_RETURN(GenericReconGadget::process_config(mb) == GADGET_OK, GADGET_FAIL);
		ISMRMRD::IsmrmrdHeader h;
		deserialize(mb->rd_ptr(), h);
		auto matrixSize = h.encoding.front().encodedSpace.matrixSize;

		kernelWidth = kernelWidthProperty.value();
		oversamplingFactor = oversamplingFactorProperty.value();
		
		imageDims.push_back(matrixSize.x); 
		imageDims.push_back(matrixSize.y);
		
		cudaDeviceProp deviceProp;
		if( cudaGetDeviceProperties( &deviceProp, 0 ) == cudaSuccess) {
			unsigned int warp_size = deviceProp.warpSize;   
			imageDimsOs.push_back(std::ceil((matrixSize.x*oversamplingFactor+warp_size-1)/warp_size)*warp_size);
			imageDimsOs.push_back(std::ceil((matrixSize.y*oversamplingFactor+warp_size-1)/warp_size)*warp_size);				
			
		}
		else{
			imageDimsOs.push_back(matrixSize.x*oversamplingFactor);
			imageDimsOs.push_back(matrixSize.y*oversamplingFactor);
		}
		
		
		ref_plan_prepared = false;
		/*ISMRMRD::TrajectoryDescription traj_desc;
		if (h.encoding[0].trajectoryDescription) {
			traj_desc = *h.encoding[0].trajectoryDescription;
	    }else {
			GDEBUG("Trajectory description missing");
			return GADGET_FAIL;
		}
	
		for (std::vector<ISMRMRD::UserParameterDouble>::iterator i (traj_desc.userParameterDouble.begin()); i != traj_desc.userParameterDouble.end(); ++i) {
			if (i->name == "krmax_per_cm") {
				kr_max= i->value;
			} else {
				GDEBUG("WARNING: unused trajectory parameter %s found\n", i->name.c_str());
			}
		}*/

		return GADGET_OK;
	}

	int CPUGriddingReconGadget::process(GadgetContainerMessage<IsmrmrdReconData> *m1){
		IsmrmrdReconData *recon_bit_ = m1->getObjectPtr();
		process_called_times_++;
		for(size_t e = 0; e < recon_bit_->rbit_.size(); e++){
					
			IsmrmrdDataBuffered* buffer;
			if(true){
				buffer = &(recon_bit_->rbit_[e].data_);
			}else{
				buffer = (IsmrmrdDataBuffered*)&(*recon_bit_->rbit_[e].ref_);
			}
			IsmrmrdImageArray imarray;
			
			size_t RO = buffer->data_.get_size(0);
			size_t E1 = buffer->data_.get_size(1);
			size_t E2 = buffer->data_.get_size(2);
			size_t CHA = buffer->data_.get_size(3);
			size_t N = buffer->data_.get_size(4);
			size_t S = buffer->data_.get_size(5);
			size_t SLC = buffer->data_.get_size(6);

			imarray.data_.create(imageDims[0], imageDims[1], 1, 1, N, S, SLC);	
			
			auto ref_images = computeCsm((IsmrmrdDataBuffered*)&(*recon_bit_->rbit_[e].ref_));
			hoNDArray<std::complex<float>> csm_host_ = std::get<0>(ref_images);
			hoNDArray<std::complex<float>> reg_host_ = std::get<1>(ref_images);	
			
			auto &trajectory = *buffer->trajectory_;
			auto trajDcw = separateDcwAndTraj(&trajectory);

			boost::shared_ptr<hoNDArray<float>> dcw_host_ = 
				boost::make_shared<hoNDArray<float>>(std::get<1>(trajDcw).get());
			boost::shared_ptr<hoNDArray<floatd2>> traj_host_ = 
				boost::make_shared<hoNDArray<floatd2>>(std::get<0>(trajDcw).get());
			
			//if(recon_bit_->rbit_[e].ref_){// && !csm_.get_number_of_elements()){
				//auto ref_images = computeCsm((IsmrmrdDataBuffered*)&(*recon_bit_->rbit_[e].ref_));
			//}
			
			boost::shared_ptr<hoNDArray<float_complext>> csm_host_ptr(new hoNDArray<float_complext>( (hoNDArray<float_complext>*)&csm_host_ ));
			boost::shared_ptr<hoNDArray<float_complext>> reg_host_ptr(new hoNDArray<float_complext>( (hoNDArray<float_complext>*)&reg_host_ ));
			boost::shared_ptr< cuNDArray<floatd2> > traj(new cuNDArray<floatd2> ( traj_host_.get() ));
			boost::shared_ptr< cuNDArray<float> > dcw(new cuNDArray<float> ( dcw_host_.get() ));
			boost::shared_ptr< cuNDArray<float_complext> > csm(new cuNDArray<float_complext> ( csm_host_ptr.get() ));
			boost::shared_ptr< cuNDArray<float_complext> >device_image(new cuNDArray<float_complext> ( &imageDims ));

			cuCgSolver<float_complext> cg_;
			
			boost::shared_ptr< cuNonCartesianSenseOperator<float,2> > E_( new cuNonCartesianSenseOperator<float,2>() );
			// Define preconditioner
			boost::shared_ptr< cuCgPreconditioner<float_complext> > D_( new cuCgPreconditioner<float_complext>() );

			// Define regularization image operator
			boost::shared_ptr< cuImageOperator<float_complext> > R_( new cuImageOperator<float_complext>() );
			R_->set_weight( .3 );
			
			boost::shared_ptr< cuNDArray<float_complext> > reg_image(new cuNDArray<float_complext> ( reg_host_ptr.get() ));
			R_->compute(reg_image.get());

			// Define preconditioning weights
			boost::shared_ptr< cuNDArray<float> > _precon_weights = sum(abs_square(csm.get()).get(), 2);
			boost::shared_ptr<cuNDArray<float> > R_diag = R_->get();
			*R_diag *= float(.1);
			*_precon_weights += *R_diag;
			R_diag.reset();
			reciprocal_sqrt_inplace(_precon_weights.get());	
			boost::shared_ptr< cuNDArray<float_complext> > precon_weights = real_to_complex<float_complext>( _precon_weights.get() );
			_precon_weights.reset();
			D_->set_weights( precon_weights );

	
			// Setup solver
			
			cg_.set_encoding_operator( E_ );        // encoding matrix
			cg_.add_regularization_operator( R_ );  // regularization matrix
			cg_.set_preconditioner( D_ );           // preconditioning matrix
			cg_.set_max_iterations( iteration_max.value() );
			cg_.set_tc_tolerance( iteration_tol.value() );
			cg_.set_output_mode( (true) ? cuCgSolver<float_complext>::OUTPUT_VERBOSE : cuCgSolver<float_complext>::OUTPUT_SILENT);
			
			E_->set_domain_dimensions(&imageDims);
			E_->set_dcw( dcw );
			E_->set_csm( csm );
			E_->setup( uint64d2( imageDims[0], imageDims[1] ), uint64d2( imageDimsOs[0], imageDimsOs[1] ), kernelWidth );
			E_->preprocess(traj.get());
			hoNDArray<float_complext> host_image;
			
			hoNDArray<float_complext> data_n(RO,E1,E2,CHA,1,S,SLC);
			data_n.fill(0);
			//std::vector<size_t> newOrder = {0, 1, 2, 3, 4, 5, 6};
			//boost::shared_ptr<hoNDArray<float_complext>> permuted = permute((hoNDArray<float_complext>*)&data_n,&newOrder);
			//boost::shared_ptr<hoNDArray<float_complext>> data(new hoNDArray<float_complext>( permuted.get() ));
			memcpy(&data_n[0],&buffer->data_(0,0,0,0,0,0,0),sizeof(std::complex<float>)*RO*E1*E2*CHA*S*SLC);
			cuNDArray<float_complext> device_samples((hoNDArray<float_complext>*)&data_n );
			E_->set_codomain_dimensions(device_samples.get_dimensions().get());
			//device_samples *= *dcw;
			device_image = cg_.solve(&device_samples);
			host_image = *device_image->to_host();
			//write_nd_array(&host_image, "host_image.cplx");
			
			for(size_t n = 0; n < recon_bit_->rbit_[e].data_.data_.get_size(4); n++){	
				std::cout << n << std::endl;
				memcpy(&data_n[0],&buffer->data_(0,0,0,0,n,0,0),sizeof(std::complex<float>)*RO*E1*E2*CHA*S*SLC);
				//write_nd_array(&data_n,"data_n.cplx");
				device_samples = data_n;
				//device_samples *= *dcw;
				device_image = cg_.solve(&device_samples);
				//E_->mult_MH( &device_samples, device_image.get());
				host_image = *device_image->to_host();
				memcpy(&imarray.data_(0,0,0,0,n,0,0), host_image.get_data_ptr(), sizeof(float)*2*host_image.get_number_of_elements());

			}
			write_nd_array(&imarray.data_, "imagearray.cplx");
			this->compute_image_header(recon_bit_->rbit_[e], imarray, e);
			this->send_out_image_array(recon_bit_->rbit_[e], imarray, e, ((int)e + 1), GADGETRON_IMAGE_REGULAR);		
			
		}

		m1->release();

		return GADGET_OK;
	}

	boost::shared_ptr<hoNDArray<std::complex<float>>> CPUGriddingReconGadget::reconstruct(
		hoNDArray<std::complex<float>> *data,
		hoNDArray<std::complex<float>> coilMap,
		hoNDArray<floatd2> *traj,
		hoNDArray<float> *dcw,
		size_t nCoils
	){

		//write_nd_array(traj,"traj.real");
		//write_nd_array(dcw,"dcw.real");
		//write_nd_array(data,"data.cplx");
		std::vector<hoNFFT_plan<float, 2>> plans_;
		if(true){//plans_.size() < nCoils){
			for(int p = 0; p < nCoils; p++){
				hoNFFT_plan<float, 2> plan(
					from_std_vector<size_t, 2>(imageDims),
					oversamplingFactor,
					kernelWidth
				);
				plan.preprocess(*traj);
				plans_.push_back(plan);
			}
		}
		hoNDArray<std::complex<float>> arg(imageDimsOs);
		arg.create(imageDimsOs[0], imageDimsOs[1]);
		arg.fill(std::complex<float>(0.0,0.0));
		imageDimsOs.push_back(nCoils);
		hoNDArray<std::complex<float>> channelRecon(imageDimsOs);
		channelRecon.fill(std::complex<float>(0.0,0.0));
		//hoNDArray<std::complex<float>> coilMap(imageDimsOs);
		//coilMap.fill(std::complex<float>(0.0,0.0));
		imageDimsOs.pop_back();
		hoNDArray<float> empty_dcw;
		int i;
		if(!iterate.value()){
			#pragma omp parallel for shared(traj,dcw,channelRecon,data) num_threads(nCoils)
			for(i = 0; i < nCoils; ++i){
				hoNDArray<std::complex<float>> tmp(imageDimsOs);
				tmp.create(imageDimsOs[0], imageDimsOs[1]);
				hoNDArray<std::complex<float>> channelData;
				channelData.create(data->get_number_of_elements()/nCoils);
				memcpy(channelData.get_data_ptr(), data->begin()+i*(data->get_number_of_elements()/nCoils), sizeof(std::complex<float>)*(data->get_number_of_elements()/nCoils));
				channelData *= *dcw;
				plans_[i].compute(channelData, tmp, empty_dcw, hoNFFT_plan<float, 2>::NFFT_BACKWARDS_NC2C);
				memcpy(&channelRecon(0,0,i), tmp.get_data_ptr(), sizeof(std::complex<float>)*imageDimsOs[0]*imageDimsOs[1]);
			}
			//multiplyConj(channelRecon, channelRecon, channelRecon);
			//sum(&channelRecon,2);	
			//Gadgetron::coil_map_2d_Inati(channelRecon,coilMap, 7, 3);
			Gadgetron::coil_combine(channelRecon, coilMap, 2, arg);
			//sqrt_inplace(&channelRecon);
			return boost::make_shared<hoNDArray<std::complex<float>>>(arg);
		}
		else{
///First step
			std::vector<size_t> flat_dim = {imageDimsOs[0]*imageDimsOs[1]};
			//empty_dcw.fill(1.0);
			hoNDArray<std::complex<float>> channelData;
			hoNDArray<std::complex<float>> tmp(imageDimsOs);
			#pragma omp parallel for shared(traj,dcw,channelRecon,data) private(channelData,tmp) num_threads(nCoils)
			for(i = 0; i < nCoils; ++i){
				//hoNDArray<std::complex<float>> tmp(imageDimsOs);
				tmp.create(imageDimsOs[0], imageDimsOs[1]);
				tmp.fill(std::complex<float>(0.0,0.0));
				//hoNDArray<std::complex<float>> channelData;
				channelData.create(data->get_number_of_elements()/nCoils);
				memcpy(channelData.get_data_ptr(), data->begin()+i*(data->get_number_of_elements()/nCoils), sizeof(std::complex<float>)*(data->get_number_of_elements()/nCoils)/data->get_size(4));
				//if(i==0){write_nd_array(&channelData,"channelData0.cplx");}
				channelData *= *dcw;
				plans_[i].compute(channelData, tmp, empty_dcw, hoNFFT_plan<float, 2>::NFFT_BACKWARDS_NC2C);
				memcpy(&channelRecon(0,0,i), tmp.get_data_ptr(), sizeof(std::complex<float>)*imageDimsOs[0]*imageDimsOs[1]);
				//write_nd_array(&tmp,"tmp.cplx");
			}
			//Gadgetron::coil_map_2d_Inati(channelRecon,coilMap, 7, 3);
			//hoNDArray<std::complex<float>> I(coilMap);
			//multiplyConj(I,coilMap,I);
			//I *= *conj(&coilMap);
			//sum_over_dimension(I,I,2);
			//sqrt_inplace(&I);
			for(i = 0; i<channelRecon.get_number_of_elements(); i++){
				channelRecon[i] *= conj(coilMap[i]);
			}
			//write_nd_array(&p,"p.cplx");
			

///Setup CG
			hoNDArray<std::complex<float>> b(imageDimsOs);
			b.fill(std::complex<float>(0.0,0.0));
			hoNDArray<std::complex<float>> p(imageDimsOs);
			p.fill(std::complex<float>(0.0,0.0));
			//write_nd_array(&p,"p.cplx");
			for(int i =0; i < p.get_number_of_elements(); i++){
				for(int j = 0; j < nCoils; j++){
					p[i] += channelRecon[j*p.get_number_of_elements()+i];
				}
			}/*
			for(int x = 0; x<p.get_size(0); x++){
				for(int y = 0; y<p.get_size(1); y++){
					if(x < (imageDimsOs[0]-imageDims[0])/2 || x >= (imageDimsOs[0]+imageDims[0])/2 || y < (imageDimsOs[1]-imageDims[1])/2 || y >= (imageDimsOs[1]+imageDims[1])/2){
						p[x+p.get_size(1)*y] *= 0;
					}
				}
			}*/
			hoNDArray<std::complex<float>> r(p);
			//r.reshape(&flat_dim); 
			std::complex<float> rhr = std::complex<float>(0.0,0.0);
			std::complex<float> phq = std::complex<float>(0.0,0.0);
			for(int i =0; i < r.get_number_of_elements(); i++){
				if(real(r[i]) != 0)
					rhr += r[i]*conj(r[i]);
				}
			std::complex<float> rhr0 = rhr;
//Iterate
			size_t niter = 0;
			while(abs(rhr/rhr0) > iteration_tol.value() && niter < iteration_max.value()) {
				//std::cout << norm2(p) << std::endl;
				niter++;
				std::cout << "Iteration #" << niter << std::endl; 
				std::cout << "Iteration Tol = " << rhr/rhr0 << std::endl;
				channelRecon.create(imageDimsOs[0],imageDimsOs[1],nCoils);
				channelRecon.fill(std::complex<float>(0.0,0.0));
				write_nd_array(&p,"p.cplx");
				int j;
				#pragma omp parallel for shared(channelRecon) private(channelData,tmp,j) num_threads(nCoils)
				for(int i = 0; i < nCoils; i++){
					//hoNDArray<std::complex<float>> channelData;
					channelData.create(data->get_number_of_elements()/nCoils);
					channelData.fill(std::complex<float>(0.0,0.0));
					tmp = p;
					hoNDArray<std::complex<float>> coilMap_i = hoNDArray<std::complex<float>>(imageDimsOs[0],imageDimsOs[1], &coilMap(0,0,i));
					for(j = 0; j<tmp.get_number_of_elements(); j++){
						tmp[j] *= coilMap_i[j];
					}
					//write_nd_array(&tmp,"tmp0.cplx");
					plans_[i].compute(tmp, channelData, empty_dcw, hoNFFT_plan<float, 2>::NFFT_FORWARDS_C2NC);
					if(i==6){write_nd_array(&coilMap_i,"coilmap.cplx");}
					channelData *= *dcw;
					tmp.fill(std::complex<float>(0.0,0.0));
					/*for(int j = 0; j<channelData.get_number_of_elements(); j++){
						channelData[j] /= 5.0;
					}*/
					plans_[i].compute(channelData, tmp, empty_dcw, hoNFFT_plan<float, 2>::NFFT_BACKWARDS_NC2C);
					//write_nd_array(&tmp,"tmp.cplx");
					for(j = 0; j<tmp.get_number_of_elements(); j++){
						tmp[j] *= conj(coilMap_i[j]);
					}
					//write_nd_array(&tmp,"tmp_post.cplx");
					memcpy(&channelRecon(0,0,i), tmp.get_data_ptr(), sizeof(std::complex<float>)*imageDimsOs[0]*imageDimsOs[1]);
				}
				#pragma omp barrier
				hoNDArray<std::complex<float>> q(imageDimsOs);
				q.fill(std::complex<float>(0.0,0.0));
				for(int i =0; i < q.get_number_of_elements(); i++){
					for(int j = 0; j < nCoils; j++){
						q[i] += channelRecon[j*q.get_number_of_elements()+i];
					}
				}/*
				for(int x = 0; x<q.get_size(0); x++){
					for(int y = 0; y<q.get_size(1); y++){
						if(x < (imageDimsOs[0]-imageDims[0])/2 || x >= (imageDimsOs[0]+imageDims[0])/2 || y < (imageDimsOs[1]-imageDims[1])/2 || y >= (imageDimsOs[1]+imageDims[1])/2){
							q[x+q.get_size(1)*y] *= 0;
						}
					}
				}*/
				//write_nd_array(&q,"q.cplx");
				//std::cout << norm2(q) << std::endl;
				phq = std::complex<float>(0.0,0.0);
				for(int i =0; i < q.get_number_of_elements(); i++){
					if(real(p[i]) != 0)
						phq += p[i]*conj(q[i]);
				}
				std::cout << phq << std::endl;
				for(int i =0; i < b.get_number_of_elements(); i++){
					if(real(p[i]) != 0)
						b[i] += real(rhr)/real(phq)*p[i];
				}
				std::complex<float> rhro = rhr;
				rhr = std::complex<float>(0.0,0.0);
				//write_nd_array(&r,"r0.cplx");
				for(int i =0; i < r.get_number_of_elements(); i++){
					if(real(r[i]) != 0)
						r[i] -= real(rhro)/real(phq)*q[i];
						rhr += r[i]*conj(r[i]);
				}
				//write_nd_array(&p,"p0.cplx");
				//write_nd_array(&r,"r.cplx");
				for(int i =0; i < p.get_number_of_elements(); i++){
					if(real(p[i]) != 0)
						p[i] = r[i]+real(rhr)/real(rhro)*p[i];
				}
				//write_nd_array(&p,"p1.cplx");
			}
			plans_[0].fft(b, hoNFFT_plan<float, 2>::NFFT_FORWARDS);
			for(int x = 0; x < b.get_size(0); x++){
				float kx = (-1+x*2./b.get_size(0));
				for(int y = 0; y < b.get_size(1); y++){
					float ky = (-1+y*2./b.get_size(1));
					b[x+y*b.get_size(0)] *= .5+std::atan(100*(1-std::sqrt(kx*kx+ky*ky)))/M_PI;
				}
			}
			plans_[0].fft(b, hoNFFT_plan<float, 2>::NFFT_BACKWARDS);			
			return boost::make_shared<hoNDArray<std::complex<float>>>(b);
		}
	}	

	boost::shared_ptr<hoNDArray<std::complex<float>>> CPUGriddingReconGadget::reconstructChannel(
		hoNDArray<std::complex<float>> *data,
		hoNDArray<floatd2> *traj,
		hoNDArray<float> *dcw,
		hoNFFT_plan<float, 2> plan
	){	
		hoNDArray<std::complex<float>> result; result.create(imageDimsOs[0], imageDimsOs[1]);
		plan.compute(*data, result, *dcw, hoNFFT_plan<float, 2>::NFFT_BACKWARDS_NC2C); 
		return boost::make_shared<hoNDArray<std::complex<float>>>(result);	
	}

	std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>>
	CPUGriddingReconGadget::separateDcwAndTraj(
		hoNDArray<float> *dcwTraj
	){
		int reducedDims = 1;
		std::vector<size_t> dims = *dcwTraj->get_dimensions();
		for( int d = 1; d < dims.size(); d++ ){
			reducedDims *= dims[d];
		}
		//std::vector<size_t> reducedDims(dims.begin()+1, dims.end());
		auto dcw = boost::make_shared<hoNDArray<float>>(reducedDims);
		auto traj = boost::make_shared<hoNDArray<floatd2>>(reducedDims);

		auto dcwPtr = dcw->get_data_ptr();
		auto trajPtr = traj->get_data_ptr();
		auto ptr = dcwTraj->get_data_ptr();
		for(unsigned int i = 0; i != dcwTraj->get_number_of_elements()/3; i++){
			trajPtr[i] = floatd2(ptr[i*3],ptr[i*3+1]);
			dcwPtr[i] = ptr[i*3+2];
		}
		return std::make_tuple(traj, dcw);
	}

	std::tuple<hoNDArray<std::complex<float>>, hoNDArray<std::complex<float>>>
	CPUGriddingReconGadget::computeCsm(
		IsmrmrdDataBuffered *ref_
	){

		//hoNDArray<std::complex<float>> argOs_(imageDimsOs);
		//argOs_.create(imageDimsOs[0], imageDimsOs[1]);
		//argOs_.fill(std::complex<float>(0.0,0.0));
		hoNDArray<std::complex<float>> arg_(imageDims);

		size_t RO = ref_->data_.get_size(0);
		size_t E1 = ref_->data_.get_size(1);
		size_t E2 = ref_->data_.get_size(2);
		size_t CHA = ref_->data_.get_size(3);
		size_t N = ref_->data_.get_size(4);
		size_t S = ref_->data_.get_size(5);
		size_t SLC = ref_->data_.get_size(6);
		
		hoNDArray<std::complex<float>> refdata(RO,E1,E2,CHA,1,S,SLC);
		memcpy(&refdata[0],&ref_->data_(0,0,0,0,0,0,0),sizeof(std::complex<float>)*RO*E1*E2*CHA*S*SLC);
		
		hoNDArray<float> &refTrajectory = *ref_->trajectory_;
		std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>> refTrajDcw = separateDcwAndTraj(&refTrajectory);

		boost::shared_ptr<hoNDArray<float>> refDcw = 
			boost::make_shared<hoNDArray<float>>(std::get<1>(refTrajDcw).get());
		boost::shared_ptr<hoNDArray<floatd2>> refTraj = 
			boost::make_shared<hoNDArray<floatd2>>(std::get<0>(refTrajDcw).get());

		std::vector<size_t> newOrder = {0, 1, 2, 4, 5, 6, 3};
		refdata = *permute(&refdata,&newOrder);
		hoNDArray<std::complex<float>> images(imageDimsOs[0], imageDimsOs[1], refdata.get_size(6));	
		//hoNDArray<std::complex<float>> csmOs_(imageDimsOs[0], imageDimsOs[1], refdata.get_size(6));
		hoNDArray<std::complex<float>> csm_(imageDims[0], imageDims[1], refdata.get_size(6));
		/*
		std::vector<hoNFFT_plan<float, 2>> plans2_;
		for(int p = 0; p < refdata.get_size(6); p++){
			hoNFFT_plan<float, 2> plan(
				from_std_vector<size_t, 2>(imageDims),
				oversamplingFactor,
				kernelWidth
			);
			plan.preprocess(*refTraj.get());
			plans2_.push_back(plan);
		}	

		#pragma omp parallel for shared(images,plans2_) num_threads(refdata.get_size(6))
		for(int n = 0; n < refdata.get_size(6); n++){
			hoNDArray<std::complex<float>> cha_data(refdata.get_size(0),refdata.get_size(1),refdata.get_size(2),refdata.get_size(3),refdata.get_size(4),refdata.get_size(5),&refdata(0,0,0,0,0,0,n));		
			hoNDArray<std::complex<float>> image = *reconstructChannel(&cha_data,refTraj.get(),refDcw.get(),plans2_[n]);
			memcpy(&images(0,0,n),image.get_data_ptr(),image.get_number_of_elements()*sizeof(std::complex<float>));
		}
		Gadgetron::coil_map_2d_Inati(images,csmOs_, 14, 5);
		Gadgetron::coil_combine(images, csmOs_, 2, argOs_);
		* */
		boost::shared_ptr< cuNDArray<floatd2> > traj(new cuNDArray<floatd2> ( refTraj.get() ));
		boost::shared_ptr< cuNDArray<float> > dcw(new cuNDArray<float> ( refDcw.get() ));
		boost::shared_ptr< cuNDArray<float_complext> >device_samples(new cuNDArray<float_complext> ( (hoNDArray<float_complext>*)&refdata ));
		boost::shared_ptr< cuNDArray<float_complext> >device_images(new cuNDArray<float_complext> ( imageDims[0], imageDims[1], refdata.get_size(6) ));
		boost::shared_ptr< cuNDArray<float_complext> >device_image(new cuNDArray<float_complext> ( imageDims[0], imageDims[1] ));
		
		write_nd_array(traj.get(),"traj.real");
		
		if(!ref_plan_prepared){
			nfft_plan_.setup( from_std_vector<size_t,2>(imageDims), from_std_vector<size_t,2>(imageDimsOs), kernelWidth );
			nfft_plan_.preprocess(traj.get(), cuNFFT_plan<float,2>::NFFT_PREP_NC2C);
			//ref_plan_prepared = true; not for mult_slice
		}
		nfft_plan_.compute( device_samples.get(), device_images.get(), dcw.get(), cuNFFT_plan<float,2>::NFFT_BACKWARDS_NC2C );

		// Setup SENSE opertor and compute CSM
		auto cu_csm = estimate_b1_map<float,2>( device_images.get() );
		csm_mult_MH<float,2>(device_images.get(), device_image.get(), cu_csm.get());
		csm_ = *(hoNDArray<std::complex<float>>*)(cu_csm->to_host().get());
		arg_ = *(hoNDArray<std::complex<float>>*)(device_image->to_host().get());
/*
		for(int n = 0; n < refdata.get_size(6); n++){
			for(int x = 0; x < csm_.get_size(0); x++){
				for(int y = 0; y < csm_.get_size(1); y++){
					/*if(x < (imageDimsOs[0]-imageDims[0])/2 || x >= (imageDimsOs[0]+imageDims[0])/2 || y < (imageDimsOs[1]-imageDims[1])/2 || y >= (imageDimsOs[1]+imageDims[1])/2){
						csm_[x+csm_.get_size(1)*y+csm_.get_size(0)*csm_.get_size(1)*n] = 0;
					}//
					csm_[x+csm_.get_size(0)*y+csm_.get_size(0)*csm_.get_size(1)*n] = csmOs_[(x+(imageDimsOs[0]-imageDims[0])/2)+csmOs_.get_size(0)*(y+(imageDimsOs[1]-imageDims[1])/2)+csmOs_.get_size(0)*csmOs_.get_size(1)*n];
					if(n==0){arg_[x+arg_.get_size(0)*y] = argOs_[(x+(imageDimsOs[0]-imageDims[0])/2)+argOs_.get_size(0)*(y+(imageDimsOs[1]-imageDims[1])/2)];}
				}
			}
		}
*/
		write_nd_array(&csm_,"csm.cplx");
		write_nd_array(&arg_,"arg.cplx");
		return std::make_tuple(csm_,arg_);
	}	

	GADGET_FACTORY_DECLARE(CPUGriddingReconGadget);
}
