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
#include <time.h>
#include <numeric>

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
		
		imageDimsOs.push_back(matrixSize.x*oversamplingFactor);
		imageDimsOs.push_back(matrixSize.y*oversamplingFactor);
		
		ISMRMRD::TrajectoryDescription traj_desc;
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
		}

		return GADGET_OK;
	}

	int CPUGriddingReconGadget::process(GadgetContainerMessage<IsmrmrdReconData> *m1){
		IsmrmrdReconData *recon_bit_ = m1->getObjectPtr();
		process_called_times_++;
		for(size_t e = 0; e < recon_bit_->rbit_.size(); e++){
			IsmrmrdDataBuffered* buffer = &(recon_bit_->rbit_[e].data_);
			IsmrmrdImageArray imarray;
			
			size_t RO = buffer->data_.get_size(0);
			size_t E1 = buffer->data_.get_size(1);
			size_t E2 = buffer->data_.get_size(2);
			size_t CHA = buffer->data_.get_size(3);
			size_t N = buffer->data_.get_size(4);
			size_t S = buffer->data_.get_size(5);
			size_t SLC = buffer->data_.get_size(6);

			imarray.data_.create(imageDims[0], imageDims[1], 1, 1, N, S, SLC);			

			auto &trajectory = *buffer->trajectory_;
			auto trajDcw = separateDcwAndTraj(&trajectory);

			boost::shared_ptr<hoNDArray<float>> dcw = 
				boost::make_shared<hoNDArray<float>>(std::get<1>(trajDcw).get());
			boost::shared_ptr<hoNDArray<floatd2>> traj = 
				boost::make_shared<hoNDArray<floatd2>>(std::get<0>(trajDcw).get());
			
			std::vector<size_t> newOrder = {0, 1, 2, 4, 5, 6, 3};
			auto permuted = permute((hoNDArray<std::complex<float>>*)&buffer->data_,&newOrder);
			hoNDArray<std::complex<float>> data(*permuted);
			if(recon_bit_->rbit_[e].ref_ && !csm_.get_number_of_elements()){
				csm_ = computeCsm((IsmrmrdDataBuffered*)&(*recon_bit_->rbit_[e].ref_));
			}

			auto image = reconstruct(&data, csm_, traj.get(), dcw.get(), CHA);
			auto img = *image;
			hoNDArray<std::complex<float>> finalImage; finalImage.create(imageDims[0], imageDims[1]);
			
			// Crop the image
			size_t halfImageDims = (imageDimsOs[0]-imageDims[0])/2;
			for(size_t i = halfImageDims; i < imageDims[0]+halfImageDims; i++)
				for(size_t j = halfImageDims; j < imageDims[1]+halfImageDims; j++)
					finalImage[(i-halfImageDims)+(j-halfImageDims)*imageDims[0]] = img[i+j*imageDimsOs[0]]; 

			auto elements = imarray.data_.get_number_of_elements();
		 	memcpy(imarray.data_.get_data_ptr(), finalImage.get_data_ptr(), sizeof(float)*2*elements);			
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
		//std::vector<hoNFFT_plan<float, 2>> plans_;
		if(plans_.size() < nCoils){
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
		std::vector<size_t> dims = *dcwTraj->get_dimensions();
		std::vector<size_t> reducedDims(dims.begin()+1, dims.end());
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

	hoNDArray<std::complex<float>>
	CPUGriddingReconGadget::computeCsm(
		IsmrmrdDataBuffered *ref_
	){

		hoNDArray<std::complex<float>> arg(imageDimsOs);
		arg.create(imageDimsOs[0], imageDimsOs[1]);
		arg.fill(std::complex<float>(0.0,0.0));

		hoNDArray<std::complex<float>> refdata = ref_->data_;
		hoNDArray<float> &refTrajectory = *ref_->trajectory_;
		std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>> refTrajDcw = separateDcwAndTraj(&refTrajectory);

		boost::shared_ptr<hoNDArray<float>> refDcw = 
			boost::make_shared<hoNDArray<float>>(std::get<1>(refTrajDcw).get());
		boost::shared_ptr<hoNDArray<floatd2>> refTraj = 
			boost::make_shared<hoNDArray<floatd2>>(std::get<0>(refTrajDcw).get());

		std::vector<size_t> newOrder = {0, 1, 2, 4, 5, 6, 3};
		refdata = *permute(&refdata,&newOrder);
		hoNDArray<std::complex<float>> images(imageDimsOs[0], imageDimsOs[1], refdata.get_size(6));	
		csm_.create(imageDimsOs[0], imageDimsOs[1], refdata.get_size(6));

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
		Gadgetron::coil_map_2d_Inati(images,csm_, 5, 5);

		for(int n = 0; n < refdata.get_size(6); n++){
			for(int x = 0; x < csm_.get_size(0); x++){
				for(int y = 0; y < csm_.get_size(1); y++){
					if(x < (imageDimsOs[0]-imageDims[0])/2 || x >= (imageDimsOs[0]+imageDims[0])/2 || y < (imageDimsOs[1]-imageDims[1])/2 || y >= (imageDimsOs[1]+imageDims[1])/2){
						csm_[x+csm_.get_size(1)*y+csm_.get_size(0)*csm_.get_size(1)*n] = 0;
					}
				}
			}
		}

		Gadgetron::coil_combine(images, csm_, 2, arg);
		//write_nd_array(&arg,"csm.cplx");
		return csm_;
	}	

	GADGET_FACTORY_DECLARE(CPUGriddingReconGadget);
}
