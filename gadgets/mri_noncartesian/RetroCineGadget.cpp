#include "RetroCineGadget.h"
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
#include "hoNDArray_fileio.h"
#include "hoCgSolver.h"
#include <time.h>
#include <numeric>

namespace Gadgetron{
	RetroCineGadget::RetroCineGadget(){}

	RetroCineGadget::~RetroCineGadget(){}

	int RetroCineGadget::process_config(ACE_Message_Block *mb){
		GADGET_CHECK_RETURN(GenericReconGadget::process_config(mb) == GADGET_OK, GADGET_FAIL);
		ISMRMRD::IsmrmrdHeader h;
		deserialize(mb->rd_ptr(), h);
		
        if (h.userParameters)
        {
            for (std::vector<ISMRMRD::UserParameterLong>::const_iterator i (h.userParameters->userParameterLong.begin()); 
                i != h.userParameters->userParameterLong.end(); i++)
            {
                    if (i->name == "RetroGatedImages") {
                        num_phases = i->value;
                        //num_phases = 50;
                    } else if (i->name == "RetroGatedSegmentSize") {
                        num_segments = i->value;
                    } else {
                        GDEBUG("WARNING: unused user parameter parameter %s found\n", i->name.c_str());
                    }
            }
        } else {
            GDEBUG("RetroGated parameters are supposed to be in the UserParameters. No user parameter section found\n");
            return GADGET_OK;
        }
        
        //num_interleaves = h.encodingLimits.kspace_encode_step_1.maximum+1;		
		
		return GADGET_OK;
	}

	int RetroCineGadget::process(GadgetContainerMessage<IsmrmrdReconData> *m1){
		IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
		
		for(size_t e = 0; e < recon_bit_->rbit_.size(); e++){
			hoNDArray<std::complex<float>> data_array = recon_bit_->rbit_[e].data_.data_;
			hoNDArray<float> traj_array = *recon_bit_->rbit_[e].data_.trajectory_;
			
			//bitreversed_order = {0 8 4 12 2 10 6 14 1 9 5 13 3 11 6 15}
			std::vector<int> bitreversed_order = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 12, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 17, 7, 23, 13, 31};

			size_t R0 = data_array.get_size(0);
			size_t E1 = data_array.get_size(1);
			size_t E2 = data_array.get_size(2);
			size_t CHA = data_array.get_size(3);
			size_t N = data_array.get_size(4);
			size_t S = data_array.get_size(5);
			size_t SLC = data_array.get_size(6);
			
			if(data_out.get_number_of_elements() < R0){
				data_out.create(R0, E1, E2, CHA, num_phases, S, SLC);
				traj_out.create(3, R0, E1, E2, 1, S, SLC);
				headers_out.create(E1,E2,num_phases,S,SLC);
			}
			
			//Setup recon times array
			std::vector<float> recon_times_ratio;
			std::vector<float> cpt_ratio;
			for(size_t i=0; i<num_phases; i++){
				recon_times_ratio.push_back((float)(i+.5)/(num_phases+1));
			}
			
			ISMRMRD::AcquisitionHeader* acqhdr;
			
			//get k_lines
			std::vector<size_t> k_lines;
			for(size_t i=0; i<E1; i++){
				acqhdr = &recon_bit_->rbit_[e].data_.headers_(i,0,0,0,0);
				if(acqhdr->acquisition_time_stamp != 0){
					k_lines.push_back(i);
				}
			}
			
			//Iterate over interleaves
			for(size_t s=0; s<E1; s++){
				
				size_t k = k_lines[s];
				//Get physio times and convert to ratio
				float cpt_max = 0.0;
				for(size_t n=0; n<N; n++){
					acqhdr = &recon_bit_->rbit_[e].data_.headers_(k,0,n,0,0);
					if(acqhdr->physiology_time_stamp[0] > cpt_max){
						cpt_max = acqhdr->physiology_time_stamp[0];
					}
				}
				cpt_ratio.clear();
				for(size_t n=0; n<N; n++){
					acqhdr = &recon_bit_->rbit_[e].data_.headers_(k,0,n,0,0);
					if(acqhdr->acquisition_time_stamp != 0){
						cpt_ratio.push_back((acqhdr->physiology_time_stamp[0])/cpt_max);
					}else{
						cpt_ratio.push_back(-1);
					}					
				}
				
				//Iterate over recon times
				for(size_t p=0; p<num_phases; p++){
					
					int start_ind = -1;
					int end_ind = -1;
					float start_time_diff = 1;
					float end_time_diff = 1;
					float curr_recon_phase = recon_times_ratio[p];//+bitreversed_order[k]*.017/3;
					
					if(curr_recon_phase > 1.0) curr_recon_phase -= 1.0;
					
					//Iterate over phases
					for(size_t n=0; n<N; n++){	
					
						if(cpt_ratio[n] >= 0){
							acqhdr = &recon_bit_->rbit_[e].data_.headers_(k,0,n,0,0);
							//Find start and end interpolation index
							float curr_cpt_phase = cpt_ratio[n];
							if(curr_cpt_phase <= curr_recon_phase && std::abs(curr_cpt_phase-curr_recon_phase) < start_time_diff){
								start_ind = n;
								start_time_diff = std::abs(curr_cpt_phase-curr_recon_phase);
							}
							else if(curr_cpt_phase > curr_recon_phase && std::abs(curr_cpt_phase-curr_recon_phase) < end_time_diff){
								end_ind = n;
								end_time_diff = abs(curr_cpt_phase-curr_recon_phase);
							}
						}
					}
												
					if(start_ind < 0){
						start_ind = end_ind-1;
					}
					if(end_ind < 0){
						end_ind = start_ind+1;
					}
					
					std::cout << start_ind << std::endl;
					std::cout << end_ind << std::endl;
					std::cout << cpt_ratio[start_ind] << std::endl;
					std::cout << curr_recon_phase << std::endl;
					std::cout << cpt_ratio[end_ind] << std::endl;
					
					//Interpolate and copy to output data array
					for(size_t c=0; c<CHA; c++){
						std::complex<float>* data1 = &recon_bit_->rbit_[e].data_.data_(0,k,0,c,start_ind,0,0);
						std::complex<float>* data2 = &recon_bit_->rbit_[e].data_.data_(0,k,0,c,end_ind,0,0);
						hoNDArray<std::complex<float>> data_interp = Interpolate(data1,data2,cpt_ratio[start_ind],cpt_ratio[end_ind],curr_recon_phase,R0);
						memcpy(&data_out(0,k,0,c,p,0,0),&data_interp[0],sizeof(std::complex<float>)*R0);
					}
					for(size_t h=0; h<num_phases; h++){
						acqhdr->idx.phase = h;
						headers_out(k,0,h,0,0) = *acqhdr;
					}
					
				}
				
				auto traj = *recon_bit_->rbit_[e].data_.trajectory_;
				memcpy(&traj_out(0,0,k,0,0,0,0),&traj[3*R0*k],sizeof(float)*R0*3);	
				acqhdr = &recon_bit_->rbit_[e].data_.headers_(k,0,0,0,0);
				acqhdr = &recon_bit_->rbit_[e].data_.headers_(k,0,0,0,0);					
				
			}

			//Pass data downstream
			if(acqhdr->idx.kspace_encode_step_1+1 == E1){// || acqhdr->idx.kspace_encode_step_1 == 0){
				recon_bit_->rbit_[e].data_.data_ = data_out;
				//write_nd_array(&data_out, "data_out.cplx");
				recon_bit_->rbit_[e].data_.trajectory_ = traj_out;
				//write_nd_array(&traj_out, "traj_out.cplx");
				recon_bit_->rbit_[e].data_.headers_ = headers_out;
				if(this->next()->putq(m1) < 0){
					GDEBUG("Failed to put job on queue. \n");
					m1->release();
					return GADGET_FAIL;
				}
			}
		}

		return GADGET_OK;
	}
	
	hoNDArray<std::complex<float>> RetroCineGadget::Interpolate(std::complex<float>* data1, std::complex<float>* data2, float start_phase, float end_phase, float interp_phase, size_t samples){
		if(interp_phase > end_phase){
			end_phase = end_phase+1;
		}
		if(interp_phase < start_phase){
			start_phase = start_phase-1;
		}
		float start_end = end_phase-start_phase;
		float start_interp = interp_phase-start_phase;
		hoNDArray<std::complex<float>> y(samples);
		std::complex<float> y0;
		std::complex<float> y1;
		for(size_t ii=0; ii<samples; ii++){
			y0 = data1[ii];
			y1 = data2[ii];
			y[ii] = data1[ii]+(start_interp)*(y1-y0)/(start_end);
		}
		return y;
	}

	GADGET_FACTORY_DECLARE(RetroCineGadget);
}
