#include "CPUGridding_PrepRef.h"
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
	CPUGridding_PrepRef::CPUGridding_PrepRef(){}

	CPUGridding_PrepRef::~CPUGridding_PrepRef(){}

	int CPUGridding_PrepRef::process_config(ACE_Message_Block *mb){
		GADGET_CHECK_RETURN(GenericReconGadget::process_config(mb) == GADGET_OK, GADGET_FAIL);
		ISMRMRD::IsmrmrdHeader h;
		deserialize(mb->rd_ptr(), h);
		acceleration_factor = h.encoding[0].parallelImaging->accelerationFactor.kspace_encoding_step_1;
		if(acceleration_factor != 10){ acceleration_factor = 10; }
		process_called_times_ = 0;
		prepared_ = false;
		return GADGET_OK;
	}

	int CPUGridding_PrepRef::process(GadgetContainerMessage<IsmrmrdReconData> *m1){
		IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
		
		for(size_t e = 0; e < recon_bit_->rbit_.size(); e++){
			hoNDArray<std::complex<float>> data_array = recon_bit_->rbit_[e].data_.data_;
			hoNDArray<float> traj_array = *recon_bit_->rbit_[e].data_.trajectory_;

			size_t R0 = data_array.get_size(0);
			size_t E1 = data_array.get_size(1);
			size_t E2 = data_array.get_size(2);
			size_t CHA = data_array.get_size(3);
			size_t N = data_array.get_size(4);
			size_t S = data_array.get_size(5);
			size_t SLC = data_array.get_size(6);

			ISMRMRD::AcquisitionHeader* acqhdr = &recon_bit_->rbit_[e].data_.headers_(0,0,0,0,0);
			if(acqhdr->idx.set == 0){
				process_called_times_++;

				if(process_called_times_ == 1){
					buffer_data.create(R0, E2, CHA, N, S, SLC, E1);
					buffer_data.fill(0);
					buffer_traj.create(3, R0, E2, N, S, SLC, E1);
					buffer_traj.fill(0);
					//clear(buffer.trajectory_.get_ptr());
				}

				std::vector<size_t> newOrder = {0, 2, 3, 4, 5, 6, 1};
				data_array = *permute(&data_array,&newOrder);
				newOrder = {0, 1, 3, 4, 5, 6, 2};
				traj_array = *permute(&traj_array,&newOrder);

				for(int l = 0; l < E1; l++){
					ISMRMRD::AcquisitionHeader* acqhdr2 = &recon_bit_->rbit_[e].data_.headers_(l,0,0,0,0);
					memcpy(&buffer_data(0,0,0,0,0,0,acqhdr2->idx.kspace_encode_step_1),&data_array(0,0,0,0,0,0,l),sizeof(std::complex<float>)*R0*E2*CHA*N*S*SLC);
					memcpy(&buffer_traj(0,0,0,0,0,0,acqhdr2->idx.kspace_encode_step_1),&traj_array(0,0,0,0,0,0,l),sizeof(float)*3*R0*E2*N*S*SLC);
					//memcpy(&buffer_data(0,0,0,0,0,0,process_called_times_%acceleration_factor),&data_array(0,0,0,0,0,0,l),sizeof(std::complex<float>)*R0*E2*CHA*N*S*SLC);
					//memcpy(&buffer_traj(0,0,0,0,0,0,process_called_times_%acceleration_factor),&traj_array(0,0,0,0,0,0,l),sizeof(float)*3*R0*E2*N*S*SLC);
				}
				write_nd_array(&buffer_data,"bufferdata.cplx");
				write_nd_array(&buffer_traj,"buffertraj.real");

				if(process_called_times_ > acceleration_factor){
					prepared_ = true;
				}
			}
		
			if(prepared_){
				IsmrmrdDataBuffered ref_;
				ref_.data_.create(R0, acceleration_factor*E1, E2, CHA, N, S, SLC);
				ref_.trajectory_ = hoNDArray<float>(3, R0, acceleration_factor*E1, E2, N, S, SLC);
				std::vector<size_t> newOrder = {0, 6, 1, 2, 3, 4, 5};
				ref_.data_ = *permute(&buffer_data,&newOrder);
				newOrder = {0, 1, 6, 2, 3, 4, 5};
				ref_.trajectory_ = *permute(&buffer_traj,&newOrder);
				std::cout << (*ref_.trajectory_).get_size(2) << std::endl;
				recon_bit_->rbit_[e].ref_ = ref_;
			}
		}

		if(prepared_){
			//GadgetContainerMessage<IsmrmrdReconData>* m2 = new GadgetContainerMessage<IsmrmrdReconData>(recon_bit_);
			if(this->next()->putq(m1) < 0){
				GDEBUG("Failed to put job on queue. \n");
				m1->release();
				//m2->release();
				return GADGET_FAIL;
			}
			//m2->release();
		}
		//m1->release();
		return GADGET_OK;
	}

	GADGET_FACTORY_DECLARE(CPUGridding_PrepRef);
}
