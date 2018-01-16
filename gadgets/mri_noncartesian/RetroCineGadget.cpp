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
		return GADGET_OK;
	}

	int RetroCineGadget::process(GadgetContainerMessage<IsmrmrdReconData> *m1){
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
			
			//Setup recon times array
			float recon_times_ratio [num_phases];
			for(size_t i=0, i<num_phases; i++){
				recon_times_ratio[i] = (i+.5)/(num_phase+1);
			}
			
			ISMRMRD::AcquisitionHeader* acqhdr;
			for(size_t s=0; s<num_segments; s++){
				//Get physio times and convert to ratio
				acqhdr = &recon_bit_->rbit_[e].data_.headers_(0,s,0,0,N-1,0,0);
				cpt_max = acqhdr->acquisition_time_stamp[0];
				acqhdr = &recon_bit_->rbit_[e].data_.headers_(0,s,0,0,0,0,0);
				cpt_max -= acqhdr->acquisition_time_stamp[0];
				for(size_t p=0; p<num_phases; p++){
					acqhdr = &recon_bit_->rbit_[e].data_.headers_(0,s,0,0,p,0,0);
					cpt_ratio = acqhdr->physiology_time_stamp[0]/cpt_max;
					
					
					
				}
			}

			//ISMRMRD::AcquisitionHeader* acqhdr = &recon_bit_->rbit_[e].data_.headers_(0,0,0,0,0);

			if(this->next()->putq(m1) < 0){
				GDEBUG("Failed to put job on queue. \n");
				m1->release();
				//m2->release();
				return GADGET_FAIL;
			}

		return GADGET_OK;
	}

	GADGET_FACTORY_DECLARE(RetroCineGadget);
}
