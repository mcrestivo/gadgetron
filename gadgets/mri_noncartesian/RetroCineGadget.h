/**
	\brief CPU Gridding reconstruction gadget

	Handles reconstruction of 2D float data with 
	density compensation provided. Iterative reconstruction 
	can be easily integreated
*/

#pragma once 
#include "GenericReconGadget.h"
#include "gadgetron_mri_noncartesian_export.h"
#include "hoNDArray.h"
#include "hoNFFT.h"

namespace Gadgetron{

	class EXPORTGADGETSMRINONCARTESIAN RetroCineGadget : public GenericReconGadget {

	/**
		CPUGriddingReconGadget class declaration 
		----------------------------------------

		interface similar to GriddingReconGadget
		
		Note: can probably be combined and templated
	*/
	
	public:

		GADGET_DECLARE(RetroCineGadget);

		RetroCineGadget();

		~RetroCineGadget();
	
	protected:
		

		virtual int process_config(ACE_Message_Block *mb);
		virtual int process(GadgetContainerMessage <IsmrmrdReconData> *m1);

	};
}
