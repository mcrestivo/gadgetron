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

	class EXPORTGADGETSMRINONCARTESIAN CPUGridding_PrepRef : public GenericReconGadget {

	/**
		CPUGriddingReconGadget class declaration 
		----------------------------------------

		interface similar to GriddingReconGadget
		
		Note: can probably be combined and templated
	*/
	
	public:

		GADGET_DECLARE(CPUGridding_PrepRef);

		CPUGridding_PrepRef();

		~CPUGridding_PrepRef();
	
	protected:
		
		/**
			Gadget properties declaration
		*/

		//GADGET_PROPERTY(kernelWidthProperty, float, "Kernel width", 5.5);

		/**
			Storage for the properties above
		*/


		/**
			Image dimensions
		*/
		hoNDArray<float> buffer_traj;
		hoNDArray<std::complex<float>> buffer_data;
		unsigned short acceleration_factor;
		unsigned short process_called_times_;
		bool prepared_;

		virtual int process_config(ACE_Message_Block *mb);
		virtual int process(GadgetContainerMessage <IsmrmrdReconData> *m1);

	};
}
