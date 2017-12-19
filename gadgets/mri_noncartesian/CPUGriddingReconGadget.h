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
#include "cuNFFT.h"

namespace Gadgetron{

	class EXPORTGADGETSMRINONCARTESIAN CPUGriddingReconGadget : public GenericReconGadget {

	/**
		CPUGriddingReconGadget class declaration 
		----------------------------------------

		interface similar to GriddingReconGadget
		
		Note: can probably be combined and templated
	*/
	
	public:

		GADGET_DECLARE(CPUGriddingReconGadget);

		CPUGriddingReconGadget();

		~CPUGriddingReconGadget();
	
	protected:
		
		/**
			Gadget properties declaration
		*/

		GADGET_PROPERTY(kernelWidthProperty, float, "Kernel width", 5.5);
		GADGET_PROPERTY(oversamplingFactorProperty, float, "Oversmapling factor", 2);
		GADGET_PROPERTY(iterate, bool, "Iterate bool", false);
		GADGET_PROPERTY(iteration_max, int, "Maximum Iterations", 10);
		GADGET_PROPERTY(iteration_tol, float, "Iteratation Tolerance", 1e-3);

		/**
			Storage for the properties above
		*/

		float kernelWidth;
		float oversamplingFactor;
		double kr_max;
		std::vector<hoNFFT_plan<float, 2>> plans_;
		bool ref_plan_prepared;
		cuNFFT_plan<float,2> nfft_plan_;
		//hoNDArray<std::complex<float>>csm_host_;

		/**
			Image dimensions
		*/

		std::vector<size_t> imageDims;
		std::vector<size_t> imageDimsOs;

		virtual int process_config(ACE_Message_Block *mb);
		virtual int process(GadgetContainerMessage <IsmrmrdReconData> *m1);

		/**
			Reconstruct multi channel data

			/param data: multi-channel k-space data
			/param traj: trajectories
			/param dcw: density compensation
			/param nCoils: number of channels
		*/

		boost::shared_ptr<hoNDArray<std::complex<float>>> reconstruct(
			hoNDArray<std::complex<float>> *data,
			hoNDArray<std::complex<float>> coilMap,
			hoNDArray<floatd2> *traj,
			hoNDArray<float> *dcw,
			size_t nCoils
		);

		/**
			Reconstruct single channel

			/param data: single channel data
			/param traj: trajectory
			/param dcw: density compensation
		*/

		boost::shared_ptr<hoNDArray<std::complex<float>>> reconstructChannel(
			hoNDArray<std::complex<float>> *data,
			hoNDArray<floatd2> *traj,
			hoNDArray<float> *dcw,
			hoNFFT_plan<float, 2> plan
		);

		/**
			Helper method for seperating the trajectory
			and the density compesnation
		*/

		std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>> 
		separateDcwAndTraj(
			hoNDArray<float> *dcwTraj
		);

		std::tuple<hoNDArray<std::complex<float>>, hoNDArray<std::complex<float>>> computeCsm(
			IsmrmrdDataBuffered *ref_
		);
	};
}
