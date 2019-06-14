/*
 * PseudoReplicator.h

 *
 *  Created on: Jun 23, 2015
 *      Author: David Hansen
 */
#pragma once
#include "Gadget.h"
#include "mri_core_data.h"
#include "gadgetron_mricore_export.h"

namespace Gadgetron {

class EXPORTGADGETSMRICORE PseudoReplicatorGadget : public Gadget1<IsmrmrdReconData>{
public:
	GADGET_PROPERTY(repetitions,int,"Number of pseudoreplicas to produce",10);
	GADGET_PROPERTY(slice,int,"Slice to replicate",0);
	PseudoReplicatorGadget()  ;
	virtual ~PseudoReplicatorGadget();

	virtual int process_config(ACE_Message_Block *);
	virtual int process(GadgetContainerMessage<IsmrmrdReconData>*);

private:
	int repetitions_;
	int slice_;
};

} /* namespace Gadgetron */
