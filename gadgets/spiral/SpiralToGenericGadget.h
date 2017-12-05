#ifndef SpiralToGenericGadget_H
#define SpiralToGenericGadget_H
#pragma once

#include "gadgetron_spiral_export.h"
#include "Gadget.h"
#include "GadgetMRIHeaders.h"
#include "hoNDArray.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>
#include <boost/shared_ptr.hpp>

namespace Gadgetron{

  class EXPORTGADGETS_SPIRAL SpiralToGenericGadget :
    public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
  {

  public:
    GADGET_DECLARE(SpiralToGenericGadget);

    SpiralToGenericGadget();
    virtual ~SpiralToGenericGadget();

  protected:
	
	GADGET_PROPERTY(vdsFactor, double, "vds FOV reduction factor", 0.0);

    virtual int process_config(ACE_Message_Block* mb);
    
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
			GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);
    
  private:
    int samples_to_skip_start_;
    int samples_to_skip_end_;
    int samples_per_interleave_;
    int interleaves_;
    long    Tsamp_ns_;
    long    Nints_;
    long    acceleration_factor_;
    double  gmax_;
    double  smax_;
    double  krmax_;
    double  fov_[2];
    bool prepared_;
    
    boost::shared_ptr< hoNDArray<float> > host_traj_;
  };
}
#endif //SpiralToGenericGadget_H
