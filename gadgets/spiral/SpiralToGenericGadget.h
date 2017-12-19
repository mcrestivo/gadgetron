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
	
	GADGET_PROPERTY(vdsFactor, double, "vds FOV reduction factor, 1.0 is no vds", 1.0);
	GADGET_PROPERTY(radialProfiles, float, "number of radial profiles, if radial", 201.0);
	GADGET_PROPERTY(goldenAngle, bool, "Use golden angle for calculating trajectories?", false);



    virtual int process_config(ACE_Message_Block* mb);
    
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
			GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);
			
	bool checkAttachedTrajectories(hoNDArray<float> *traj);
    
  private:
    int samples_to_skip_start_;
    int samples_to_skip_end_;
    int samples_per_interleave_;
    int num_profiles_per_frame_;
    int num_frames_;
    int interleaves_;
    int rep_;
    long    Tsamp_ns_;
    long    Nints_;
    long    acceleration_factor_;
    double  gmax_;
    double  smax_;
    double  krmax_;
    double  fov_[2];
    bool prepared_;
    bool radial;
    bool golden_angle;
    
    boost::shared_ptr< hoNDArray<float> > host_traj_;
  };
}
#endif //SpiralToGenericGadget_H
