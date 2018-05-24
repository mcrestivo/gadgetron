#include "GadgetIsmrmrdReadWrite.h"
#include "CombineGadget.h"
#include "mri_core_coil_map_estimation.h"
#include "hoNDArray_fileio.h"
#include <boost/thread/mutex.hpp>

namespace Gadgetron{

  CombineGadget::CombineGadget() {}
  CombineGadget::~CombineGadget() {}

int CombineGadget::
process( GadgetContainerMessage<ISMRMRD::ImageHeader>* m1,
	 GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{

  // Get the dimensions
  size_t nx = m2->getObjectPtr()->get_size(0);
  size_t ny = m2->getObjectPtr()->get_size(1);
  size_t nz = m2->getObjectPtr()->get_size(2);
  size_t nc = m2->getObjectPtr()->get_size(3);

  std::vector<size_t> dimensions(3);
  dimensions[0] = nx;
  dimensions[1] = ny; 
  dimensions[2] = nz;

  std::vector<size_t> img_dimensions(3);
  img_dimensions[0] = nx;
  img_dimensions[1] = ny; 
  img_dimensions[2] = nc;

  /*try{m3->getObjectPtr()->create(&dimensions);}
  catch (std::runtime_error &err){
  	GEXCEPTION(err,"CombineGadget, failed to allocate new array\n");
    return -1;
  }*/

    hoNDArray<std::complex<float>> d1 = *m2->getObjectPtr();
    hoNDArray<std::complex<float>> d2;
    d1.reshape(&img_dimensions);
    d2.create(&dimensions);

    size_t img_block = nx*ny*nz;

    /*  for (size_t z = 0; z < nz; z++) {
    for (size_t y = 0; y < ny; y++) {
    for (size_t x = 0; x < nx; x++) {
    float mag = 0;
    float phase = 0;
    size_t offset = z*ny*nx+y*nx+x;
    for (size_t c = 0; c < nc; c++) {
    float mag_tmp = norm(d1[offset + c*img_block]);
    phase += mag_tmp*arg(d1[offset + c*img_block]);
    mag += mag_tmp;
    }
    d2[offset] = std::polar(std::sqrt(mag),phase);
    }
    }
    }*/

    //hoNDArray<std::complex<float> > chunk = hoNDArray<std::complex<float> >(dimensions, &d1(0,0,0,0,n,s,loc));
    hoNDArray<std::complex<float> > coil_map = hoNDArray<std::complex<float> >(img_dimensions);
    //Zero out the output
    //clear(output);

    size_t ks = 4;
    size_t power = 3;

    Gadgetron::coil_map_2d_Inati(d1, coil_map, ks, power);
    Gadgetron::coil_combine(d1, coil_map, 2, d2);
    //write_nd_array(&d1, "d1.cplx");
    //write_nd_array(&d2, "d2.cplx");

    // Create a new message with an hoNDArray for the combined image
    GadgetContainerMessage< hoNDArray<std::complex<float> > >* m3 = new GadgetContainerMessage< hoNDArray<std::complex<float> > >(d2);


    // Modify header to match the size and change the type to real
    m1->getObjectPtr()->channels = 1;

    // Now add the new array to the outcgoing message
    m1->cont(m3);

    // Release the old data
    m2->release();      

  return this->next()->putq(m1);
}

GADGET_FACTORY_DECLARE(CombineGadget)
}
