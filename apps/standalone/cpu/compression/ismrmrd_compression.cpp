#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/dataset.h>
#include <ismrmrd/meta.h>
#include <ismrmrd/xml.h>
#include "NHLBICompression.h"
#include "hoNDArray.h"
#include "hoNDArray_fileio.h"

using namespace std;
using namespace Gadgetron;

boost::mutex mtx;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
	//read in parameters
	bool open_input_file = true;
	po::options_description desc("Allowed options");
	std::string in_filename;
	std::string out_filename;
	std::string out2_filename;
	desc.add_options()
 	 	("filename,f", po::value<std::string>(&in_filename), "Input file")
		("outfile,o", po::value<std::string>(&out_filename)->default_value("out.h5"), "Output file")
		("outfile2,p", po::value<std::string>(&out2_filename)->default_value("out2.h5"), "Output2 file")
	;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //Let's open the input file
    boost::shared_ptr<ISMRMRD::Dataset> ismrmrd_dataset;
    std::string xml_config;
    if (open_input_file) {
      ismrmrd_dataset = boost::shared_ptr<ISMRMRD::Dataset>(new ISMRMRD::Dataset(in_filename.c_str(), "/dataset", false));
      // Read the header
      ismrmrd_dataset->readHeader(xml_config);
    }

	uint32_t acquisitions = 0;
	{
	mtx.lock();
	acquisitions = ismrmrd_dataset->getNumberOfAcquisitions();
	mtx.unlock();
	}
	std::cout << "contains " << acquisitions << " acquisitions" << std::endl;

	ISMRMRD::Dataset d(out_filename.c_str(),"dataset", true);
	ISMRMRD::Acquisition acq_tmp;
	hoNDArray<float> data1;
	hoNDArray<float> data2;

	float local_tolerance = 4e-6*.25;
	for (uint32_t i = 0; i < acquisitions; i++) {
		{			  
		boost::mutex::scoped_lock scoped_lock(mtx);
		ismrmrd_dataset->readAcquisition(i, acq_tmp);
		}

				std::random_device generator;
				std::normal_distribution<float> distribution(0.0,4e-6);
				
				float* dptr = (float*)acq_tmp.getDataPtr();
				for(int i = 0; i < acq_tmp.getHead().number_of_samples*acq_tmp.getHead().active_channels*2; i++){
					dptr[i] += distribution(generator);
				}

		if(i == 0){
			data1.create(acquisitions,2*acq_tmp.getHead().number_of_samples*acq_tmp.getHead().active_channels);
		}
		memcpy(data1.get_data_ptr()+i*2*acq_tmp.getHead().number_of_samples*acq_tmp.getHead().active_channels,acq_tmp.getDataPtr(),2*acq_tmp.getHead().number_of_samples*acq_tmp.getHead().active_channels*sizeof(float));
		d.appendAcquisition(acq_tmp);
		//std::cout << "compression ratio = " << float(N*sizeof(float))/float(serialized_buffer.size()) << std::endl;
	}
	d.writeHeader(xml_config);

/*

    if (open_input_file) {
      ismrmrd_dataset = boost::shared_ptr<ISMRMRD::Dataset>(new ISMRMRD::Dataset(out_filename.c_str(), "/dataset", false));
      // Read the header
      ismrmrd_dataset->readHeader(xml_config);
    }

	//uint32_t acquisitions = 0;
	{
	mtx.lock();
	acquisitions = ismrmrd_dataset->getNumberOfAcquisitions();
	mtx.unlock();
	}
	std::cout << "contains " << acquisitions << " acquisitions" << std::endl;

	//ISMRMRD::Acquisition acq_tmp;
	ISMRMRD::Dataset d2(out2_filename.c_str(),"dataset", true);

	//float local_tolerance = 4e-6*.25;
	for (uint32_t i = 0; i < acquisitions; i++) {
		{			  
		boost::mutex::scoped_lock scoped_lock(mtx);
		ismrmrd_dataset->readAcquisition(i, acq_tmp);
		}
		if(i == 0){
			data2.create(acquisitions,2*acq_tmp.getHead().number_of_samples*acq_tmp.getHead().active_channels);
		}
		memcpy(data2.get_data_ptr()+i*2*acq_tmp.getHead().number_of_samples*acq_tmp.getHead().active_channels,acq_tmp.getDataPtr(),2*acq_tmp.getHead().number_of_samples*acq_tmp.getHead().active_channels*sizeof(float));
		d2.appendAcquisition(acq_tmp);
		//std::cout << acq_tmp.data[0] << std::endl;
		//std::cout << "compression ratio = " << float(N*sizeof(float))/float(serialized_buffer.size()) << std::endl;
	}
	d2.writeHeader(xml_config);
*/

}

	















