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

#if defined GADGETRON_COMPRESSION_ZFP
#include "zfp/zfp.h"
#endif


class GadgetronClientConnector
{

public:

	#if defined GADGETRON_COMPRESSION_ZFP
	size_t compress_zfp_tolerance(float* in, size_t samples, size_t coils, double tolerance, char* buffer, size_t buf_size)
	{
		zfp_type type = zfp_type_float;
		zfp_field* field = NULL;
		zfp_stream* zfp = NULL;
		bitstream* stream = NULL;
		size_t zfpsize = 0;

		zfp = zfp_stream_open(NULL);
		field = zfp_field_alloc();

		zfp_field_set_pointer(field, in);

		zfp_field_set_type(field, type);
		zfp_field_set_size_2d(field, samples, coils);
		zfp_stream_set_accuracy(zfp, tolerance, type);

		if (zfp_stream_maximum_size(zfp, field) > buf_size) {
		    zfp_field_free(field);
		    zfp_stream_close(zfp);
		    stream_close(stream);
		    throw std::runtime_error("Insufficient buffer space for compression");
		}

		stream = stream_open(buffer, buf_size);
		if (!stream) {
		    zfp_field_free(field);
		    zfp_stream_close(zfp);
		    stream_close(stream);
		    throw std::runtime_error("Cannot open compressed stream");
		}
		zfp_stream_set_bit_stream(zfp, stream);

		if (!zfp_write_header(zfp, field, ZFP_HEADER_FULL)) {
		    zfp_field_free(field);
		    zfp_stream_close(zfp);
		    stream_close(stream);
		    throw std::runtime_error("Unable to write compression header to stream");
		}

		zfpsize = zfp_compress(zfp, field);
		if (zfpsize == 0) {
		    zfp_field_free(field);
		    zfp_stream_close(zfp);
		    stream_close(stream);
		    throw std::runtime_error("Compression failed");
		}

		zfp_field_free(field);
		zfp_stream_close(zfp);
		stream_close(stream);
		return zfpsize;
	}
	
	size_t decompress_zfp(char* comp_buffer, float* data_out, size_t samples, size_t coils, size_t comp_size){
		    zfp_type type = zfp_type_float;
            zfp_field* field = NULL;
            zfp_stream* zfp = NULL;
            bitstream* cstream = NULL;
            size_t zfpsize = comp_size;
            
            zfp = zfp_stream_open(NULL);
            field = zfp_field_alloc();
            
            cstream = stream_open(comp_buffer, comp_size);
            if (!cstream) {
                zfp_field_free(field);
                zfp_stream_close(zfp);
                stream_close(cstream);            
                return 0;
            }
            zfp_stream_set_bit_stream(zfp, cstream);
            
            zfp_stream_rewind(zfp);
            if (!zfp_read_header(zfp, field, ZFP_HEADER_FULL)) {
                zfp_field_free(field);
                zfp_stream_close(zfp);
                stream_close(cstream);            
                return 0;
            }
            
            size_t nx = std::max(field->nx, 1u);
            size_t ny = std::max(field->ny, 1u);
            size_t nz = std::max(field->nz, 1u);
            
            if (nx*ny*nz != (samples*coils)) {
                zfp_field_free(field);
                zfp_stream_close(zfp);
                stream_close(cstream);            
                return 0;                
            }
            zfp_field_set_pointer(field, data_out);
            
            if (!zfp_decompress(zfp, field)) {          
                zfp_field_free(field);
                zfp_stream_close(zfp);
                stream_close(cstream);            
                return 0;                
            }
        
            zfp_field_free(field);
            zfp_stream_close(zfp);
            stream_close(cstream);            
	}
	#endif
};

int main(int argc, char** argv)
{
	//read in parameters
	bool open_input_file = true;
	po::options_description desc("Allowed options");
	std::string in_filename;
	desc.add_options()
 	 	("filename,f", po::value<std::string>(&in_filename), "Input file")
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

	ISMRMRD::Acquisition acq;

	std::vector<float> CR(acquisitions);
	std::vector<float> CR_ZFP(acquisitions);
	std::vector<float> std_error(acquisitions);
	std::vector<float> std_error_zfp(acquisitions);

	float SNR_Loss = .1;

	float sigma_noise = 9.16e-6/std::sqrt(2)*std::sqrt(5/3.9)*.793;
	for (uint32_t ii = 0; ii < acquisitions; ii++) {
		{			  
		boost::mutex::scoped_lock scoped_lock(mtx);
		ismrmrd_dataset->readAcquisition(ii, acq);
		}

		std::vector<float> data_in((float*)&acq.getDataPtr()[0], (float*)&acq.getDataPtr()[0]+acq.getHead().active_channels*acq.getHead().number_of_samples*2);
		std::vector<float> data_out(data_in.size());
		std::vector<float> diff(data_in.size());

		//NHLBI Compression/Decompression
		float local_tolerance = sigma_noise*std::sqrt(3)*std::sqrt(1/std::pow((1-SNR_Loss),2)-1);
		
		//Segmented NHLBI compression buffers
		int N = acq.getHead().active_channels*acq.getHead().number_of_samples*2;
		std::vector<uint8_t> serialized_buffer;
		int segments = acq.getHead().active_channels*4;	
		int segment_size = N/segments;
		for(int ch = 0; ch < segments; ch ++){
	        std::vector<float> input_data((float*)&acq.getDataPtr()[0]+ch*segment_size, (float*)&acq.getDataPtr()[0]+(ch+1)*segment_size);
	        CompressedBuffer<float> comp_buffer(input_data, local_tolerance);
			std::vector<uint8_t> serialized = comp_buffer.serialize();
			if(ch == 0){serialized_buffer = serialized;}
			else{serialized_buffer.insert(serialized_buffer.end(), serialized.begin(), serialized.end());}
		}
		float compression_ratio = (float(N)*sizeof(float))/serialized_buffer.size();
		size_t bytes_needed = 0;
		size_t total_size = 0;
		CompressedBuffer<float> comp;
		bytes_needed = comp.deserialize(serialized_buffer);
		serialized_buffer.erase(serialized_buffer.begin(),serialized_buffer.begin()+bytes_needed);
        for (size_t i = 0; i < comp.size(); i++) {
            data_out[i] = comp[i];
        }
		total_size += comp.size();
		while(serialized_buffer.size()>0){
            CompressedBuffer<float> comp;
            bytes_needed = comp.deserialize(serialized_buffer);
			serialized_buffer.erase(serialized_buffer.begin(),serialized_buffer.begin()+bytes_needed);
			//float* d_ptr = (float*)m2->getObjectPtr()->get_data_ptr()+ch*samples;
            for (size_t i = 0; i < comp.size(); i++) {
                data_out[i+total_size] = comp[i]; //This uncompresses sample by sample into the uncompressed array
            }
			total_size += comp.size();
		}
		for(int i = 0; i < diff.size(); i++){
			diff[i] = data_in[i]-data_out[i];
		}
		float mean = std::accumulate(diff.begin(), diff.end(), 0.0)/diff.size();
		float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		float stdev = std::sqrt(sq_sum / diff.size() - mean * mean);
		CR[ii] = compression_ratio;
		std_error[ii] = stdev;

		//ZFP Compression/Decompression
		local_tolerance = 11.64*sigma_noise*std::sqrt(1/std::pow((1-SNR_Loss),2)-1);
		GadgetronClientConnector con;
		size_t comp_buffer_size = 4*sizeof(float)*acq.getHead().active_channels*acq.getHead().number_of_samples;
		char* comp_buffer_zfp = new char[comp_buffer_size];
		size_t compressed_size = 0;
		compressed_size = con.compress_zfp_tolerance((float*)&acq.getDataPtr()[0],
                                                         acq.getHead().number_of_samples*2, acq.getHead().active_channels,
                                                         local_tolerance, comp_buffer_zfp, comp_buffer_size);
		compression_ratio = sizeof(float)*float(acq.getHead().active_channels*acq.getHead().number_of_samples*2)/float(compressed_size);
		std::cout << compression_ratio << std::endl;
		con.decompress_zfp(comp_buffer_zfp, &data_out[0], acq.getHead().number_of_samples*2, acq.getHead().active_channels, comp_buffer_size);
		delete [] comp_buffer_zfp;
		for(int i = 0; i < diff.size(); i++){
			diff[i] = data_in[i]-data_out[i];
		}
		mean = std::accumulate(diff.begin(), diff.end(), 0.0)/diff.size();
		sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		stdev = std::sqrt(sq_sum / diff.size() - mean * mean);
		CR_ZFP[ii] = compression_ratio;
		std_error_zfp[ii] = stdev;
	}

	hoNDArray<float> Out;
	Out.create(acquisitions,4);
	memcpy(Out.get_data_ptr(), &CR[0], sizeof(float)*acquisitions);
	memcpy(Out.get_data_ptr()+acquisitions, &std_error[0], sizeof(float)*acquisitions);
	memcpy(Out.get_data_ptr()+2*acquisitions, &CR_ZFP[0], sizeof(float)*acquisitions);
	memcpy(Out.get_data_ptr()+3*acquisitions, &std_error_zfp[0], sizeof(float)*acquisitions);
	write_nd_array<float>(&Out, "CR_Array.real");

}




































