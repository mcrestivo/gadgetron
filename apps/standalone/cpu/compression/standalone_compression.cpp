#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/dataset.h>
#include <ismrmrd/meta.h>
#include <ismrmrd/xml.h>

#include <fstream>
#include <streambuf>
#include <time.h>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <exception>
#include <map>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <random>

#include "NHLBICompression.h"
#include "parameterparser.h"
#include "hoNDArray.h"
#include "hoNDArray_fileio.h"

#if defined GADGETRON_COMPRESSION_ZFP
#include "zfp/zfp.h"
#endif


using namespace std;
using namespace Gadgetron;

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
	//ParameterParser parms;
  	//parms.add_parameter( 'T', COMMAND_LINE_FLOAT,  1, "Tolerance", true, "0.1");
	//parms.parse_parameter_list(argc, argv);
	//float local_tolerance = (float) parms.get_parameter('T')->get_float_value();
	//std::cout << local_tolerance << std::endl;
	int L = 200;
	float max = 1;
	std::vector<float> CR(L);
	std::vector<float> CR_ZFP(L);
	std::vector<float> std_error(L);
	std::vector<float> std_error_zfp(L);

	for(int ii = 0; ii < L; ii++){
		float local_tolerance = float(ii+1)/(float(L)/max);
		std::cout << local_tolerance << std::endl;
		//Generate random distribution
		int32_t number_elements = 10*256*16;
		std::random_device generator;
		std::normal_distribution<float> distribution(0,.1);
		std::vector<float> data_in(number_elements);
		std::vector<float> data_out(number_elements);
		std::vector<float> data_out_zfp(number_elements);
		std::vector<float> diff(number_elements);
		for(int i = 0; i < data_in.size(); i++){
			data_in[i] = distribution(generator);
		}
	
		//NHLBI Compression/Decompression
		CompressedBuffer<float> comp_buffer(data_in, local_tolerance);
		std::vector<uint8_t> serialized = comp_buffer.serialize();
		CompressedBuffer<float> comp;
		int32_t bytes_needed = comp.deserialize(serialized);
		float compression_ratio = sizeof(float)*float(number_elements)/float(bytes_needed);
		std::cout << compression_ratio << std::endl;
		for (size_t i = 0; i < comp.size(); i++) {
		    data_out[i] = comp[i];
		}

		for(int i = 0; i < diff.size(); i++){
			diff[i] = data_in[i]-data_out[i];
		}

		float mean = std::accumulate(diff.begin(), diff.end(), 0.0)/diff.size();
		float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		float stdev = std::sqrt(sq_sum / diff.size() - mean * mean);
		CR[ii] = compression_ratio;
		std_error[ii] = stdev;
		//std::cout << "mean error = " << mean << std::endl;
		//std::cout << "std error = " << stdev << std::endl;
		//std::cout << "total noise = " << std::sqrt(1+stdev*stdev) << std::endl;

		//ZFP Compression/Decompression
		local_tolerance = 11.588*local_tolerance*1e-6;
		GadgetronClientConnector con;
		size_t comp_buffer_size = 4*sizeof(float)*number_elements;
		char* comp_buffer_zfp = new char[comp_buffer_size];
		size_t compressed_size = 0;
		compressed_size = con.compress_zfp_tolerance(&data_in[0], number_elements/16, 16, local_tolerance, comp_buffer_zfp, comp_buffer_size);
		compression_ratio = sizeof(float)*float(number_elements)/float(compressed_size);
		std::cout << compression_ratio << std::endl;
		con.decompress_zfp(comp_buffer_zfp, &data_out_zfp[0], number_elements/16, 16, comp_buffer_size);
		delete [] comp_buffer_zfp;

		for(int i = 0; i < diff.size(); i++){
			diff[i] = data_in[i]-data_out_zfp[i];
		}
	
		mean = std::accumulate(diff.begin(), diff.end(), 0.0)/diff.size();
		sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		stdev = std::sqrt(sq_sum / diff.size() - mean * mean);
		CR_ZFP[ii] = compression_ratio;
		std_error_zfp[ii] = stdev;

		//std::cout << "mean error = " << mean << std::endl;
		//std::cout << "std error = " << stdev << std::endl;
		//std::cout << "total noise = " << std::sqrt(1+stdev*stdev) << std::endl;
	}

	hoNDArray<float> Out;
	Out.create(L,4);
	memcpy(Out.get_data_ptr(), &CR[0], sizeof(float)*L);
	memcpy(Out.get_data_ptr()+L, &std_error[0], sizeof(float)*L);
	memcpy(Out.get_data_ptr()+2*L, &CR_ZFP[0], sizeof(float)*L);
	memcpy(Out.get_data_ptr()+3*L, &std_error_zfp[0], sizeof(float)*L);
	write_nd_array<float>(&Out, "CR_Array_zeromean.real");

}




































