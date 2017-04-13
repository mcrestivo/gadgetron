#ifndef GADGETISMRMRDREADWRITE_H
#define GADGETISMRMRDREADWRITE_H

#include "GadgetMRIHeaders.h"
#include "GadgetContainerMessage.h"
#include "GadgetMessageInterface.h"
#include "hoNDArray.h"
#include "url_encode.h"
#include "gadgetron_mricore_export.h"
#include <ismrmrd/ismrmrd.h>
#include <ace/SOCK_Stream.h>
#include <ace/Task.h>
#include <complex>

#include "NHLBICompression.h"
#include "hoNDFFT.h"
#include "hoNDArray_fileio.h"
#include "fwht.h"
#include <fftw3.h>
#include <mutex>

#if defined GADGETRON_COMPRESSION_ZFP
#include <zfp/zfp.h>
#endif //GADGETRON_COMPRESSION_ZFP

namespace Gadgetron{

    class EXPORTGADGETSMRICORE GadgetIsmrmrdAcquisitionMessageWriter : public GadgetMessageWriter
    {

    public:

        virtual int write(ACE_SOCK_Stream* sock, ACE_Message_Block* mb)
        {
	  auto h = AsContainerMessage<ISMRMRD::AcquisitionHeader>(mb);

	  if (!h) {
	    GERROR("GadgetAcquisitionMessageWriter, invalid acquisition message objects");
	    return -1;
	  }

	  ssize_t send_cnt = 0;
	  
	  GadgetMessageIdentifier id;
	  id.id = GADGET_MESSAGE_ISMRMRD_ACQUISITION;

	  if ((send_cnt = sock->send_n (&id, sizeof(GadgetMessageIdentifier))) <= 0) {
	    GERROR("Unable to send acquisition message identifier\n");
	    return -1;
	  }

	  ISMRMRD::AcquisitionHeader* acqHead = h->getObjectPtr();
	  if ((send_cnt = sock->send_n (acqHead, sizeof(ISMRMRD::AcquisitionHeader))) <= 0) {
	    GERROR("Unable to send acquisition header\n");
	    return -1;
	  }

	  unsigned long trajectory_elements = acqHead->trajectory_dimensions*acqHead->number_of_samples;
	  unsigned long data_elements = acqHead->active_channels*acqHead->number_of_samples;

	  auto d = AsContainerMessage< hoNDArray<std::complex<float> > >(h->cont());
	  
	  if (trajectory_elements) {
	    auto t = AsContainerMessage< hoNDArray<float> >(h->cont());
	    if ((send_cnt = sock->send_n (t->getObjectPtr()->get_data_ptr(), sizeof(float)*trajectory_elements)) <= 0) {
	      GERROR("Unable to send acquisition trajectory elements\n");
	      return -1;
	    }
	  }

	  if (data_elements) {
	    if ((send_cnt = sock->send_n (d->getObjectPtr()->get_data_ptr(), 2*sizeof(float)*data_elements)) <= 0) {
	      GERROR("Unable to send acquisition data elements\n");
	      return -1;
	    }
	  }
	  
	  return 0;
        }
    };

    /**
    Default implementation of GadgetMessageReader for IsmrmrdAcquisition messages
    */
    class EXPORTGADGETSMRICORE GadgetIsmrmrdAcquisitionMessageReader : public GadgetMessageReader
    {

    public:
		Gadgetron::hoNDArray< std::complex<float> > tmp;
        GADGETRON_READER_DECLARE(GadgetIsmrmrdAcquisitionMessageReader);
		fftwf_plan p_bkw;
		float *in, *out;

        virtual ACE_Message_Block* read(ACE_SOCK_Stream* stream)
        {

            GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1 =
                new GadgetContainerMessage<ISMRMRD::AcquisitionHeader>();

            GadgetContainerMessage<hoNDArray< std::complex<float> > >* m2 =
                new GadgetContainerMessage< hoNDArray< std::complex<float> > >();

            m1->cont(m2);


            ssize_t recv_count = 0;

            if ((recv_count = stream->recv_n(m1->getObjectPtr(), sizeof(ISMRMRD::AcquisitionHeader))) <= 0) {
	      GERROR("GadgetIsmrmrdAcquisitionMessageReader, failed to read ISMRMRDACQ Header\n");
	      return 0;
            }

            if (m1->getObjectPtr()->trajectory_dimensions) {
                GadgetContainerMessage<hoNDArray< float > >* m3 =
                    new GadgetContainerMessage< hoNDArray< float > >();

                m2->cont(m3);

                std::vector<size_t> tdims;
                tdims.push_back(m1->getObjectPtr()->trajectory_dimensions);
                tdims.push_back(m1->getObjectPtr()->number_of_samples);

                try { m3->getObjectPtr()->create(&tdims);}
                catch (std::runtime_error &err){
                    GEXCEPTION(err,"(%P|%t) Allocate trajectory data\n");
                    m1->release();

                    return 0;
                }

                if ((recv_count =
		     stream->recv_n
		     (m3->getObjectPtr()->get_data_ptr(),
		     sizeof(float)*tdims[0]*tdims[1])) <= 0) {
		  
		        GERROR("Unable to read trajectory data\n");
		        m1->release();
                        return 0;
                }

            }

            std::vector<size_t> adims;
            adims.push_back(m1->getObjectPtr()->number_of_samples);
            adims.push_back(m1->getObjectPtr()->active_channels);

            try{ m2->getObjectPtr()->create(&adims); }
            catch (std::runtime_error &err ){
                GEXCEPTION(err,"(%P|%t) Allocate sample data\n")
                    m1->release();

                return 0;
            }


            if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_COMPRESSION1)) { //Is this ZFP compressed data

#if defined GADGETRON_COMPRESSION_ZFP
                
                uint32_t comp_size = 0;
                if ((recv_count = stream->recv_n(&comp_size, sizeof(uint32_t))) <= 0) {
	            GERROR("Unable to read size of compressed data\n");
                    m1->release();
                    return 0;
                }

                char* comp_buffer = new char[comp_size];
                if ((recv_count = stream->recv_n(comp_buffer, comp_size)) <= 0) {
	            GERROR("Unable to read compressed data\n");
                    m1->release();
                    return 0;
                }


                zfp_type type = zfp_type_float;
                zfp_field* field = NULL;
                zfp_stream* zfp = NULL;
                bitstream* cstream = NULL;
                size_t zfpsize = comp_size;
                
                zfp = zfp_stream_open(NULL);
                field = zfp_field_alloc();
                
                cstream = stream_open(comp_buffer, comp_size);
                if (!cstream) {
                    GERROR("Unable to open compressed stream\n");
                    zfp_field_free(field);
                    zfp_stream_close(zfp);
                    stream_close(cstream);            
                    delete [] comp_buffer;
                    m1->release();
                    return 0;
                }
                zfp_stream_set_bit_stream(zfp, cstream);
                
                zfp_stream_rewind(zfp);
                if (!zfp_read_header(zfp, field, ZFP_HEADER_FULL)) {
                    GERROR("Unable to read compressed stream header\n");
                    zfp_field_free(field);
                    zfp_stream_close(zfp);
                    stream_close(cstream);            
                    delete [] comp_buffer;
                    m1->release();
                    return 0;
                }
                
                size_t nx = std::max(field->nx, 1u);
                size_t ny = std::max(field->ny, 1u);
                size_t nz = std::max(field->nz, 1u);
                
                if (nx*ny*nz != (m1->getObjectPtr()->number_of_samples*2*m1->getObjectPtr()->active_channels)) {
                    GERROR("Size of decompressed stream does not match the acquisition header\n");
                    GERROR("nx=%d, ny=%d, nz=%d, number_of_samples=%d, active_channels=%d\n", nx, ny, nz,  m1->getObjectPtr()->number_of_samples,  m1->getObjectPtr()->active_channels);
                    zfp_field_free(field);
                    zfp_stream_close(zfp);
                    stream_close(cstream);            
                    delete [] comp_buffer;
                    m1->release();
                    return 0;                
                }
                zfp_field_set_pointer(field, m2->getObjectPtr()->get_data_ptr());
                
                if (!zfp_decompress(zfp, field)) {
                    GERROR("Unable to decompress stream\n");            
                    zfp_field_free(field);
                    zfp_stream_close(zfp);
                    stream_close(cstream);            
                    delete [] comp_buffer;
                    m1->release();
                    return 0;                
                }
            
                zfp_field_free(field);
                zfp_stream_close(zfp);
                stream_close(cstream);            
                delete [] comp_buffer;
	
				int cha = m2->getObjectPtr()->get_size(1);
				int samples = m2->getObjectPtr()->get_size(0);

				/*std::vector<float> real(samples*cha);
				std::vector<float> imag(samples*cha);
				std::vector<float> data(samples*cha*2);
				memcpy(&data[0], m2->getObjectPtr()->get_data_ptr(), samples*cha*2*sizeof(float));
				for(int i = 0; i < samples*cha; i++){
					real[i] = data[i];
					imag[i] = data[i+samples*cha];
				}
				for(int i = 0; i < samples*cha; i++){
					data[2*i] = real[i];
					data[2*i+1] = imag[i];
				}
				
				//Print data for analysis				
				//int cha = m2->getObjectPtr()->get_size(1);
				//int samples = m2->getObjectPtr()->get_size(0);
				//std::cout << "channels = " << cha << std::endl;
				//std::cout << "samples = " << samples << std::endl;
				*/
				if(tmp.get_number_of_elements() == 0){
					//tmp.create(samples,cha,128);
				}
				//memcpy(tmp.get_data_ptr()+m1->getObjectPtr()->idx.kspace_encode_step_1*samples*cha, m2->getObjectPtr()->get_data_ptr(), samples*cha*2*sizeof(float));
				if(m1->getObjectPtr()->idx.kspace_encode_step_1 == 127){
					//std::cout << 7 << std::endl;
					//Gadgetron::write_nd_array<std::complex<float>>(&tmp, "tmp_random_uncompressed_zfp.cplx");
				}

                //At this point the data is no longer compressed and we should clear the flag
                m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_COMPRESSION1);

#else //GADGETRON COMPRESSION_ZFP
                
                //This is compressed data, but Gadgetron was not compiled with compression
                GERROR("Receiving compressed (ZFP) data, but Gadgetron was not compiled with ZFP support");
                m1->release();
                return 0;

#endif //GADGETRON_COMPRESSION_ZFP

            } else if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_COMPRESSION2)) {


				//std::cout << "line = " << m1->getObjectPtr()->idx.kspace_encode_step_1 << std::endl;
                //NHLBI Compression
                uint32_t comp_size = 0;
                if ((recv_count = stream->recv_n(&comp_size, sizeof(uint32_t))) <= 0) {
	            GERROR("Unable to read size of compressed data\n");
                    m1->release();
                    return 0;
                }

                std::vector<uint8_t> comp_buffer(comp_size,0);
                if ((recv_count = stream->recv_n(&comp_buffer[0], comp_size)) <= 0) {
	            GERROR("Unable to read compressed data\n");
                    m1->release();
                    return 0;
                }

				float* d_ptr = (float*)m2->getObjectPtr()->get_data_ptr();
				int cha = m2->getObjectPtr()->get_size(1);
				int samples = m2->getObjectPtr()->get_size(0);
				size_t bytes_needed = 0;
				size_t total_size = 0;
				CompressedBuffer<float> comp;
				bytes_needed = comp.deserialize(comp_buffer);
				comp_buffer.erase(comp_buffer.begin(),comp_buffer.begin()+bytes_needed);
				//float* d_ptr = (float*)m2->getObjectPtr()->get_data_ptr()+ch*samples;
	            for (size_t i = 0; i < comp.size(); i++) {
	                d_ptr[i] = comp[i]; //This uncompresses sample by sample into the uncompressed array
	            }
				total_size += comp.size();
				while(comp_buffer.size()>0){
		            CompressedBuffer<float> comp;
		            bytes_needed = comp.deserialize(comp_buffer);
					comp_buffer.erase(comp_buffer.begin(),comp_buffer.begin()+bytes_needed);
					//float* d_ptr = (float*)m2->getObjectPtr()->get_data_ptr()+ch*samples;
		            for (size_t i = 0; i < comp.size(); i++) {
		                d_ptr[i+total_size] = comp[i]; //This uncompresses sample by sample into the uncompressed array
		            }
					total_size += comp.size();
				}


                if (total_size != m2->getObjectPtr()->get_number_of_elements()*2) { //*2 for complex
	            GERROR("Mismatch between uncompressed data samples (%d) and expected number of samples (%d)\n", total_size, m2->getObjectPtr()->get_number_of_elements()*2);
                    m1->release();
                    return 0;
                }


                //Gadgetron::hoNDFFT<float>::instance()->fft(m2->getObjectPtr(),0);
				//std::cout << "channels = " << cha << std::endl;
				//std::cout << "samples = " << samples << std::endl;
				bool use_transform = false;
				int N = samples*cha*2;
				if(use_transform){
					//std::cout << "size " << N << std::endl;
					if(N > 50000) { 
						Gadgetron::hoNDFFT<float>::instance()->dct(d_ptr, N, 1, -1);
					}
					else {
						Gadgetron::hoNDFFT<float>::instance()->dct(d_ptr, N, 1, -1);
					}
				}
				
				if(m1->getObjectPtr()->idx.kspace_encode_step_1 == 0){
					//tmp.create(samples,cha,8);
				}
				//memcpy(tmp.get_data_ptr()+m1->getObjectPtr()->idx.kspace_encode_step_1*samples*cha, m2->getObjectPtr()->get_data_ptr(), samples*cha*2*sizeof(float));
				if(m1->getObjectPtr()->idx.kspace_encode_step_1 == 7){
					//std::cout << 7 << std::endl;
					//Gadgetron::write_nd_array<std::complex<float>>(&tmp, "tmp_spiral_uncompressed.cplx");
				}
                //At this point the data is no longer compressed and we should clear the flag
                m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_COMPRESSION2);


            } else { 
                //Uncompressed data
                if ((recv_count =
                     stream->recv_n
                     (m2->getObjectPtr()->get_data_ptr(),
                      sizeof(std::complex<float>)*adims[0]*adims[1])) <= 0) {
                    
	            GERROR("Unable to read Acq data\n");
                    m1->release();
                    return 0;
                }
            }
                
            return m1;
        }

protected:
			std::mutex mutex_;

    };    
}
#endif //GADGETISMRMRDREADWRITE_H
