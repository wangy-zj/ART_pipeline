#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
  This is the main function to check the performance of process
*/

#include <getopt.h>

#include "../include/cuda/cuda_utilities.h"
#include "../include/test.h"
#include "../include/krnl.h"
#include "../include/dada_header.h"
#include "../include/dada_util.h"

// better to put this here
#include "dada_cuda.h"

using namespace std;

#define CUDA_STARTTIME(x)  cudaEventRecord(x ## _start, 0);

#define CUDA_STOPTIME(x) {					\
    float dtime;						\
    cudaEventRecord(x ## _stop, 0);				\
    cudaEventSynchronize(x ## _stop);				\
    cudaEventElapsedTime(&dtime, x ## _start, x ## _stop);	\
    x ## time += dtime; }

void usage(){
  fprintf(stdout,
	  "process - process data from a PSRDADA ring buffer with input_key and\n"
	  "           write result to another PSRDADA ring buffer with output_key\n"
	  
	  "Usage: process [options]\n"
	  " -input_key/-i <key>       Hexadecimal shared memory key of input PSRDADA ring buffer [default: %x]\n"
	  " -amplitude_output_key/-a <key>      Hexadecimal shared memory key of amplitude output PSRDADA ring buffer [default: %x]\n"
	  " -phase_output_key/-p <key>      Hexadecimal shared memory key of phase output PSRDADA ring buffer [default: %x]\n"
	  " -nthread/-n <N>           N thread per block for crossCorrPFT/crossCorr kernel [default: 64]\n"
	  " -gpu/-g <ID>              Run on ID GPU [default: 1]\n"
	  " -help/-h                  Show help\n",
	  DADA_DEFAULT_BLOCK_KEY,
	  DADA_DEFAULT_BLOCK_KEY+20,
    DADA_DEFAULT_BLOCK_KEY+40
	  );
}

int main(int argc, char *argv[]){  
  struct option options[] = {
			     {"input_key",              1, 0, 'i'},
			     {"amplitude_output_key",   1, 0, 'a'},
			     {"phase_output_key",       1, 0, 'p'},
			     {"nthread",                1, 0, 'n'},
			     {"gpu",                    1, 0, 'g'},
			     {"help",                   0, 0, 'h'}, 
			     {0, 0, 0, 0}
  };

  key_t input_key = DADA_DEFAULT_BLOCK_KEY;
  key_t amplitude_output_key = DADA_DEFAULT_BLOCK_KEY+20;  
  key_t phase_output_key = DADA_DEFAULT_BLOCK_KEY+40;  
  int gpu = 0;
  int nthread_amp = 128;   
  int nthread_phase = 128;   
  int reset_amp = 1;
  int reset_phi = 1;
  
  while (1) {
    unsigned ss;
    unsigned opt=getopt_long_only(argc, argv, "i:a:p:n:g:h", 
				  options, NULL);
    if (opt==EOF) break;
    
    switch (opt) {
      
    case 'i':
      key_t input_key_tmp;
      ss = sscanf(optarg, "%x", &input_key_tmp);
      if(ss != 1) {
	fprintf(stderr, "PROCESS_ERROR: Could not parse input key from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	usage();
	
	exit(EXIT_FAILURE);
      }
      else{
	input_key = input_key_tmp;
      }
      break;

    case 'a':
      key_t amplitude_key_tmp;
      ss = sscanf(optarg, "%x", &amplitude_key_tmp);
      if(ss != 1) {
	fprintf(stderr, "PROCESS_ERROR: Could not parse output key from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	usage();
	
	exit(EXIT_FAILURE);
      }
      else{
	amplitude_output_key = amplitude_key_tmp;
      }
      break;

    case 'p':
      key_t phase_key_tmp;
      ss = sscanf(optarg, "%x", &phase_key_tmp);
      if(ss != 1) {
	fprintf(stderr, "PROCESS_ERROR: Could not parse output key from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	usage();
	
	exit(EXIT_FAILURE);
      }
      else{
	phase_output_key = phase_key_tmp;
      }
      break;

    case 'g':
      unsigned gpu_tmp;
      ss = sscanf(optarg, "%d", &gpu_tmp);
      if (ss!=1){
        fprintf(stderr, "PROCESS ERROR: Could not parse GPU id from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);

	exit(EXIT_FAILURE);
      }
      else {
	gpu = gpu_tmp;
      }
      break; 

    case 'n':
      unsigned nthread_tmp;
      ss = sscanf(optarg, "%d", &nthread_tmp);
      if (ss!=1){
        fprintf(stderr, "PROCESS ERROR: Could not parse nthread from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);

	exit(EXIT_FAILURE);
      }
      else {
	nthread_amp = nthread_tmp;
      }
      break; 
      
    case 'h':
      usage();
      exit(EXIT_SUCCESS);
      
    case '?':
    default:
      break;
    }
  }

  fprintf(stdout, "DEBUG: gpu = %d\n", gpu);
  fprintf(stdout, "DEBUG: nthread = %d\n", nthread_amp);

  fprintf(stdout, "DEBUG: input_key = %x\n", input_key);
  fprintf(stdout, "DEBUG: amplitude_output_key = %x\n", amplitude_output_key);
  fprintf(stdout, "DEBUG: phase_output_key = %x\n", phase_output_key);
  
  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
  fprintf(stdout, "GPU name is %s\n", prop.name);
  
  // Get hold of dada ring buffers
    // Setup input dada ring buffer
  // lock hdu will be in loop
  dada_hdu_t *input_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(input_hdu, input_key);
  if(dada_hdu_connect(input_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to input hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    input_key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  }  
  // registers the existing host memory range for use by CUDA
  //dada_cuda_dbregister(input_hdu);   
  ipcbuf_t *input_dblock = (ipcbuf_t *)(input_hdu->data_block);
  ipcbuf_t *input_hblock = (ipcbuf_t *)(input_hdu->header_block);
  
  if(dada_hdu_lock_read(input_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking input HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
      
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have input HDU setup\n");
  
  // Setup amplitude output ring buffer
  dada_hdu_t *amplitude_output_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(amplitude_output_hdu, amplitude_output_key);
  if(dada_hdu_connect(amplitude_output_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to amplitude output hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    amplitude_output_key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  } 	   

  ipcbuf_t *amplitude_output_dblock = (ipcbuf_t *)(amplitude_output_hdu->data_block);
  ipcbuf_t *amplitude_output_hblock = (ipcbuf_t *)(amplitude_output_hdu->header_block);
  
  // make ourselves the write client
  if(dada_hdu_lock_write(amplitude_output_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking amplitude output HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
      
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have amplitude output HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have amplitude output HDU setup\n");
  
  // Setup phase output ring buffer
  dada_hdu_t *phase_output_hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(phase_output_hdu, phase_output_key);
  if(dada_hdu_connect(phase_output_hdu) < 0){ 
    fprintf(stderr, "PROCESS_ERROR:\tCan not connect to phase output hdu with key %x"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    phase_output_key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  } 

  ipcbuf_t *phase_output_dblock = (ipcbuf_t *)(phase_output_hdu->data_block);
  ipcbuf_t *phase_output_hblock = (ipcbuf_t *)(phase_output_hdu->header_block);
  
  // make ourselves the write client
  if(dada_hdu_lock_write(phase_output_hdu) < 0) {
    fprintf(stderr, "PROCESS_ERROR:\tError locking phase output HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
      
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have phase output HDU locked\n");
  fprintf(stdout, "PROCESS_INFO:\tWe have phase output HDU setup\n");


  // Now first read configuration from input ring buffer header
  dada_header_t dada_header = {0};
  char *input_hbuf = ipcbuf_get_next_read(input_hblock, NULL);
  read_dada_header(input_hbuf, &dada_header);

  // parse values from header buffer
  int FFTlen = 131072;   // FPGA中FFT的长度
  int nchan = 1024;    // FFT后选出的通道数
  int naverage = 1;    // 积分的block数
  int npkt = 2048;
  int nsamp = npkt*nchan;

  // Need to check it against expected value here
  int input_dbuf_size = nchan * sizeof(cufftComplex);   // 输入为FFT后的实部虚部
  int amplitude_dbuf_size = nchan * sizeof(float);   // 输出幅度
  int phase_dbuf_size = nchan * sizeof(float);   // 输出相位
  
  unsigned bytes_block_input  = ipcbuf_get_bufsz(input_dblock);
  unsigned bytes_block_amplitude = ipcbuf_get_bufsz(amplitude_output_dblock);  
  unsigned bytes_block_phase = ipcbuf_get_bufsz(phase_output_dblock);  

  fprintf(stdout, "PROCESS_INFO:\tinput buffer block size is %d bytes, output buffer block size is %d bytes\n",
	  bytes_block_input, bytes_block_amplitude);
  
  if (bytes_block_input!=input_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tinput buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_input, input_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input buffer block size checked\n");

  if (bytes_block_amplitude!=amplitude_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tamplitude output buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_amplitude, amplitude_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have amplitude output buffer block size checked\n");

  if (bytes_block_phase!=phase_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tphase output buffer block size mismatch, "
	    "%d vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_phase, phase_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have phase output buffer block size checked\n");

  // now we can setup new dada header buffer for output
  char *amplitude_hbuf = ipcbuf_get_next_write(amplitude_output_hblock);
  memcpy(amplitude_hbuf, input_hbuf, DADA_DEFAULT_HEADER_SIZE);

  // We update file size
  // a single buffer block per file
  // BYTES_PER_SECOND also need update, but ignore it for now
  if (ascii_header_set(amplitude_hbuf, "FILE_SIZE", "%d", bytes_block_amplitude) < 0)  {
    fprintf(stderr, "PROCESS_ERROR:\tError setting FILE_SIZE, "
	    "which happens at \"%s\", line [%d].\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  char *phase_hbuf = ipcbuf_get_next_write(phase_output_hblock);
  memcpy(phase_hbuf, input_hbuf, DADA_DEFAULT_HEADER_SIZE);

  // We update file size
  // a single buffer block per file
  // BYTES_PER_SECOND also need update, but ignore it for now
  if (ascii_header_set(phase_hbuf, "FILE_SIZE", "%d", bytes_block_phase) < 0)  {
    fprintf(stderr, "PROCESS_ERROR:\tError setting FILE_SIZE, "
	    "which happens at \"%s\", line [%d].\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  ipcbuf_mark_cleared(input_hblock);
  ipcbuf_mark_filled(amplitude_output_hblock, DADA_DEFAULT_HEADER_SIZE);
  ipcbuf_mark_filled(phase_output_hblock, DADA_DEFAULT_HEADER_SIZE);

  // Setup kernel dims
  dim3 grid_unpack(nsamp/128+1);
  dim3 blck_amp(1, 1);
  dim3 grid_amp(1, 1);
  dim3 blck_phi(1, 1);
  dim3 grid_phi(1, 1);
	blck_amp.x = nthread_amp;  
  grid_amp.x = (nchan - 1 + blck_amp.x) / blck_amp.x;
	blck_phi.x = nthread_phase;  
  grid_phi.x = (nchan - 1 + blck_phi.x) / blck_phi.x;

  // Setup cuda buffers

  /* 声明并开辟CPU相关变量内存 */
  float *amplitude = (float*) malloc(nsamp * sizeof(float));
  float *phase = (float*) malloc(nsamp * sizeof(float));

	/* 声明并开辟GPU相关变量内存 */
  int32_t *d_input;
  cuComplex *d_unpack;
  float *d_amplitude, *d_phase;
	checkCudaErrors(cudaMalloc(&d_input, nsamp * 2 * sizeof(int32_t)));
  checkCudaErrors(cudaMalloc(&d_unpack, nsamp * sizeof(cuComplex)));
	checkCudaErrors(cudaMalloc(&d_amplitude, nsamp * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_phase, nsamp * sizeof(float)));

  // fprintf(stdout, "PROCESS_INFO:\t device input buffer size is %lud bytes\n", pkt_nsamp*sizeof(int8_t));
  
  print_cuda_memory_info();

  // Setup timer
  cudaEvent_t pipeline_start;
  cudaEvent_t pipeline_stop;
  float pipelinetime = 0;

  checkCudaErrors(cudaEventCreate(&pipeline_start));
  checkCudaErrors(cudaEventCreate(&pipeline_stop));

  cudaEvent_t memcpyh2d_start;
  cudaEvent_t memcpyh2d_stop;
  float memcpyh2dtime = 0;

  checkCudaErrors(cudaEventCreate(&memcpyh2d_start));
  checkCudaErrors(cudaEventCreate(&memcpyh2d_stop));
  
  cudaEvent_t memcpyd2h_start;
  cudaEvent_t memcpyd2h_stop;
  float memcpyd2htime = 0;

  checkCudaErrors(cudaEventCreate(&memcpyd2h_start));
  checkCudaErrors(cudaEventCreate(&memcpyd2h_stop));

  int nblock = 0;
  
  // FILE *fp1 = fopen("/home/hero/Desktop/ART-pipeline/amplitude.txt", "w");   // 存储幅度数据到.txt文件方便自己查看结果
  // FILE *fp2 = fopen("/home/hero/Desktop/ART-pipeline/phase.txt", "w");   // 存储相位数据到.txt文件方便自己查看结果

  CUDA_STARTTIME(pipeline);  
  while(!ipcbuf_eod(input_dblock)){
    
    fprintf(stdout, "We are at %d block\n", nblock);

    // block memory copy
    char *input_cbuf = ipcbuf_get_next_read(input_dblock, NULL);
    if(!input_cbuf){
      fprintf(stderr, "Could not get next read data block\n");
      exit(EXIT_FAILURE);
    }
    
    CUDA_STARTTIME(memcpyh2d);  
    checkCudaErrors(cudaMemcpy(d_input, input_cbuf,npkt * nchan * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_STOPTIME(memcpyh2d);  
    ipcbuf_mark_cleared(input_dblock);
    fprintf(stdout, "Memory copy from host to device of %d block done\n", nblock);

    /* 解析输入数据 */
    krnl_unpack<<<grid_unpack, 128>>>(d_input,d_unpack,nsamp,chan)
    
    /* 计算幅度 */
    krnl_amplitude<<<grid_amp, blck_amp>>>(d_amplitude, d_input, FFTlen, nchan, reset_amp);
    getLastCudaError("Kernel execution failed [ amplitude ]");

    /* 计算相位 */
    krnl_phase<<<grid_phi, blck_phi>>>(d_phase, d_input, nchan, reset_phi);
    getLastCudaError("Kernel execution failed [ phase ]");

    nblock++;
    if(nblock % naverage == 0){
      reset_amp = 1;
    
      //// we copy data to ring buffer only when we get naverage blocks done
      //// block memory copy
      //// We will not get any output if the runtime is short than integration time
      char *output_amplitude = ipcbuf_get_next_write(amplitude_output_dblock);
      if(!output_amplitude){
      	fprintf(stderr, "Could not get next amplitude write data block\n");
      	exit(EXIT_FAILURE);
      }
      CUDA_STARTTIME(memcpyd2h);  
      checkCudaErrors(cudaMemcpy(output_amplitude, d_amplitude, bytes_block_amplitude, cudaMemcpyDeviceToHost));
      CUDA_STOPTIME(memcpyd2h);  
      ipcbuf_mark_filled(amplitude_output_dblock, bytes_block_amplitude);

      fprintf(stdout, "we copy data out\n");
    }
    else{
      reset_amp = 0;
    }

    if(nblock % naverage == 0){
      reset_phi = 1;
    
      //// we copy data to ring buffer only when we get naverage blocks done
      //// block memory copy
      //// We will not get any output if the runtime is short than integration time
      char *output_phase = ipcbuf_get_next_write(phase_output_dblock);
      if(!output_phase){
      	fprintf(stderr, "Could not get next phase write data block\n");
      	exit(EXIT_FAILURE);
      }
      CUDA_STARTTIME(memcpyd2h);  
      checkCudaErrors(cudaMemcpy(output_phase, d_phase, bytes_block_phase, cudaMemcpyDeviceToHost));
      CUDA_STOPTIME(memcpyd2h);  
      ipcbuf_mark_filled(phase_output_dblock, bytes_block_phase);

      fprintf(stdout, "we copy data out\n");
    }
    else{
      reset_phi = 0;
    }

  //   /* 存储幅度数据到.txt文件 */
  //   checkCudaErrors(cudaMemcpy(amplitude, d_amplitude, bytes_block_amplitude, cudaMemcpyDeviceToHost));
  //   // fwrite(amplitude, sizeof(float), nchan, fp1);   // 写入二进制数据
  //   for (int i = 0; i < nchan; i++) {
  //     fprintf(fp1, "%f \n", amplitude[i]);
  //   }
  //   /* 存储相位数据到.txt文件 */
  //   checkCudaErrors(cudaMemcpy(phase, d_phase, bytes_block_phase, cudaMemcpyDeviceToHost));
  //   // fwrite(phase, sizeof(float), nchan, fp2);   // 写入二进制数据
  //   for (int i = 0; i < nchan; i++) {
  //     fprintf(fp2, "%f \n", phase[i]);
  //   }
  }
  
  // fclose(fp1);
  // fclose(fp2);

  CUDA_STOPTIME(pipeline);

  fprintf(stdout, "pipeline   %f milliseconds, pipline with memory transfer averaged with %d blocks\n", pipelinetime/(float)nblock, nblock);
  fprintf(stdout, "pipeline   %f milliseconds, memory transfer h2d averaged with %d blocks\n", memcpyh2dtime/(float)nblock, nblock);
  fprintf(stdout, "pipeline   %f milliseconds, memory transfer d2h averaged with %d blocks\n", memcpyd2htime/(float)nblock, nblock);
  
  dada_hdu_unlock_read(input_hdu);
  dada_hdu_unlock_write(amplitude_output_hdu);
  dada_hdu_unlock_write(phase_output_hdu);

  dada_hdu_destroy(input_hdu);
  dada_hdu_destroy(amplitude_output_hdu);
  dada_hdu_destroy(phase_output_hdu);

  
  return EXIT_SUCCESS;
}    
