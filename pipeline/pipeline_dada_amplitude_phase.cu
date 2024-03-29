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
  //int nthread_phase = 128;
  int nblocksave = 128;  
  
  //参数解析
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
  fprintf(stdout, "DEBUG: input_key = %x\n", input_key);
  fprintf(stdout, "DEBUG: amplitude_output_key = %x\n", amplitude_output_key);
  fprintf(stdout, "DEBUG: phase_output_key = %x\n", phase_output_key);
  
  // Setup GPU with ID and print out its name
  cudaDeviceProp prop = {0};
  int gpu_get = gpuDeviceInit(gpu); // The required gpu might be different from what we get
  fprintf(stdout, "Asked for GPU %d, got GPU %d\n", gpu, gpu_get);
  checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_get));
  fprintf(stdout, "GPU name is %s\n", prop.name);
  
  // 设置并锁定输入和输出的环形缓冲
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


  // 解析输入环形缓冲头数据，并计算相应参数
  dada_header_t dada_header = {0};
  char *input_hbuf = ipcbuf_get_next_read(input_hblock, NULL);
  read_dada_header(input_hbuf, &dada_header);

  // parse values from header buffer
  //int FFTlen = 131072;   // FPGA中FFT的长度
  int nchan = dada_header.pkt_nchan;    // FFT后选出的通道数
  int naverage = dada_header.naverage;    // 积分的block数
  int ntime = dada_header.pkt_ntime;    //每个包的时间数目
  //int ntime = 1;
  int nstream = dada_header.nstream;    // 数据路数
  int npkt = dada_header.npkt;          // 每个block包含的包数目
  int nsamp = npkt*nchan*ntime*nstream; // 总的数据点数

  fprintf(stdout, "DEBUG: nsamp = %d, npkt = %d, ntime = %d \n", nsamp, npkt, ntime);

  // Need to check it against expected value here
  // 计算每个环形缓冲的大小并和设置的大小匹配
  int input_dbuf_size = nsamp * sizeof(int32_t) * 2;            // 输入为FFT后的实部虚部
  int amplitude_dbuf_size = nsamp * sizeof(float) / naverage;   // 输出幅度,积分之后
  int phase_dbuf_size = nsamp * sizeof(float) / naverage;       // 输出相位,积分之后
  
  uint64_t bytes_block_input  = ipcbuf_get_bufsz(input_dblock);
  uint64_t bytes_block_amplitude = ipcbuf_get_bufsz(amplitude_output_dblock);  
  uint64_t bytes_block_phase = ipcbuf_get_bufsz(phase_output_dblock);  

  fprintf(stdout, "PROCESS_INFO:\tinput buffer block size is %" PRId64 " bytes, amplitude buffer block size is %" PRId64 " bytes, phase buffer block size is %" PRId64 " bytes\n",
	  bytes_block_input, bytes_block_amplitude, bytes_block_phase);
  
  if (bytes_block_input!=input_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tinput buffer block size mismatch, "
	    "%" PRId64 "vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_input, input_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have input buffer block size checked\n");

  if (bytes_block_amplitude!=amplitude_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tamplitude output buffer block size mismatch, "
	    "%" PRId64 "vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_amplitude, amplitude_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have amplitude output buffer block size checked\n");

  if (bytes_block_phase!=phase_dbuf_size){
    fprintf(stderr, "PROCESS_ERROR:\tphase output buffer block size mismatch, "
	    "%" PRId64 " vs %d "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    bytes_block_phase, phase_dbuf_size,
	    __FILE__, __LINE__);	
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "PROCESS_INFO:\tWe have phase output buffer block size checked\n");

  // now we can setup new dada header buffer for output
  // 更新输出的dada头，主要更新FILE_SIZE,影响存储文件大小
  // BYTES_PER_SECOND也要更新，但不影响文件存储
  char *amplitude_hbuf = ipcbuf_get_next_write(amplitude_output_hblock);
  memcpy(amplitude_hbuf, input_hbuf, DADA_DEFAULT_HEADER_SIZE);

  if (ascii_header_set(amplitude_hbuf, "FILE_SIZE", "%d", bytes_block_amplitude*nblocksave) < 0)  {
    fprintf(stderr, "PROCESS_ERROR:\tError setting FILE_SIZE, "
	    "which happens at \"%s\", line [%d].\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  char *phase_hbuf = ipcbuf_get_next_write(phase_output_hblock);
  memcpy(phase_hbuf, input_hbuf, DADA_DEFAULT_HEADER_SIZE);

  if (ascii_header_set(phase_hbuf, "FILE_SIZE", "%d", bytes_block_phase*nblocksave) < 0)  {
    fprintf(stderr, "PROCESS_ERROR:\tError setting FILE_SIZE, "
	    "which happens at \"%s\", line [%d].\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  ipcbuf_mark_cleared(input_hblock);
  ipcbuf_mark_filled(amplitude_output_hblock, DADA_DEFAULT_HEADER_SIZE);
  ipcbuf_mark_filled(phase_output_hblock, DADA_DEFAULT_HEADER_SIZE);

  // 设置GPU计算的kernel数目
  dim3 grid_unpack(npkt*nstream/100+1,100); //kernel dims for amp and pha computing
  int blck_inte = npkt/naverage;    //kernel dims for amp and pha intergration

	// 声明并开辟GPU相关变量内存
  int32_t *d_input;
  float *d_amplitude, *d_phase, *out_amplitude, *out_phase;
	checkCudaErrors(cudaMalloc(&d_input, nsamp * 2 * sizeof(int32_t)));
	checkCudaErrors(cudaMalloc(&d_amplitude, nsamp * sizeof(float)));
  checkCudaErrors(cudaMalloc(&out_amplitude, nsamp * sizeof(float)/naverage));
	checkCudaErrors(cudaMalloc(&d_phase, nsamp * sizeof(float)));
  checkCudaErrors(cudaMalloc(&out_phase, nsamp * sizeof(float)/naverage));
  print_cuda_memory_info();

  // 设置计时器，记录数据处理耗时
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

  // 初始化block计数
  int nblock = 0;

  CUDA_STARTTIME(pipeline);  
  // 当block有数据时，循环处理
  while(!ipcbuf_eod(input_dblock)){
    
    fprintf(stdout, "We are at %d block\n", nblock);

    char *input_cbuf = ipcbuf_get_next_read(input_dblock, &bytes_block_input);
    if(!input_cbuf){
      fprintf(stderr, "Could not get next read data block\n");
      exit(EXIT_FAILURE);
    }
    
    // 将数据由内存拷贝至显存
    CUDA_STARTTIME(memcpyh2d);  
    checkCudaErrors(cudaMemcpy(d_input, input_cbuf, bytes_block_input, cudaMemcpyHostToDevice));
    CUDA_STOPTIME(memcpyh2d);  
    ipcbuf_mark_cleared(input_dblock);
    fprintf(stdout, "Memory copy from host to device of %d block done\n", nblock);

    //解析输入数据 并 计算幅度和相位 
    krnl_amplitude_phase<<<grid_unpack, nchan>>>(d_input,d_amplitude,d_phase, nsamp);
    getLastCudaError("Kernel execution failed [ amplitude & phase computing ]");

    // 幅度积分 
    vectorSum<<<blck_inte,nchan>>>(d_amplitude, out_amplitude, naverage);
    getLastCudaError("Kernel execution failed [ amplitude integration ]");

    // 相位积分
    vectorSum<<<blck_inte,nchan>>>(d_phase, out_phase, naverage);
    getLastCudaError("Kernel execution failed [ phase integration ]");

    // 更新block计数
    nblock++;

    // 将计算结果由显存拷贝至内存
    CUDA_STARTTIME(memcpyd2h); 
    char *output_amplitude = ipcbuf_get_next_write(amplitude_output_dblock);
    if(!output_amplitude){
      fprintf(stderr, "Could not get next amplitude write data block\n");
      exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaMemcpy(output_amplitude, out_amplitude, bytes_block_amplitude, cudaMemcpyDeviceToHost));
    ipcbuf_mark_filled(amplitude_output_dblock, bytes_block_amplitude);


    char *output_phase = ipcbuf_get_next_write(phase_output_dblock);
    if(!output_phase){
      fprintf(stderr, "Could not get next phase write data block\n");
      exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaMemcpy(output_phase, out_phase, bytes_block_phase, cudaMemcpyDeviceToHost));
    ipcbuf_mark_filled(phase_output_dblock, bytes_block_phase);
    CUDA_STOPTIME(memcpyd2h);

    fprintf(stdout, "we copy amplitude and phase data out\n"); 
  }
  
  CUDA_STOPTIME(pipeline);

  // 当数据流停止时结束，并输出平均数据速率
  fprintf(stdout, "pipeline   %f milliseconds, pipline with memory transfer averaged with %d blocks\n", pipelinetime/(float)nblock, nblock);
  fprintf(stdout, "pipeline   %f milliseconds, memory transfer h2d averaged with %d blocks\n", memcpyh2dtime/(float)nblock, nblock);
  fprintf(stdout, "pipeline   %f milliseconds, memory transfer d2h averaged with %d blocks\n", memcpyd2htime/(float)nblock, nblock);

  // 释放对应的显存
  cudaFree(d_input);
  cudaFree(d_amplitude);
  cudaFree(d_phase);
  cudaFree(out_amplitude);
  cudaFree(out_phase);

  return EXIT_SUCCESS;
}
