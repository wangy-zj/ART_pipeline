#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/krnl.h"

// 数据解析,求解幅度和相位
__global__ void krnl_amplitude_phase(int32_t *input, float *d_amplitudeOut, float * d_divisionOut, int nsamp)
{
  int samp = blockIdx.x + blockIdx.y * gridDim.x;
  int nchan = blockDim.x;
  int chan = threadIdx.x;
  int index = samp*nchan + chan;
  if(index<nsamp){
    //int32_t real = input[index*2];
    //int32_t img = input[index*2+1];
    int32_t real = index*2;
    int32_t img = index*2+1;
    d_amplitudeOut[index] = 10*log10f(powf(real,2)+powf(img,2)+(10e-8))-180+7;
    d_divisionOut[index] = atan2f(real,img+(10e-8));
  }
  //if(d_divisionOut[i]<0){
   // d_divisionOut[i] += 2*M_PI;
  //}
}

/*
__global__ void vectorSum(float *g_idata, float *g_odata){
  extern __shared__ int sdata[];
  // each thread loads one element from global to shared mem
  //unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int chan = blockIdx.x;
  unsigned int inte = blockDim.x;
  unsigned int i = blockIdx.x + gridDim.x * threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s=1;s<blockDim.x;s*=2) {
    if (tid%(2*s)==0) {
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
*/

// 对幅度和相位进行积分
__global__ void vectorSum(float *g_idata, float *g_odata, int naverage){
  unsigned int chan = threadIdx.x;
  unsigned int samp = blockIdx.x;
  unsigned int nchan = blockDim.x;
  g_odata[chan+samp*nchan] = g_idata[chan+samp*nchan*naverage]/naverage;
  for(unsigned int s=1;s<naverage;s++){
    g_odata[chan+samp*nchan] += g_idata[chan+nchan*s+samp*nchan*naverage]/naverage;
  }
  __syncthreads();
}

/*
  This kernel is purely for the transpose of [NFFT-NCHAN] data into [NCHAN-NFFT]
  n: NCHAN
  m: NFFT
*/
__global__ void krnl_tf2ft_1ant1pol(const cuComplex* in, cuComplex *out, int m, int n){  
  
  __shared__ cuComplex tile[TILE_DIM][TILE_DIM + 1];

  //gridsize_transpose.x = ceil(NCHAN / TILE_DIM) = ceil(n / TILE_DIM);
  //gridsize_transpose.y = ceil(NFFT  / TILE_DIM) = ceil(m / TILE_DIM);
  //gridsize_transpose.z = 1;
  //blocksize_transpose.x = TILE_DIM;
  //blocksize_transpose.y = NROWBLOCK_TRANS;
  //blocksize_transpose.z = 1;
  
  // Load matrix into tile
  // Every Thread loads in this case 4 elements into tile.
  // TILE_DIM/NROWBLOCK_TRANS = 32/8 = 4 
  
  int i_n = blockIdx.x * TILE_DIM + threadIdx.x;
  int i_m = blockIdx.y * TILE_DIM + threadIdx.y; 
  for (int i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS) {
    if(i_n < n  && (i_m+i) < m){
      int64_t loc_in = (i_m+i)*n + i_n;
      tile[threadIdx.y+i][threadIdx.x] = in[loc_in];
    }
  }
  __syncthreads();
  
  i_n = blockIdx.y * TILE_DIM + threadIdx.x; 
  i_m = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS) {
    if(i_n < m  && (i_m+i) < n){
      int64_t loc_out = (i_m+i)*m + i_n;
      out[loc_out] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}
