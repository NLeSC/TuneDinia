#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <stdio.h>
#include <assert.h>

__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{   
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

	if(threadIdx.x + blockIdx.x * block_size_x >= Size-1-t) return;
	*(m_cuda+Size*(block_size_x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(block_size_x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}
#endif