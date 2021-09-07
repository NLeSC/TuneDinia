#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <stdio.h>
#include <assert.h>

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int t)
{
	#ifdef OPTIMIZED
	if(threadIdx.y + blockIdx.y * block_size_y >= Size-1-t) return;
	if(threadIdx.x + blockIdx.x * block_size_x >= Size-t) return;
	int xidx = blockIdx.x * block_size_x + threadIdx.x;
	int yidx = blockIdx.y * block_size_y + threadIdx.y;
	//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
	
	a_cuda[Size*(yidx+1+t)+(xidx+t)] -= m_cuda[Size*(yidx+1+t)+t] * a_cuda[Size*t+(xidx+t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if(xidx == 0){
		//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//printf("xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[yidx+1+t] -= m_cuda[Size*(yidx+1+t)+(xidx+t)] * b_cuda[t];
	}
	#endif
	#ifdef ORIGINAL
	if(threadIdx.x + blockIdx.x * block_size_x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * block_size_y >= Size-t) return;
	int xidx = blockIdx.x * block_size_x + threadIdx.x;
	int yidx = blockIdx.y * block_size_y + threadIdx.y;
	//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
	
	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if(yidx == 0){
		//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//printf("xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}
	#endif
}

#endif