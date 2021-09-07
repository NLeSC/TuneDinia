#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>


__global__ void invert_mapping(float *input,			/* original */
							   float *output,			/* inverted */
							   int npoints,				/* npoints */
							   int nfeatures)			/* nfeatures */
{
	int point_id = threadIdx.x + block_size_x*blockIdx.x;	/* id of thread */
	int i;

	if(point_id < npoints){
		for(i=0;i<nfeatures;i++){
			output[point_id + npoints*i] = input[point_id*nfeatures + i];
    }
	}
	return;
}

#endif 