#include <stdio.h>
#include "math.h"
#include "cuda.h"
#include "backprop.h"
#include "bpnn_layerforward_CUDA.h"
#define stop_loop 0

__global__ void
bpnn_layerforward_CUDA_test(float *input_cuda,
             float *input_hidden_cuda,
             float *hidden_partial_sum,
             int in,
             int hid) 
{
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y; 


    int index =  ( hid + 1 ) * block_size_y * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 );

    int index_in = block_size_y * by + ty + 1;

    __shared__ float input_node[block_size_y];
    __shared__ float weight_matrix[block_size_y][HIDDEN];

    if ( tx == 0 ) { // read the inputs fed to the NN
       input_node[ty] = input_cuda[index_in];
    }



    weight_matrix[ty][tx] = input_hidden_cuda[index];
    __syncthreads();
 //for-loop much in same way as the for-loop on the bottom
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
//end of for-loop

    __syncthreads();

   //sum all the weights for every hidden node
   for (unsigned int s=block_size_y/2; s>stop_loop; s>>=1) {
        if (ty < s) {
            weight_matrix[ty][tx] += weight_matrix[ty + s][tx];
        }
        __syncthreads();
   }

   //store the value to the output matrix
   if ( ty == 0 ) {
    //use threads in the x-dimension to write back the results of the partial sum for each y
   	for (int i=tx; i<HIDDEN; i+=block_size_x) { // condition i < HIDDEN
	      hidden_partial_sum[by * hid + i] = weight_matrix[0][i];
	   }
   }

}
