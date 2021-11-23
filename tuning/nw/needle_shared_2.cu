#include "needle_cuda_shared_2.h" 

__device__ __host__ int 
maximum( int a,
		 int b,
		 int c){

  int k;
  k = (a < b) ? b : a;
  k = (k < c) ? c : k;
  return k;
}

__global__ void
needle_cuda_shared_2(  int* referrence,
			  int* matrix_cuda, 
			  int cols,
			  int penalty) 
{

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int block_width = (cols - 1) / block_size_x;
  int iteration = block_width - 1;
  int b_index_x = bx + block_width - iteration;
  int b_index_y = block_width - bx -1;

  int index   = cols * block_size_x * b_index_y + block_size_x * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * block_size_x * b_index_y + block_size_x * b_index_x + tx + ( 1 );
  int index_w   = cols * block_size_x * b_index_y + block_size_x * b_index_x + ( cols );
    int index_nw =  cols * block_size_x * b_index_y + block_size_x * b_index_x;

  __shared__  int temp[block_size_x+1][block_size_x+1];
  __shared__  int ref[block_size_x][block_size_x];

  for ( int ty = 0 ; ty < block_size_x ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

   if (tx == 0)
		  temp[tx][0] = matrix_cuda[index_nw];
 
  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  temp[0][tx + 1] = matrix_cuda[index_n];
  
  __syncthreads();
  
  for( int m = 0 ; m < block_size_x ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);	  
	  
	  }
	  __syncthreads();
  }

 for( int m = block_size_x - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + block_size_x - m ;
		  int t_index_y =  block_size_x - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);


	  }
	  __syncthreads();
  }


  for ( int ty = 0 ; ty < block_size_x ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}