__global__ void
srad_cuda_2(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,	
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float lambda,
		  float q0sqr
) 
{
	  //block id
	  int bx = blockIdx.x;
    int by = blockIdx.y;

	  //thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	  //indices
    int index   = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x + cols * ty + tx;
	int index_s = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x  + cols * block_size_y + tx;
    int index_e = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x  + cols * ty + block_size_x;
	
    float cc[tile_size_y][tile_size_x];
    float cn[tile_size_y][tile_size_x];
    float cs[tile_size_y][tile_size_x];
    float ce[tile_size_y][tile_size_x];
    float cw[tile_size_y][tile_size_x];
    float d_sum[tile_size_y][tile_size_x];

	  //shared memory allocation
	__shared__ float south_c[block_size_y * tile_size_y][block_size_x * tile_size_x];
    __shared__ float east_c[block_size_y * tile_size_y][block_size_x * tile_size_x];
    __shared__ float c_cuda_temp[block_size_y * tile_size_y][block_size_x * tile_size_x];
    __shared__ float temp[block_size_y * tile_size_y][block_size_x * tile_size_x];

    //load data to shared memory
    #pragma unroll
    for(int j = 0; j < tile_size_y; j++){
        #pragma unroll
        for(int i = 0; i < tile_size_x; i++){
            temp[ty + j * block_size_y][tx + i * block_size_x] = J_cuda[index  + (j * block_size_y * cols) + (i * block_size_x)]; 
            south_c[ty + j * block_size_y][tx + i * block_size_x] = ( by == gridDim.y - 1) ? C_cuda[cols * block_size_y * ((gridDim.y * tile_size_y) - 1)  + block_size_x * bx * tile_size_x + cols * ( block_size_y - 1 ) + tx + i * block_size_x] : C_cuda[index_s + (i * block_size_x)  + (j * block_size_y * cols)];
            east_c[ty + j * block_size_y][tx + i * block_size_x] = (bx == gridDim.x - 1 ) ? C_cuda[cols * block_size_y * by * tile_size_y + block_size_x * ((gridDim.x * tile_size_x) - 1) + cols * ty + block_size_x-1 + (j * block_size_y * cols)] : C_cuda[index_e + (i * block_size_x) + (j * block_size_y * cols)];
            c_cuda_temp[ty + j * block_size_y][tx + i * block_size_x] = C_cuda[index + (j * block_size_y * cols) + (i * block_size_x)];
        }
    }    
        

    __syncthreads();
    #pragma unroll
    for(int j = 0; j < tile_size_y; j++){
        #pragma unroll
        for(int i = 0; i < tile_size_x; i++){
            cc[j][i] = c_cuda_temp[ty + j * block_size_y][tx + i * block_size_x];
            cn[j][i]  = cc[j][i];
            cs[j][i] = ((ty + j * block_size_y) == ((block_size_y * tile_size_y) - 1)) ? south_c[ty + j * block_size_y][tx + i * block_size_x] : c_cuda_temp[ty+1  + j * block_size_y][tx + i * block_size_x];
            cw[j][i] = cc[j][i];
            ce[j][i] = ((tx + i * block_size_x) == (block_size_x * tile_size_x - 1)) ? east_c[ty + j * block_size_y][tx + i * block_size_x] : c_cuda_temp[ty + j * block_size_y][tx+1 + i * block_size_x];
            // // divergence 
            d_sum[j][i] = cn[j][i] * N_C[index + j * block_size_y * cols + i * block_size_x] + cs[j][i] * S_C[index + j * block_size_y * cols + i * block_size_x] + cw[j][i] * W_C[index + j * block_size_y * cols + i * block_size_x] + ce[j][i] * E_C[index + j * block_size_y * cols + i * block_size_x];
            // // image update 
            J_cuda[index + j * block_size_y * cols + i * block_size_x] = temp[ty + j * block_size_y][tx + i * block_size_x] + 0.25 * lambda * d_sum[j][i];
        }
    }
        
}
