#ifndef kernel_tuner
  #define grid_width 4096
  #define grid_height 4096
#endif

#define tile_width block_size_x*tile_size_x + 2
#define tile_height block_size_y*tile_size_y + 2

#define lambda 3.14
#define width (grid_width + 2)
#define height (grid_height + 2)

__global__ void
srad_cuda_2(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,	
		  float * J_cuda, 
		  float * C_cuda,
		  float q0sqr
) 
{
	  //thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int offset = 1;
	  float cc, cn, cs, ce, cw, d_sum;

	  //shared memory allocation
    __shared__ float c_cuda_temp[tile_height][tile_width];
    __shared__ float temp[tile_height][tile_width];

    //load data to shared memory
    #pragma unroll
    for (int j = ty; j < tile_height; j += block_size_y) {
      #pragma unroll
      for (int i = tx; i < tile_width; i += block_size_x) {
        int x = tile_size_x * block_size_x * blockIdx.x + i;
        int y = tile_size_y * block_size_y * blockIdx.y + j;
        if (x < width && y < height) {
            temp[j][i] = J_cuda[y * width + x];
            c_cuda_temp[j][i] = C_cuda[y * width + x];
        } else {
            temp[j][i] = 0.0;
            c_cuda_temp[j][i] = 0.0;
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for (int j = ty + offset; j < tile_height - offset; j += block_size_y) {
      int S = j + 1;

      #pragma unroll
      for (int i = tx + offset; i < tile_width - offset; i += block_size_x) {

        int x = tile_size_x * block_size_x * blockIdx.x + i;
        int y = tile_size_y * block_size_y * blockIdx.y + j;
        if (x > grid_width || y > grid_height) { continue; }
        
        int E = i + 1;
        cc = c_cuda_temp[j][i];
        cw = cc;
        cn = cc;
        cs = c_cuda_temp[S][i];
        ce = c_cuda_temp[j][E];
       // divergence 
        d_sum = cn * N_C[y * width + x] + cs * S_C[y * width + x] + cw * W_C[y * width + x] + ce * E_C[y * width + x];
        // image update 
        J_cuda[y * width + x] = temp[j][i] + 0.25 * lambda * d_sum;
      }
    }
    
}