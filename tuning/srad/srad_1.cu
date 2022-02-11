#ifndef kernel_tuner
  #define grid_width 4096
  #define grid_height 4096
#endif

#define tile_width block_size_x*tile_size_x + 2
#define tile_height block_size_y*tile_size_y + 2

#define width (grid_width + 2)
#define height (grid_height + 2)

__global__ void
srad_cuda_1(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,
		  float * J_cuda, 
		  float * C_cuda,
		  float q0sqr
) 
{

  // thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;  
  // variables used in the main part of this stencil computation 
  float n, w, e, s, jc, g2, l, num, den, qsqr, c;
  // variable used to access properly matrices with extra columns/rows
  int offset = 1;
  // shared memory allocation
  __shared__ float temp[tile_height][tile_width];

  // fill shared memory with values
  #pragma unroll
  for (int j = ty; j < tile_height; j += block_size_y) {
      #pragma unroll
      for (int i = tx; i < tile_width; i += block_size_x) {
          int x = tile_size_x * block_size_x * blockIdx.x + i;
          int y = tile_size_y * block_size_y * blockIdx.y + j;
          if (x < width && y < height) {
              temp[j][i] = J_cuda[y * width + x];
          } else {
              temp[j][i] = 0.0;
          }
      }
  }
  __syncthreads();
  
  //Compute each index taking into account the empty rows/columns
  #pragma unroll
  for (int j= ty + offset; j < tile_height - offset; j+= block_size_y) {
      int N = j - 1;
      int S = j + 1;

      #pragma unroll
      for (int i= tx + offset; i < tile_width - offset; i+= block_size_x) {
          int x = tile_size_x * block_size_x * blockIdx.x + i;
          int y = tile_size_y * block_size_y * blockIdx.y + j;

          if ((x > grid_width) || (y > grid_height)) { continue; }
          int W = i - 1;
          int E = i + 1;

          jc = temp[j][i];
          n = temp[N][i];
          s = temp[S][i];
          w = temp[j][W];
          e = temp[j][E];

          //do computation
          g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);
          l = ( n + s + w + e ) / jc;
          num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
          den  = 1 + (.25*l);
          qsqr = num/(den*den);

          // diffusion coefficent (equ 33)
          den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
          c = 1.0 / (1.0+den) ;

          // store the results back
          if (c < 0){ C_cuda[y * width + x] = 0; }
          else if (c > 1) { C_cuda[y * width + x] = 1; }
          else { C_cuda[y * width + x] = c; }
          E_C[y * width + x] = e;
          W_C[y * width + x] = w;
          N_C[y * width + x] = n;
          S_C[y * width + x] = s;
      }
  }


}