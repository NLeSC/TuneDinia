__global__ void
srad_cuda_1(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float q0sqr
) 
{

  // block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // indices
  int index   = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x + cols * ty + tx;
  int index_n = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x + tx - cols;
  int index_s = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x + cols * block_size_y + tx;
  int index_w = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x + cols * ty - 1;
  int index_e = cols * block_size_y * by * tile_size_y + block_size_x * bx * tile_size_x + cols * ty + block_size_x;

  // private memory per thread
  float n[tile_size_y][tile_size_x];
  float w[tile_size_y][tile_size_x];
  float e[tile_size_y][tile_size_x];
  float s[tile_size_y][tile_size_x];
  float jc, g2, l, num, den, qsqr, c;
  
  // shared memory allocation
  __shared__ float temp[block_size_y * tile_size_y][block_size_x * tile_size_x];
  __shared__ float north[block_size_y * tile_size_y][block_size_x * tile_size_x];
  __shared__ float south[block_size_y * tile_size_y][block_size_x * tile_size_x];
  __shared__ float  east[block_size_y * tile_size_y][block_size_x * tile_size_x];
  __shared__ float  west[block_size_y * tile_size_y][block_size_x * tile_size_x];

  // load data to shared memory
  #pragma unroll
  for(int j = 0; j < tile_size_y; j++){
    #pragma unroll
    for(int i = 0; i < tile_size_x; i++){
      north[ty + j * block_size_y][tx + i * block_size_x] = ( by == 0) ? J_cuda[(block_size_x * bx * tile_size_x) + tx + (i * block_size_x) + (j * block_size_y * cols)] : J_cuda[index_n + (j * block_size_y * cols) + (i * block_size_x)]; 
      south[ty + j * block_size_y][tx + i * block_size_x] = ( by == gridDim.y - 1) ? J_cuda[cols * block_size_y * ((gridDim.y * tile_size_y) - 1)  + block_size_x * bx * tile_size_x + cols * ( block_size_y - 1 ) + tx + i * block_size_x] : J_cuda[index_s + (i * block_size_x)  + (j * block_size_y * cols)];
      west[ty + j * block_size_y][tx + i * block_size_x] = (bx == 0 ) ? J_cuda[cols * block_size_y * by * tile_size_y + cols * ty + i * block_size_x  + (j * block_size_y * cols)] : J_cuda[index_w + i * block_size_x + j * block_size_y * cols];
      east[ty + j * block_size_y][tx + i * block_size_x] = (bx == gridDim.x - 1 ) ? J_cuda[cols * block_size_y * by * tile_size_y + block_size_x * ((gridDim.x * tile_size_x) - 1) + cols * ty + block_size_x-1 + (j * block_size_y * cols)] : J_cuda[index_e + (i * block_size_x) + (j * block_size_y * cols)];
      temp[ty + j * block_size_y][tx + i * block_size_x] = J_cuda[index + j * block_size_y * cols + i * block_size_x];
    }
  }

  __syncthreads();
  
  // // compute neighbouring elements for each thread
  #pragma unroll
  for(int j = 0; j < tile_size_y; j++){
    #pragma unroll
    for(int i = 0; i < tile_size_x ; i++){
      jc = temp[ty + j * block_size_y][tx + i * block_size_x];
      n[j][i] = ((ty + j * block_size_y) == 0) ? (north[ty + j * block_size_y][tx + i * block_size_x] - jc) : (temp[ty - 1  + j * block_size_y][tx + i * block_size_x] - jc);
      s[j][i] = ((ty + j * block_size_y) == ((block_size_y * tile_size_y) - 1)) ? (south[ty + j * block_size_y][tx + i * block_size_x] - jc) : (temp[ty+ 1 + j * block_size_y][tx + i * block_size_x] - jc);
      e[j][i] = ((tx + i * block_size_x) == (block_size_x * tile_size_x - 1)) ? (east[ty + j * block_size_y][tx + i * block_size_x] - jc) : (temp[ty + j * block_size_y][tx + 1 + i * block_size_x] - jc);
      w[j][i] = (tx + i * block_size_x == 0) ? (west[ty + j * block_size_y][tx + i * block_size_x]  - jc) : (temp[ty + j * block_size_y][tx - 1 + i * block_size_x] - jc);
  
  
      g2 = ( n[j][i] * n[j][i] + s[j][i] * s[j][i] + w[j][i] * w[j][i] + e[j][i] * e[j][i] ) / (jc * jc);

      l = ( n[j][i] + s[j][i] + w[j][i] + e[j][i] ) / jc;

      num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
      den  = 1 + (.25*l);
      qsqr = num/(den*den);

      // diffusion coefficent (equ 33)
      den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr));
      c = 1.0 / (1.0+den) ;
      
      // saturate diffusion coefficent
      if (c < 0){ C_cuda[index + j * block_size_y * cols + i * block_size_x] = 0;}
      else if (c > 1) { C_cuda[index + j * block_size_y * cols + i * block_size_x] = 1;}
      else {C_cuda[index + j * block_size_y * cols + i * block_size_x] = c;}
    }  
  }
  // write the results back	
  #pragma unroll
  for(int j = 0; j < tile_size_y; j++){
    #pragma unroll
    for(int i = 0; i < tile_size_x; i++){
      E_C[index + j * block_size_y * cols + i * block_size_x] = e[j][i];
      W_C[index + j * block_size_y * cols + i * block_size_x] = w[j][i];
      S_C[index + (j * block_size_y * cols) + i * block_size_x] = s[j][i];
      N_C[index + (j * block_size_y * cols) + (i * block_size_x)] = n[j][i];
    }
  }
}