
#ifndef kernel_tuner
  #define grid_width 4096
  #define grid_height 4096
  #define block_size_x 16
  #define block_size_y 16
  #define tile_size_x 1
  #define tile_size_y 1
  #define temporal_tiling_factor 1
  #define max_tfactor 10
#endif


//calculate shared memory size, depends on temporal_tiling_factor and on tile_size_x/y
#define tile_width block_size_x*tile_size_x + temporal_tiling_factor * 2
#define tile_height block_size_y*tile_size_y + temporal_tiling_factor * 2


#define amb_temp 80.0f


#define input_width (grid_width+max_tfactor*2)
#define input_height (grid_height+max_tfactor*2)

#define output_width grid_width
#define output_height grid_height

__global__ void calculate_temp(float *power,          //power input
                               float *temp,           //temperature input
                               float *temp_dst,       //temperature output
                               const float Rx_1,
                               const float Ry_1,
                               const float Rz_1,
                               const float step_div_cap) {

    //offset input pointers to make the code testable with different temporal tiling factors
    float* power_src = power+(max_tfactor-temporal_tiling_factor)*input_width+max_tfactor-temporal_tiling_factor;
    float* temp_src = temp+(max_tfactor-temporal_tiling_factor)*input_width+max_tfactor-temporal_tiling_factor;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float temp_on_cuda[tile_height][tile_width];
    __shared__ float power_on_cuda[tile_height][tile_width];
    __shared__ float temp_t[tile_height][tile_width];

    // fill shared memory with values
    for (int j=ty; j<tile_height; j+=block_size_y) {
        for (int i=tx; i<tile_width; i+=block_size_x) {
            int x = tile_size_x*block_size_x*blockIdx.x+i;
            int y = tile_size_y*block_size_y*blockIdx.y+j;
            if (x < input_width && y < input_height) {
                temp_on_cuda[j][i] = temp_src[y*input_width + x];
                power_on_cuda[j][i] = power_src[y*input_width + x];
            } else {
                temp_on_cuda[j][i] = 0.0;
                power_on_cuda[j][i] = 0.0;
            }
        }
    }
    __syncthreads();


    //main computation

    for (int iteration=1; iteration <= temporal_tiling_factor; iteration++) {

        //cooperatively compute the area, shrinking with each iteration
        for (int j=ty+iteration; j<tile_height-iteration; j+=block_size_y) {
            int N = j-1;
            int S = j+1;

            for (int i=tx+iteration; i<tile_width-iteration; i+=block_size_x) {
                int W = i-1;
                int E = i+1;

                //do computation
                temp_t[j][i] = temp_on_cuda[j][i] + step_div_cap * (power_on_cuda[j][i] +
                     (temp_on_cuda[S][i] + temp_on_cuda[N][i] - 2.0*temp_on_cuda[j][i]) * Ry_1 +
                     (temp_on_cuda[j][E] + temp_on_cuda[j][W] - 2.0*temp_on_cuda[j][i]) * Rx_1 +
                     (amb_temp - temp_on_cuda[j][i]) * Rz_1);

            }
        }

        __syncthreads();

        if(iteration == temporal_tiling_factor)
            break;

        //swap
        for (int j=ty+iteration; j<tile_height-iteration; j+=block_size_y) {
            for (int i=tx+iteration; i<tile_width-iteration; i+=block_size_x) {
                temp_on_cuda[j][i] = temp_t[j][i];
            }
        }
        __syncthreads();

    }


    //write out result, should be 1 per thread unless spatial blocking is used
    int iteration = temporal_tiling_factor;
    for (int tj=0; tj<tile_size_y; tj++) {
        for (int ti=0; ti<tile_size_x; ti++) {
            int x = tile_size_x*block_size_x*blockIdx.x+ti*block_size_x+tx;
            int y = tile_size_y*block_size_y*blockIdx.y+tj*block_size_y+ty;
            if (x < output_width && y < output_height) {
                temp_dst[y*output_width + x] = temp_t[tj*block_size_y+ty+iteration][ti*block_size_x+tx+iteration];
            }
        }
    }


}
