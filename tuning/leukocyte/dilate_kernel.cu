// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
#ifndef STREL_M
	#define STREL_M 25
#endif

#ifndef STREL_N
	#define STREL_N 25
#endif

// The size of the structuring element used in dilation
#ifndef STREL_SIZE
	#define STREL_SIZE (12 * 2 + 1)
#endif

#ifndef IMG_M
	#define IMG_M 219
#endif

#ifndef IMG_N
	#define IMG_N 640
#endif

#ifndef USE_CONSTANT 
	#define USE_CONSTANT 0
#endif

// Constant device array holding the structuring element used by the dilation kernel
__constant__ float c_strel[STREL_SIZE * STREL_SIZE];

__global__ void dilate_kernel(float *dilated, float* strel, float* img) {	
	// Find the center of the structuring element
	int el_center_i = STREL_M / 2;
	int el_center_j = STREL_N / 2;

	// Determine this thread's location in the matrix
	int thread_id = (blockIdx.x * block_size_x) + threadIdx.x;
	int i = thread_id % IMG_M;
	int j = thread_id / IMG_M;

	// Initialize the maximum GICOV score seen so far to zero
	float max = 0.0;

	// Iterate across the structuring element in one dimension
	int el_i, el_j, x, y;
	#pragma unroll
	for(el_i = 0; el_i < STREL_M; el_i++) {
		y = i - el_center_i + el_i;
		// Make sure we have not gone off the edge of the matrix
		if( (y >= 0) && (y < IMG_M) ) {
			// Iterate across the structuring element in the other dimension
			#pragma unroll
			for(el_j = 0; el_j < STREL_N; el_j++) {
				x = j - el_center_j + el_j;
				// Make sure we have not gone off the edge of the matrix
				//  and that the current structuring element value is not zero
				#if USE_CONSTANT == 1
					if( (x >= 0) && (x < IMG_N) && (c_strel[(el_i * STREL_N) + el_j] != 0) ) {
							// Determine if this is maximal value seen so far
							int addr = (x * IMG_M) + y;
							float temp = img[addr];
							if (temp > max) max = temp;
					}
				#else
					if( (x >= 0) && (x < IMG_N) && (strel[(el_i * STREL_N) + el_j] != 0) ) {
							// Determine if this is maximal value seen so far
							int addr = (x * IMG_M) + y;
							float temp = img[addr];
							if (temp > max) max = temp;
					}
				#endif
			}
		}
	}
	
	// Store the maximum value found
	dilated[(i * IMG_N) + j] = max;
}