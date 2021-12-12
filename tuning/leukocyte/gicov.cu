// The number of sample points in each ellipse (stencil)
#ifndef NPOINTS
	#define NPOINTS 150
#endif

// The total number of sample ellipses
#ifndef NCIRCLES
	#define NCIRCLES 7
#endif

#ifndef GRAD_M
	#define GRAD_M 219 
#endif

#ifndef USE_TEXTURE 
	#define USE_TEXTURE 0
#endif

#ifndef USE_CONSTANT 
	#define USE_CONSTANT 0
#endif

__constant__ int c_tX[NCIRCLES * NPOINTS];
__constant__ int c_tY[NCIRCLES * NPOINTS];
__constant__ float c_sin_angle[NPOINTS];
__constant__ float c_cos_angle[NPOINTS];
// #ifndef inner_loop
// 	#define inner_loop 0
// #endif

//texture<float, 1, cudaReadModeElementType> tex;

// Kernel to find the maximal GICOV value at each pixel of a
//  video frame, based on the input x- and y-gradient matrices
__global__ void GICOV_kernel(float *gicov, float* device_grad_x, float* device_grad_y, float* cos_angle, float* sin_angle, int* tX, int* tY) {
	int k, n, x, y;
	
	int thread_id = blockIdx.x*block_size_x + threadIdx.x;

	// Determine this thread's pixel
	int row = thread_id / GRAD_M;
	int col = thread_id % GRAD_M;

	// Initialize the maximal GICOV score to 0
	float max_GICOV = 0.f;

	#pragma unroll
	// Iterate across each stencil
	for (k = 0; k < NCIRCLES; k++) {
		// Variables used to compute the mean and variance
		//  of the gradients along the current stencil
		float sum = 0.f, M2 = 0.f, mean = 0.f;		
		
		// Iterate across each sample point in the current stencil
		#pragma unroll 
		for (n = 0; n < NPOINTS; n++) {
			// Determine the x- and y-coordinates of the current sample point
			// Compute the combined gradient value at the current sample point
			#if USE_CONSTANT == 1
				y = col + c_tY[(k * NPOINTS) + n];
				x = row + c_tX[(k * NPOINTS) + n];
				int addr = x * GRAD_M + y;
				float p = device_grad_x[addr] * c_cos_angle[n] + 
							device_grad_y[addr] * c_sin_angle[n];
			#else
				y = col + tY[(k * NPOINTS) + n];
				x = row + tX[(k * NPOINTS) + n];
				int addr = x * GRAD_M + y;
				float p = device_grad_x[addr] * cos_angle[n] + 
							device_grad_y[addr] * sin_angle[n];
			#endif
			
			// #if USE_TEXTURE == 1
			// 	float p = tex1D(tex,addr) * cos_angle[n] + 
			// 				device_grad_y[addr] * sin_angle[n];
			// #endif
			
			// Update the running total
			sum += p;
			
			// Partially compute the variance
			float delta = p - mean;
			mean = mean + (delta / (float) (n + 1));
			M2 = M2 + (delta * (p - mean));
		}
		
		// Finish computing the mean
		mean = sum / ((float) NPOINTS);
		
		// Finish computing the variance
		float var = M2 / ((float) (NPOINTS - 1));
		
		// Keep track of the maximal GICOV value seen so far
		if (((mean * mean) / var) > max_GICOV){
			max_GICOV = (mean * mean) / var;
		}
	}
	
	// Store the maximal GICOV value
	gicov[(row * GRAD_M) + col] = max_GICOV;
	
}