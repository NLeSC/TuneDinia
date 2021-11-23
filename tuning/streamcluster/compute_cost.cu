#ifndef DIM
#define DIM 256
#endif

__global__ void
kernel_compute_cost(int num, int dim, int x, float* weight_d, long* assign_d, float* cost_d, int stride,
					float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d)
{
	// block ID and global thread ID
	const int bid  = blockIdx.x + gridDim.x * blockIdx.y;
	const int tid = block_size_x * bid + threadIdx.x;
	float retval = 0.0;

	if(tid < num)
	{
		float *lower = &work_mem_d[tid*stride];
		
		// cost between this point and point[x]: euclidean distance multiplied by weight
		//float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;
		#pragma unroll
    for(int i = 0; i < DIM; i++){
			float tmp = coord_d[(i*num)+tid] - coord_d[(i*num)+x];
			retval += tmp * tmp;
		}
		float x_cost = retval * weight_d[tid];
		// if computed cost is less then original (it saves), mark it as to reassign
		if ( x_cost < cost_d[tid] )
		{
			switch_membership_d[tid] = 1;
			lower[K] += x_cost - cost_d[tid];
		}
		// if computed cost is larger, save the difference
		else
		{
			lower[center_table_d[assign_d[tid]]] += cost_d[tid] - x_cost;
		}
	}
}