#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#define USE_READ_ONLY_CACHE read_only
#if USE_READ_ONLY_CACHE == 1
#define LDG(x, y) __ldg(x+y)
#elif USE_READ_ONLY_CACHE == 0
#define LDG(x, y) x[y]
#endif

#ifndef NFEATURES
	#define NFEATURES 35
#endif
#ifndef NPOINTS
	#define NPOINTS 1000
#endif
#ifndef NCLUSTERS
	#define NCLUSTERS 6
#endif

#define GPU_DELTA_REDUCTION
#define GPU_NEW_CENTER_REDUCTION
#define THREADS_PER_BLOCK block_size_x*block_size_y

#ifdef USE_CUB
#include <cub/cub.cuh>
#endif

extern "C"
__global__ void
kmeansPoint(float  *features,			/* in: [npoints*nfeatures] */
            int    *membership,
			float  *clusters, int* block_deltas,
			float* block_clusters,
			float* feature_flipped_d) 
{

	// block ID
	const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
	// point/thread ID  
	const unsigned int point_offset = block_id*block_size_x*block_size_y;
	const unsigned int point_id = point_offset + threadIdx.x;
	int  index = -1;

	if (point_id < NPOINTS)
	{
		int i, j;
		float min_dist = 1e10;
		float dist;													/* distance square between a point to cluster center */
		
		/* find the cluster center id with min distance to pt */
		#pragma unroll
		for (i=0; i< NCLUSTERS; i++) {
			int cluster_base_index = i*NFEATURES;					/* base index of cluster centers for inverted array */			
			float ans=0.0;												/* Euclidean distance sqaure */
			#pragma unroll
			for (j=0; j < NFEATURES; j++)
			{					
				int addr = point_id + j*NPOINTS;					/* appropriate index of data point */
				float diff = LDG(features,addr) -  //t_features[addr]
							  clusters[cluster_base_index + j];	/* distance between a data point to cluster centers */
				ans += diff*diff;									/* sum of squares */
			}
			dist = ans;		

			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			if (dist < min_dist) {
				min_dist = dist;
				index = i;
			}
		}
	}
	

#ifdef GPU_DELTA_REDUCTION
    // count how many points are now closer to a different cluster center	
	__shared__ int deltas[THREADS_PER_BLOCK];
	if(threadIdx.x < THREADS_PER_BLOCK) {
		deltas[threadIdx.x] = 0;
	}
#endif
	if (point_id < NPOINTS)
	{
#ifdef GPU_DELTA_REDUCTION
		/* if membership changes, increase delta by 1 */
		if (membership[point_id] != index) {
			deltas[threadIdx.x] = 1;
		}
#endif
		/* assign the membership to object point_id */
		membership[point_id] = index;
	}
	// make sure all the deltas have finished writing to shared memory
	__syncthreads();
#ifdef GPU_DELTA_REDUCTION
	#ifdef USE_CUB
	typedef cub::BlockReduce<int, THREADS_PER_BLOCK> BlockReduce;
	__shared__ typename BlockReduce::TempStorage smem_storage;
	int data = deltas[threadIdx.x];;
	int aggregate = BlockReduce(smem_storage).Sum(data);
	// propagate number of changes to global counter
	if(threadIdx.x == 0) {
		block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = aggregate;
	}
	// primitve reduction follows
	#else
	unsigned int threadids_participating = THREADS_PER_BLOCK / 2;
	for(;threadids_participating > 0; threadids_participating /= 2) {
   		if(threadIdx.x < threadids_participating) {
			deltas[threadIdx.x] += deltas[threadIdx.x + threadids_participating];
		}
   		__syncthreads();
	}
	// propagate number of changes to global counter
	if(threadIdx.x == 0) {
		block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = deltas[0];
	}
	#endif

#endif


#ifdef GPU_NEW_CENTER_REDUCTION
	
	__shared__ int new_center_ids[THREADS_PER_BLOCK];
	new_center_ids[threadIdx.x] = index;
	__syncthreads();

	float accumulator = 0.f;
	#pragma unroll
	for(int j = threadIdx.x; j < (NFEATURES*NCLUSTERS); j += THREADS_PER_BLOCK){
		accumulator = 0.f;
		int cluster = j / NFEATURES;    //0..clusters-1
		int feature = j % NFEATURES;	//0..nfeatures
		int new_base_index = point_offset*NFEATURES + feature;
		// accumulate over all the elements of this threadblock 
		#pragma unroll
		for(int i = 0; i < THREADS_PER_BLOCK; i++) {
			if(new_center_ids[i] == cluster) 
				accumulator += feature_flipped_d[new_base_index+i*NFEATURES];
		}
		block_clusters[(blockIdx.y*gridDim.x + blockIdx.x) * NCLUSTERS * NFEATURES + j] = accumulator;
	}
#endif

}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_