#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#define GPU_DELTA_REDUCTION
#define GPU_NEW_CENTER_REDUCTION

__global__ void
kmeansPoint(float  *features,			/* in: [npoints*nfeatures] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			float  *clusters, int* block_deltas,
			float* block_clusters,
			float* feature_flipped_d) 
{

	// block ID
	const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
	// point/thread ID  
	const unsigned int point_id = block_id*block_size_x*block_size_y + threadIdx.x;

	int  index = -1;

	if (point_id < npoints)
	{
		int i, j;
		float min_dist =  1E+37;
		float dist;													/* distance square between a point to cluster center */
		
		/* find the cluster center id with min distance to pt */
		for (i=0; i<nclusters; i++) {
			int cluster_base_index = i*nfeatures;					/* base index of cluster centers for inverted array */			
			float ans=0.0;												/* Euclidean distance sqaure */

			for (j=0; j < nfeatures; j++)
			{					
				int addr = point_id + j*npoints;					/* appropriate index of data point */
				float diff = (features[addr] -  //t_features[addr]
							  clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
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
	__shared__ int deltas[block_size_x*block_size_y];
	if(threadIdx.x < block_size_x*block_size_y) {
		deltas[threadIdx.x] = 0;
	}
#endif
	if (point_id < npoints)
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

#ifdef GPU_DELTA_REDUCTION
	// make sure all the deltas have finished writing to shared memory
	__syncthreads();

	// now let's count them
	// primitve reduction follows
	unsigned int threadids_participating = (block_size_x*block_size_y) / 2;
	for(;threadids_participating > 1; threadids_participating /= 2) {
   		if(threadIdx.x < threadids_participating) {
			deltas[threadIdx.x] += deltas[threadIdx.x + threadids_participating];
		}
   		__syncthreads();
	}
	if(threadIdx.x < 1)	{deltas[threadIdx.x] += deltas[threadIdx.x + 1];}
	__syncthreads();
		// propagate number of changes to global counter
	if(threadIdx.x == 0) {
		block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = deltas[0];
		//printf("original id: %d, modified: %d\n", blockIdx.y*gridDim.x+blockIdx.x, blockIdx.x);
		
	}

#endif


#ifdef GPU_NEW_CENTER_REDUCTION
	int center_id = threadIdx.x / nfeatures;    
	int dim_id = threadIdx.x - nfeatures*center_id;

	__shared__ int new_center_ids[block_size_x*block_size_y];

	new_center_ids[threadIdx.x] = index;
	__syncthreads();

	/***
	determine which dimension calculte the sum for
	mapping of threads is
	center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...
	***/ 	

	int new_base_index = (point_id - threadIdx.x)*nfeatures + dim_id;
	float accumulator = 0.f;

	if(threadIdx.x < nfeatures * nclusters) {
		// accumulate over all the elements of this threadblock 
		for(int i = 0; i< (block_size_x*block_size_y); i++) {
			float val = feature_flipped_d[new_base_index+i*nfeatures];
			if(new_center_ids[i] == center_id) 
				accumulator += val;
		}
	
		// now store the sum for this threadblock
		/***
		mapping to global array is
		block0[center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...]block1[...]...
		***/
		block_clusters[(blockIdx.y*gridDim.x + blockIdx.x) * nclusters * nfeatures + threadIdx.x] = accumulator;
	}
#endif
}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_