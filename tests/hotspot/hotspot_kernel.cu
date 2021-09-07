#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <stdio.h>
#include <assert.h>
#define BLOCK_SIZE 16    
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
							   int border_cols,  // border offset 
							   int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx, 
                               float Ry, 
                               float Rz, 
                               float step){
	
        __shared__ float temp_on_cuda[block_size_y*tile_size_y][block_size_x];
        __shared__ float power_on_cuda[block_size_y*tile_size_y][block_size_x];
        __shared__ float temp_t[block_size_y*tile_size_y][block_size_x]; // saving temparary temperature result

	float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;
        
	int bx = blockIdx.x;
        int by = blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ty_new = 0;
        
	step_div_Cap=step/Cap;
                                
	Rx_1=1/Rx;
	Ry_1=1/Ry;
	Rz_1=1/Rz;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size 
        //with tile_size_y -> small block size on y-axis increased
	int small_block_rows = block_size_y*tile_size_y-iteration*2;//EXPAND_RATE
	int small_block_cols = block_size_x-iteration*2;//EXPAND_RATE

        /* The following values should not be altered since the small block size 
        is modified*/
        
        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkXmax = blkX+block_size_x-1;
        int blkYmax = blkY+block_size_y-1;

        // calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

        //FIRST STEP LOAD DATA TO S.M
        // load data if it is within the valid input range
	int loadXidx=xidx;
        int loadYidx;
        int index;
        for(int i = 0; i < tile_size_y; i++){
                ty_new = ty + block_size_y*i;
                loadYidx = yidx + i * block_size_y;
                index = grid_cols*loadYidx+loadXidx;
                if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
                        temp_on_cuda[ty_new][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
                        power_on_cuda[ty_new][tx] = power[index];// Load the power data from global memory to shared memory
                }
        }
	__syncthreads();

      

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? block_size_y-1-(blkYmax-grid_rows+1) : block_size_y*tile_size_y -1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? block_size_x-1-(blkXmax-grid_cols+1) : block_size_x-1;

        int W = tx-1;
        int E = tx+1;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E; 

        // SECOND STEP COMPUTE
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            
            /* These values need to be recomputed since we access 
            the y-axis in a loop. The loop must be expanded to include the computation*/
            for(int j=0; j<tile_size_y; j++){
                computed = false;
                ty_new = ty + block_size_y*j;
                int N = ty + block_size_y*j-1;
                int S = ty + block_size_y*j+1;
                N = (N < validYmin) ? validYmin : N;
                S = (S > validYmax) ? validYmax : S;
                // the effective range should also be modified in the y-axis
                if( IN_RANGE(tx, i+1, block_size_x-i-2) &&  \
                        IN_RANGE(ty_new, i+1, block_size_y*tile_size_y-i-2) &&  \ 
                        IN_RANGE(tx, validXmin, validXmax) && \
                        IN_RANGE(ty_new, validYmin, validYmax) ) {
                        computed = true;
                        temp_t[ty_new][tx] =   temp_on_cuda[ty_new][tx] + step_div_Cap * (power_on_cuda[ty_new][tx] + 
                                (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty_new][tx]) * Ry_1 + 
                                (temp_on_cuda[ty_new][E] + temp_on_cuda[ty_new][W] - 2.0*temp_on_cuda[ty_new][tx]) * Rx_1 + 
                                (amb_temp - temp_on_cuda[ty_new][tx]) * Rz_1);
                
                }
            }
        
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed){
                for(int j=0; j < tile_size_y; j++){
                    ty_new = ty + block_size_y*j;
                    temp_on_cuda[ty_new][tx]= temp_t[ty_new][tx];
                }
            }
            __syncthreads();
        }

        // THIRD STEP WRITE DATA BACK

        // update the global memory
        // after the last iteration, only threads coordinated within the 
        // small block perform the calculation and switch on ``computed''
        if (computed){
                for(int j=0;j < tile_size_y; j++){
                loadYidx = yidx + j * block_size_y;
                index = grid_cols*loadYidx+loadXidx;
                ty_new = ty + block_size_y*j;
                temp_dst[index]= temp_t[ty_new][tx];
            }		
        } 		
}

#endif