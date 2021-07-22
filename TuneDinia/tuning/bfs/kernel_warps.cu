\/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_
// System includes
#include <stdio.h>
#include <assert.h>
#ifndef threads_per_node
    #define threads_per_node 32
#endif

// CUDA runtime
//#include <cuda_runtime.h>
#define warp_size threads_per_node

__global__ void
Kernel( int* starting, int* no_of_edges, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes) {
    //global warp index and within warp
    int i = (blockIdx.x * block_size_x + threadIdx.x)/warp_size;
    int tx = threadIdx.x & (warp_size-1); //lane ID
    if(i<no_of_nodes && g_graph_mask[i]) {
        int start = starting[i] + tx;
        int end = starting[i] + no_of_edges[i];
        g_graph_mask[i]=false;
        //printf("no.%d,g_graph_mask[%d] -> %d\n",tx,i,starting[i]);
        for(int j=start; j<end; j+=warp_size) {
            int id = g_graph_edges[j];
            if(!g_graph_visited[id]) {
                //printf("g_graph_visited[%d] -> %d, warps_size -> %d\n",id,g_graph_visited[id],warp_size);
                g_cost[id]=g_cost[i]+1;
                g_updating_graph_mask[id]=true;
            }
        }
    }
}

#endif
