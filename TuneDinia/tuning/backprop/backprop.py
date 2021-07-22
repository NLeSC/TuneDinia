#!/usr/bin/env python
"""Test project for device target creation"""

import os
import sys
from collections import OrderedDict
import json

import numpy as np
import kernel_tuner
from kernel_tuner import tune_kernel
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]

def tune(size,hidden):

    width_16 = 16
    width_32 = 32
    num_blocks_16 = int(size / width_16)
    num_blocks_32 = int(size / width_32)
    problem_size = (width_16,size)
    input_cuda = 100+np.random.randn(size+1).astype(np.float32)
    output_hidden_cuda = np.zeros((hidden+1), dtype=np.float32)
    input_hidden_cuda = np.absolute(np.random.randn((size+1)*(hidden+1)).astype(np.float32))
    hidden_partial_sum_16 = np.zeros(num_blocks_16*hidden, dtype=np.float32)
    hidden_partial_sum_32 = np.zeros(size*hidden, dtype=np.float32) #making sure this is large enough for all block_size_y
    in_size = np.int32(size)
    hid = np.int32(hidden)

    
    args2 = [input_cuda, input_hidden_cuda, hidden_partial_sum_32, in_size, hid]

    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda x: (size/1e9)/(x['time']/1e3)

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16] #should be large enough to cover all of hidden at the moment
    tune_params["block_size_y"] = [2**i for i in range(10)] #powers of two

    

    results, env = tune_kernel("bpnn_layerforward_CUDA_test", "bpnn_layerforward_CUDA.cu", problem_size, args2, tune_params, metrics=metrics,
                             compiler_options=cp)

    store_results("bpnn_layerforward_CUDA.json", "bpnn_layerforward_CUDA_test", "bpnn_layerforward_CUDA.cu", tune_params, problem_size, results, env, top=3, objective="GFLOP/s")

    create_device_targets("bpnn_layerforward_CUDA.h", "bpnn_layerforward_CUDA.json", objective="GFLOP/s")



if __name__ == "__main__":

    hidden = 16
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = int(32)

    tune(size,hidden)
