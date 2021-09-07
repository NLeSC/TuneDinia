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

    args1 = [input_cuda, output_hidden_cuda, input_hidden_cuda, hidden_partial_sum_16, in_size, hid]
    args2 = [input_cuda, input_hidden_cuda, hidden_partial_sum_32, in_size, hid]

    params1 = {"block_size_x" : 16,"block_size_y": 16}
    params2 = {"block_size_x" : 16,"block_size_y": 32}
    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda x: (size/1e9)/(x['time']/1e3)

    results = kernel_tuner.run_kernel("bpnn_layerforward_CUDA", "backprop_cuda_kernel.cu", problem_size, args1, params1,
                                      compiler_options=cp)
    print(results[3])
    answer = [None, None, results[3], None, None]

    results_test = kernel_tuner.run_kernel("bpnn_layerforward_CUDA_test", "bpnn_layerforward_CUDA.cu", problem_size, args2, params2,
                                            compiler_options=cp)

    print(results_test[2])
    print("TEST PASSED" if np.allclose(np.sum(results[3].reshape(num_blocks_16, hidden), axis=0),
                                       np.sum(results_test[2].reshape(len(hidden_partial_sum_32)//hidden, hidden), axis=0)) else "TEST FAILED")

    


if __name__ == "__main__":

    hidden = 16
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = int(32)

    tune(size,hidden)
