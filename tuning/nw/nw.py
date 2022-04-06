import numpy as np
import os
import sys
from collections import OrderedDict
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]

def verify_kernel_1():
    #get input data
    cols, problem_size, reference, input_itemsets, penalty = get_input_data()
    #setup exec parameters
    params = {"block_size_x" : 16, "BLOCK_SIZE" : 16}
    #setup args
    iteration_orig = np.int32(problem_size/params["BLOCK_SIZE"])
    iteration = np.int32(problem_size/params["block_size_x"])
    args_orig = [reference,input_itemsets,cols,penalty,iteration_orig]
    args = [reference,input_itemsets,cols,penalty,iteration]
    #run both kernels
    default_output = run_kernel("needle_cuda_shared_1","needle_shared_orig1.cu",problem_size,args_orig,params)
    tuned_output = run_kernel("needle_cuda_shared_1","needle_shared_1.cu",problem_size,args,params)
    #compare outputs
    if(np.allclose(default_output[1],tuned_output[1])):
        print("VERIFICATION SUCCEEDED FOR KERNEL 1") 
    else:
        print("VERIFICATION FAILED FOR KERNEL 1")

def verify_kernel_2():
    #get input data
    cols, problem_size, reference, input_itemsets, penalty = get_input_data()
    #setup exec parameters
    params = {"block_size_x" : 8, "BLOCK_SIZE" : 8}
    #setup args
    block_width_orig = np.int32(problem_size/params["BLOCK_SIZE"])
    block_width = np.int32(problem_size/params["block_size_x"])
    iteration_orig = np.int32(block_width_orig - 1)
    iteration = np.int32(block_width -1)
    args_orig = [reference,input_itemsets,cols,penalty,iteration_orig,block_width_orig]
    args = [reference,input_itemsets,cols,penalty,iteration,block_width]
    #run both kernels
    default_output = run_kernel("needle_cuda_shared_2","needle_shared_orig2.cu",problem_size,args_orig,params)
    tuned_output = run_kernel("needle_cuda_shared_2","needle_shared_2.cu",problem_size,args,params)
    #compare outputs
    if(np.allclose(default_output[1],tuned_output[1])):
        print("VERIFICATION SUCCEEDED FOR KERNEL 2") 
    else:
        print("VERIFICATION FAILED FOR KERNEL 2")

    

def get_tunable_parameters():  
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(0,8)]
    return tune_params


def get_input_data(size):
    rows = cols = np.int32(size)
    problem_size = cols - 1
    array_size = rows * cols
    reference = np.random.randint(low=-10,high=0,size=array_size,dtype=np.int32)
    input_itemsets = np.random.randint(low=-10,high=0,size=array_size,dtype=np.int32)
    penalty = np.int32(10)
    return cols, problem_size, reference, input_itemsets, penalty

def tune(size):
    cols, problem_size, reference, input_itemsets, penalty = get_input_data(size)

    args = [reference,input_itemsets,cols,penalty]
    
    tune_params = get_tunable_parameters()
    
    print("--------------TUNING KERNEL 1---------------")
    results, env = tune_kernel("needle_cuda_shared_1","needle_shared_1.cu",problem_size,args,tune_params,compiler_options=cp)
    store_results("needle_cuda_shared_1.json", "needle_cuda_shared_1", "needle_shared_1.cu", tune_params, problem_size, results, env, top=3)
    create_device_targets("needle_cuda_shared_1.h", "needle_cuda_shared_1.json")

    print("-------------TUNING KERNEL 2----------------")
    tune_kernel("needle_cuda_shared_2","needle_shared_2.cu",problem_size,args,tune_params,compiler_options=cp)
    store_results("needle_cuda_shared_2.json", "needle_cuda_shared_2", "needle_shared_2.cu", tune_params, problem_size, results, env, top=3)
    create_device_targets("needle_cuda_shared_2.h", "needle_cuda_shared_2.json")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = 2049
    tune(size)