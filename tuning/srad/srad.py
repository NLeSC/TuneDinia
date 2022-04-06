from typing import OrderedDict
import numpy as np
from kernel_tuner import run_kernel, tune_kernel
import os
import sys
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]

def create_args_kernel_1(size):
  rows = np.int32(size)
  cols = np.int32(size)
  q0sqr = np.float32(3.14)
  size_I = np.int32(rows*cols)

  J = np.zeros(size_I,dtype=np.float32)
  J[0:size_I] = np.random.random(size_I)
  E_C = np.zeros(size_I,dtype=np.float32)
  W_C = np.zeros(size_I,dtype=np.float32)
  S_C = np.zeros(size_I,dtype=np.float32)
  N_C = np.zeros(size_I,dtype=np.float32)
  C = np.zeros(size_I,dtype=np.float32)

  return_values = [E_C, W_C, N_C, S_C, J, C, cols, rows, q0sqr]
  
  return return_values

def create_args_kernel_2(size):
  rows = np.int32(size)
  cols = np.int32(size)
  q0sqr = np.float32(3.14)
  size_I = np.int32(rows*cols)
  lamda = np.float32(np.random.random())

  J = np.zeros(size_I,dtype=np.float32)
  E_C = np.zeros(size_I,dtype=np.float32)
  E_C[0:size_I] = np.random.random(size_I)
  W_C = np.zeros(size_I,dtype=np.float32)
  W_C[0:size_I] = np.random.random(size_I)
  S_C = np.zeros(size_I,dtype=np.float32)
  S_C[0:size_I] = np.random.random(size_I)
  N_C = np.zeros(size_I,dtype=np.float32)
  N_C[0:size_I] = np.random.random(size_I)
  C = np.zeros(size_I,dtype=np.float32)
  C[0:size_I] = np.random.random(size_I)

  return_values = [E_C, W_C, N_C, S_C, J, C, cols, rows, lamda, q0sqr]
  
  return return_values


def tune_kernel_1(size):
  #get kernel arguments
  kernel_args = create_args_kernel_1(size)
  #setup kernel parameters for the original kernel
  kernel_params_orig = {"block_size_x" : 16, "block_size_y" : 16, "BLOCK_SIZE" : 16}
  kernel_params = {"block_size_x" : 16, "block_size_y" : 16, "tile_size_x" : 2, "tile_size_y" : 2}
  #define the problem size
  problem_size = (kernel_args[6],kernel_args[7])
  #setup tunable parameters
  tune_params = OrderedDict()
  tune_params["block_size_x"] = [16, 32 ,64]
  tune_params["block_size_y"] = [16, 32 ,64]
  tune_params["tile_size_y"] = [1, 2, 4 ,8]
  tune_params["tile_size_x"] = [1, 2, 4, 8]
  tune_params["grid_width"] = [kernel_args[6]]
  tune_params["grid_height"] = [kernel_args[7]]
  grid_div_x = ["block_size_x", "tile_size_x"]
  grid_div_y = ["block_size_y", "tile_size_y"]
  #run the original kernel for output verification
  #output_orig = run_kernel("srad_cuda_1","srad_orig_1.cu",problem_size,kernel_args,kernel_params_orig)
  #setup metrics for the kernel
  gflops = 28 * kernel_args[6] * kernel_args[7]
  srad_1_metrics = OrderedDict()
  srad_1_metrics["GFLOP/s"] = lambda x: (gflops/1e9)/(x['time']/1e3)
  # #tune the kernel and verify the output
  # #answer = [output_orig[0], output_orig[1], output_orig[2], output_orig[3], None, output_orig[5], None, None, None]
  tune_kernel("srad_cuda_1","srad_1.cu",problem_size,kernel_args,tune_params,grid_div_x=grid_div_x,grid_div_y=grid_div_y, metrics=srad_1_metrics, verbose=True)
  # results, env = tune_kernel("srad_cuda_1","srad_1.cu",problem_size,kernel_args,tune_params,grid_div_x=grid_div_x,grid_div_y=grid_div_y, metrics=srad_1_metrics, verbose=True)
  # store_results("srad_1.json", "srad_cuda_1", "srad_1_tiling.cu", tune_params, problem_size, results, env, top=3, objective="GFLOP/s")
  # create_device_targets("srad_1.h","srad_1.json",objective="GFLOP/s")


def tune_kernel_2(size):
  #get kernel arguments
  kernel_args = create_args_kernel_2(size)
  #setup kernel parameters for the original kernel
  kernel_params_orig = {"block_size_x" : 16, "block_size_y" : 16, "BLOCK_SIZE" : 16}
  #define the problem size
  problem_size = (kernel_args[6],kernel_args[7])
  #setup tunable parameters
  tune_params = OrderedDict()
  tune_params["block_size_x"] = [16, 32 ,64]
  tune_params["block_size_y"] = [16, 32 ,64]
  tune_params["tile_size_y"] = [1, 2, 4 ,8]
  tune_params["tile_size_x"] = [1, 2, 4, 8]
  grid_div_x = ["block_size_x", "tile_size_x"]
  grid_div_y = ["block_size_y", "tile_size_y"]
  #setup metrics for the kernel
  gflops = 10 * kernel_args[6] * kernel_args[7]
  srad_1_metrics = OrderedDict()
  srad_1_metrics["GFLOP/s"] = lambda x: (gflops/1e9)/(x['time']/1e3)
  #tune the kernel and verify the output
  results, env = tune_kernel("srad_cuda_2","srad_2_tiling.cu",problem_size,kernel_args,tune_params,grid_div_x=grid_div_x,grid_div_y=grid_div_y,metrics=srad_1_metrics)
  store_results("srad_2.json", "srad_cuda_2", "srad_2_tiling.cu", tune_params, problem_size, results, env, top=3, objective="GFLOP/s")
  create_device_targets("srad_2.h","srad_2.json",objective="GFLOP/s")

if __name__ == "__main__":
  if len(sys.argv) > 1:
        size = int(sys.argv[1])
  else:
        size = 4096
  print("----------- TUNING SRAD 1 KERNEL -------------------")
  tune_kernel_1(size)
  print("----------------------------------------------------")
  print("----------- TUNING SRAD 2 KERNEL -------------------")
  tune_kernel_2(size)
  print("----------------------------------------------------")