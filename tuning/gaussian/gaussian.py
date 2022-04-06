import sys
import numpy as np
import os
from kernel_tuner import tune_kernel,run_kernel
from kernel_tuner.kernelbuilder import PythonKernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]
cmd_args = 3

def create_problem_size():
  init_size = 200
  sizes_list = [init_size]
  for i in range(1,9):
    sizes_list.append(sizes_list[i-1]*2)
  return sizes_list

def create_args(sizes):
  b_size = np.int32(sizes)
  print(sizes)
  a_size = np.int32(sizes*sizes)
  a = np.zeros(a_size,dtype=np.float32)
  a[0:a_size] = np.random.random(a_size)
  b = np.zeros(b_size,dtype=np.float32)
  b[0:b_size] = np.random.random(b_size)
  m = np.zeros(b_size * b_size,dtype=np.float32)
  m[0:(b_size*b_size)] = np.random.random(b_size*b_size)
  t = np.int32(0)
  args = [m,a,b,b_size,t]
  return args

def tune(args):
  json_file = "gaussian.json"
  header_file = "gaussian.h"
  cp_options = ["-D" + str(sys.argv[1])]
  problem_size_fan1 = args[3]
  problem_size_fan2 = (args[3],args[3])
  tune_params_fan1 = OrderedDict()
  tune_params_fan1["block_size_x"] = [2**i for i in range(8)]
  args_fan1 = [args[0],args[1],args[3],args[4]]
  metrics_fan1 = OrderedDict()
  metrics_fan1["GFLOP/s"] = lambda x: ((problem_size_fan1-1)/1e9)/(x['time']/1e3)
  print("----------- TUNING FAN1 KERNEL ----------------")
  results1, env = tune_kernel("Fan1","Fan1.cu",problem_size_fan1,args_fan1,tune_params_fan1,metrics=metrics_fan1,compiler=cp)
  print("-----------------------------------------------")
  tune_params_fan2 = OrderedDict()
  tune_params_fan2["block_size_x"] = [2**i for i in range(8)]
  tune_params_fan2["block_size_y"] = [2**i for i in range(8)]
  metrics_fan2 = OrderedDict()
  metrics_fan2["GFLOP/s"] = lambda x: (((2*problem_size_fan1*problem_size_fan1) + (problem_size_fan1*problem_size_fan1)/8)/1e9)/(x['time']/1e3)
  fan1_params = {"block_size_x" : 32}
  fan1_output = run_kernel("Fan1","Fan1.cu",problem_size_fan1,args_fan1,params=fan1_params)
  args_fan2 = [fan1_output[0],args[1],args[2],args[3],args[4]]
  #tune the kernel and store the results
  print("----------- TUNING FAN2 KERNEL ----------------")
  results2, env = tune_kernel("Fan2","Fan2.cu",problem_size_fan2,args_fan2,tune_params_fan2,metrics=metrics_fan2,compiler=cp,compiler_options=cp_options)
  store_results(json_file,"Fan2","Fan2.cu",tune_params_fan2,problem_size_fan2,results2,env,top=3,objective="GFLOP/s")
  create_device_targets(header_file,json_file,objective="GFLOP/s")
  print("-----------------------------------------------")

def usage():
  print("Running the script requires a parameter")
  print("OPTIMIZED or ORIGINAL depending on the kernel version you want to tune")

if __name__ == "__main__":
  if(len(sys.argv) != cmd_args):
    usage()
    exit()
  if len(sys.argv) > 2:
      size = int(sys.argv[2])
  else:
      size = int(20000)
  args = create_args(size)
  tune(args)
  