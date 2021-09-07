import sys
import numpy as np
import os
from kernel_tuner import tune_kernel,run_kernel
from kernel_tuner.kernelbuilder import PythonKernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]
cmd_args = 3

def parse_args():
  with open(sys.argv[1]) as f:
        all_file_list = f.read().strip().split('\n')  # make a list of lines
        size = np.int32(all_file_list[0])
        problem_size = np.int32(size*size)
        a_list = [[float(each_int) for each_int in line.split()] for line in all_file_list[2:2+int(size)]]
        a = np.array(a_list,dtype=np.float32)
        b_list = [[float(each_int) for each_int in all_file_list[3+size].split()]] 
        b = np.array(b_list,dtype=np.float32)
        m = np.zeros(problem_size,dtype=np.float32)
        t = np.int32(0)
        args = [m,a,b,size,t]
        return args

def tune(args):
  cp_options = ["-D" + str(sys.argv[2])]
  problem_size_fan1 = args[3]
  problem_size_fan2 = (args[3],args[3])
  tune_params_fan1 = OrderedDict()
  tune_params_fan1["block_size_x"] = [2**i for i in range(8)]
  args_fan1 = [args[0],args[1],args[3],args[4]]
  metrics_fan1 = OrderedDict()
  metrics_fan1["GFLOP/s"] = lambda x: (problem_size_fan1/1e9)/(x['time']/1e3)
  print("----------- TUNING FAN1 KERNEL ----------------")
  tune_fan1, env = tune_kernel("Fan1","Fan1.cu",problem_size_fan1,args_fan1,tune_params_fan1,metrics=metrics_fan1,compiler=cp)
  print("-----------------------------------------------")
  tune_params_fan2 = OrderedDict()
  tune_params_fan2["block_size_x"] = [2**i for i in range(8)]
  tune_params_fan2["block_size_y"] = [2**i for i in range(8)]
  metrics_fan2 = OrderedDict()
  metrics_fan2["GFLOP/s"] = lambda x: (problem_size_fan1*problem_size_fan1/1e9)/(x['time']/1e3)
  args_fan2 = args
  print("----------- TUNING FAN2 KERNEL ----------------")
  tune_fan2, env = tune_kernel("Fan2","Fan2.cu",problem_size_fan2,args_fan2,tune_params_fan2,metrics=metrics_fan2,compiler=cp,compiler_options=cp_options)
  print("-----------------------------------------------")

def usage():
  print("Running the script requires 2 parameters")
  print("1) Input data file")
  print("2) OPTIMIZED or ORIGINAL depending on the kernel version you want to tune")

if __name__ == "__main__":
  if(len(sys.argv) != cmd_args):
    usage()
    exit()
  args = parse_args()
  tune(args)
  