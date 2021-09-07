import sys
import numpy as np
import os
from kernel_tuner import tune_kernel,run_kernel
from kernel_tuner.kernelbuilder import PythonKernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]

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

def run(args):
  params_fan1 = {"block_size_x" : 256}
  params_fan2 = {"block_size_x" : 4,"block_size_y" : 4}
  args_fan1 = [args[0],args[1],args[3],args[4]]
  size_fan1 = args[3]
  size_fan2 = (args[3],args[3])
  fan1_output = run_kernel("Fan1","Fan1.cu",size_fan1,args_fan1,params_fan1)
  args_fan2 = [fan1_output[0],fan1_output[1],args[2],args[3],args[4]]
  fan2_org_output = run_kernel("Fan2","Fan2_org.cu",size_fan2,args_fan2,params_fan2)
  fan2_output = run_kernel("Fan2","Fan2.cu",size_fan2,args_fan2,params_fan2)
  if(np.allclose(fan2_org_output[2],fan2_output[2])):
    print("TEST PASSED")
  else:
    print("TEST FAILED")


if __name__ == "__main__":
  args = parse_args()
  run(args)
  