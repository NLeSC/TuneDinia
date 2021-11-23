import sys
import numpy as np
import os
import random
from kernel_tuner import tune_kernel,run_kernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]


def create_args():
  dim = np.int32(256)
  num = np.int32(65536)
  x = np.int32(20672)
  stride = np.int32(13)
  center_table = np.random.randint(low=1,high=10,size=num)
  assign = np.random.randint(low=0,high=num,size=num)
  coord = np.random.uniform(low=0.00001, high=1, size=(num*dim))
  cost = np.random.uniform(low=30, high=40, size=num)
  weight = np.array([1] * num)
  switch_membership = np.array([True] * num)
  work_mem = np.array([0] * (stride * (num + 1)))
  args = [num,dim,x,weight,assign,cost,stride,coord,work_mem,center_table,switch_membership]
  return args

def tune(args): 
  #setup args
  computecost_args = args
  #size of the kernel
  computecost_size = args[0]
  #measure GFLOP/s
  gflops = args[0]*args[1]*3 + 3*args[0]
  #tuning parameters
  computecost_tuning = OrderedDict()
  computecost_tuning["block_size_x"] = [2**i for i in range(3,10)]
  computecost_tuning["K"] = [2*i for i in range(5)] # number of centers
  #setup metrics for kernel
  computecost_metrics = OrderedDict()
  computecost_metrics["GFLOP/s"] = lambda x: (gflops/1e9)/(x['time']/1e3)
  #add macros
  macro = " -DDIM=" + str(args[1])
  compiler = cp.append(macro)

  results, env = tune_kernel("kernel_compute_cost","compute_cost.cu",computecost_size,computecost_args,computecost_tuning,compiler=compiler,metrics=computecost_metrics)
  store_results("compute_cost.json","kernel_compute_cost","compute_cost.cu",computecost_tuning,computecost_size,results,env,top=3,objective="GFLOP/s")
  create_device_targets("compute_cost.h","compute_cost.json",objective="GFLOP/s")

if __name__ == "__main__":
  args = create_args()
  tune(args)