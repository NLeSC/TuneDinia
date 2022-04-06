#!/usr/bin/env python
from typing import OrderedDict
import numpy as np
from math import cos,sin
from kernel_tuner import run_kernel, tune_kernel
import os
import sys
from kernel_tuner.observers import BenchmarkObserver
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]
# GICOV =  Gradient Inverse Coefficient of Variation 

def create_args_GICOV(gradn, gradm):

  npoints = 150
  min_rad = np.int32(8)
  ncircles = 10
  pi = np.float32(3.14159)
  grad_m = np.int32(gradm)
  grad_n = np.int32(gradn)
  grad_mem_size = grad_n * grad_m
  offset = np.int32(10)
  problem_size = (grad_m * grad_n)

  tX = np.zeros(npoints*ncircles,dtype=np.int32)
  tY = np.zeros(npoints*ncircles,dtype=np.int32)
  theta = np.zeros(npoints,dtype=np.float32)
  sin_angle = np.zeros(npoints,dtype=np.float32)
  cos_angle = np.zeros(npoints,dtype=np.float32)
  gicov = np.zeros(grad_mem_size,dtype=np.float32)
  grad_x = np.zeros(grad_mem_size,dtype=np.float32)
  grad_x[offset:(grad_mem_size-offset)] = np.random.random(grad_mem_size-2*offset) + 10
  grad_y = np.zeros(grad_mem_size,dtype=np.float32)
  grad_y[offset:(grad_mem_size-offset)] = np.random.random(grad_mem_size-2*offset) + 10

  for n in range(0,npoints):
    theta[n] = np.float32((n * 2.0 * pi) / npoints)
    sin_angle[n] = np.float32(sin(theta[n]))
    cos_angle[n] = np.float32(cos(theta[n]))

  for k in range(0,ncircles):
    rad = np.double(min_rad + 2*k)
    for n in range(0,npoints):
      tX[k * npoints + n] = np.int32(cos(theta[n]) * rad)
      tY[k * npoints + n] = np.int32(sin(theta[n]) * rad)

  return_values = [grad_m, gicov, grad_x, grad_y, cos_angle, sin_angle, tX, tY, problem_size, ncircles, npoints]
     
  return return_values

def create_args_dilate(gradn, gradm):
  strel_m = np.int32(25)
  strel_n = np.int32(25)
  grad_m = np.int32(gradm)
  grad_n = np.int32(gradn)

  dilated = np.zeros((grad_m*grad_n),dtype=np.float32)
  strel = np.empty((strel_m*strel_n),dtype=np.float32)
  gicov = np.empty((grad_m*grad_n),dtype=np.float32)

  return_values = [grad_m,grad_n,strel_m,strel_n,dilated,strel,gicov]
  return return_values

def tune_GICOV(gradn, gradm):
  # setup args for GICOV kernel
  gicov_kernel_args = create_args_GICOV(gradn, gradm)
  problem_size = gicov_kernel_args[8]
  ncircles = gicov_kernel_args[9]
  npoints = gicov_kernel_args[10]
  kernel_args = [gicov_kernel_args[1],gicov_kernel_args[2],gicov_kernel_args[3],gicov_kernel_args[4],gicov_kernel_args[5],gicov_kernel_args[6],gicov_kernel_args[7]]
  # metrics for GICOV kernel
  gicov_metrics = OrderedDict()
  gicov_gflops = 17 * ncircles * npoints * problem_size
  gicov_metrics["GFLOP/s"] = lambda x: (gicov_gflops/1e9)/(x['time']/1e3)
  #setup constant memory args
  cmem_args = {'c_tX' : gicov_kernel_args[6], 'c_tY' : gicov_kernel_args[7], "c_cos_angle" : gicov_kernel_args[4], "c_sin_angle" : gicov_kernel_args[5]}
  # tunable parameters for GICOV kernel
  gicov_tuning_params = OrderedDict()
  gicov_tuning_params["block_size_x"] = [2**i for i in range(3,11)]
  gicov_tuning_params["USE_CONSTANT"] = [0,1]
  gicov_tuning_params["NCIRCLES"] = [ncircles]
  gicov_tuning_params["NPOINTS"] = [npoints]
  # tune the GICOV kernel
  results, env = tune_kernel("GICOV_kernel", "gicov.cu", problem_size, kernel_args, gicov_tuning_params, compiler=cp,metrics=gicov_metrics, cmem_args=cmem_args)
  store_results("gicov.json", "GICOV_kernel" ,"gicov.cu", gicov_tuning_params, problem_size, results,env,top=3,objective="GFLOP/s")
  create_device_targets("gicov.h","gicov.json",objective="GFLOP/s")

def tune_dilate():
  # setup args for dilate kernel
  kernel_args = create_args_dilate(gradn, gradm)
  dilate_kernel_args = [kernel_args[4],kernel_args[5],kernel_args[6]]
  problem_size = kernel_args[0] * kernel_args[1]
  # preprocessor macros for GICOV kernel
  macro_1 = " -DSTREL_M=" + str(kernel_args[2]) + " -DSTREL_N=" + str(kernel_args[3])
  macro_2 = " -DIMG_M=" + str(kernel_args[0]) + " -DIMG_N=" + str(kernel_args[1])
  cp.append(macro_1)
  cp.append(macro_2)
  compiler = cp
  # metrics for dilate kernel
  dilate_metrics = OrderedDict()
  dilate_gflops = 2 * problem_size
  dilate_metrics["GFLOP/s"] = lambda x: (dilate_gflops/1e9)/(x['time']/1e3)
  # tunable parameters for dilate kernel
  dilate_tuning_params = OrderedDict()
  dilate_tuning_params["block_size_x"] = [2**i for i in range(3,11)]
  dilate_tuning_params["USE_CONSTANT"] = [0,1]
  # tune the dilate kernel
  results, env = tune_kernel("dilate_kernel","dilate_kernel.cu",problem_size,dilate_kernel_args,dilate_tuning_params,compiler=compiler,metrics=dilate_metrics)
  store_results("dilate.json", "dilate_kernel" ,"dilate_kernel.cu", dilate_tuning_params, problem_size, results,env,top=3,objective="GFLOP/s")
  create_device_targets("dilate.h","dilate.json",objective="GFLOP/s")


if __name__ == "__main__":

  if len(sys.argv) > 2:
      gradn = int(sys.argv[1])
      gradm = int(sys.argv[2])
  else:
      gradn = int(640)
      gradm = int(219)
  print("\n------- TUNING GICOV KERNEL -------------")
  tune_GICOV(gradn, gradm)
  print("-----------------------------------------\n")
  print("------- TUNING DILATE KERNEL ------------")
  tune_dilate(gradn, gradm)
  print("-----------------------------------------\n")


