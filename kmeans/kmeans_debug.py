import sys
import numpy as np
import os
import math
from kernel_tuner import tune_kernel,run_kernel
from kernel_tuner.kernelbuilder import PythonKernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets

def parse_args():
  with open(sys.argv[1]) as f:
    all_file_list = f.read().strip().split('\n')  # make a list of lines
    npoints = np.int32(len(all_file_list))
    feature_flipped_d_list = [[np.float32(each_int) for each_int in line.split(" ")] for line in all_file_list]
  for i in range(npoints):
    feature_flipped_d_list[i].pop(0) 
  nfeatures = np.int32(len(feature_flipped_d_list[0]))
  feature_flipped_d = np.array(feature_flipped_d_list,dtype=np.float32)
  nclusters = np.int32(5) #hardcoded parameter should be runtime variable. I set it to 5 based on a test case to verify correctness of implementation
  membership = np.full(npoints,-1,np.int32)
  clusters = np.zeros((nclusters*nfeatures),np.float32)
  args = [feature_flipped_d,nfeatures,npoints,nclusters,membership,clusters]
  return args
    
def run(args):
  invert_mapping_args = [args[0],args[0],args[2],args[1]]
  invert_mapping_size = np.int32(args[2])
  invert_mapping_params = {"block_size_x" : 256}
  invert_mapping_output = run_kernel("invert_mapping","invert_mapping.cu",invert_mapping_size,invert_mapping_args,invert_mapping_params)
  kmeanspoint_params = {"block_size_x" : 256, "block_size_y" : 1}
  kmeanspoint_size = (np.int32(args[1]*args[3]))
  num_blocks = np.int32(math.ceil(kmeanspoint_size/kmeanspoint_params["block_size_x"]))
  num_blocks_per_dim = np.int32(math.sqrt(num_blocks))
  block_clusters_size = np.int32(args[2]*args[3]*args[1])
  block_deltas_size = np.int32(args[2])
  block_deltas = np.zeros(block_deltas_size,np.int32)
  block_clusters = np.zeros(block_clusters_size,np.float32)
  kmeanspoint_args = [invert_mapping_output[1],args[1],args[2],args[3],args[4],args[5],block_deltas,block_clusters,args[0]]
  kmeanspoint_output = run_kernel("kmeansPoint","kmeanspoint.cu",kmeanspoint_size,kmeanspoint_args,kmeanspoint_params)
  return kmeanspoint_output


if __name__ == "__main__":
  args = parse_args()
  run(args)
