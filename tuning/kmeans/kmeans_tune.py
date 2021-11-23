import sys
import numpy as np
import os
from kernel_tuner import tune_kernel,run_kernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets
from kmeans_debug import run

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]

def parse_args():
  with open(sys.argv[1]) as f:
    all_file_list = f.read().strip().split('\n')  # make a list of lines
    npoints = np.int32(len(all_file_list))
    feature_flipped_d_list = [np.fromstring(line, dtype=np.float32, sep=" ") for line in all_file_list]
  for i in range(npoints):
    feature_flipped_d_list[i].tolist().pop(0) 
  nfeatures = np.int32(len(feature_flipped_d_list[0]))
  feature_flipped_d = np.array(feature_flipped_d_list,dtype=np.float32)
  nclusters = np.int32(5)
  membership = np.full(npoints,-1,np.int32)
  clusters = np.zeros((nclusters*nfeatures),np.float32)
  args = [feature_flipped_d,nfeatures,npoints,nclusters,membership,clusters]
  return args
    
def tune(args):
  invert_mapping_args = [args[0],args[0],args[2],args[1]]
  invert_mapping_size = np.int32(args[2])
  invert_params_tuning = OrderedDict()
  invert_params_tuning["block_size_x"] = [2**i for i in range(8)]
  #add macros to compiler to unroll loops
  macro = " -DNFEATURES=" + str(args[1])
  compiler = cp.append(macro)
  #tune the first kernel
  print("-------- TUNING INVERT MAPING KERNEL ------------")
  tune_inv, env = tune_kernel("invert_mapping","invert_mapping.cu",invert_mapping_size,invert_mapping_args,invert_params_tuning,compiler=cp)
  print("-------------------------------------------------")
  invert_mapping_params = {"block_size_x" : 256}
  invert_mapping_output = run_kernel("invert_mapping","invert_mapping.cu",invert_mapping_size,invert_mapping_args,invert_mapping_params)
  block_clusters_size = np.int32(args[2]*args[3]*args[1])
  block_deltas_size = np.int32(args[2])
  block_deltas = np.zeros(block_deltas_size,np.int32)
  block_clusters = np.zeros(block_clusters_size,np.float32)
  kmeanspoint_args = [invert_mapping_output[1],args[1],args[2],args[3],args[4],args[5],block_deltas,block_clusters,args[0]]  #run the debug script to verify the corectness of the implementation
  kmeanspoint_size = (np.int32(args[2])*np.int32(args[3]))
  #flops in the kernel
  gflops = 3*np.int32(args[1])*np.int32(args[2])*np.int32(args[3]) + np.int32(args[2]) + np.int32(args[1])*np.int32(args[3])
  # set of tunable parameters
  kmeanspoint_params_tuning = OrderedDict()
  kmeanspoint_params_tuning["block_size_x"] = [2**i for i in range(3,10)]
  kmeanspoint_params_tuning["block_size_y"] = [1] 
  kmeanspoint_params_tuning["NCLUSTERS"] = [5]
  kmeanspoint_params_tuning["read_only"] = [0]
  kmeanspoint_params_tuning["USE_CUB"] = [0]
  kmeanspoint_metrics = OrderedDict()
  kmeanspoint_metrics["GFLOP/s"] = lambda x: (gflops/1e9)/(x['time']/1e3)
  #tune the second kernel 
  print("-------- TUNING KMEANS POINT KERNEL ------------")
  results, env = tune_kernel("kmeansPoint","kmeanspoint.cu",kmeanspoint_size,kmeanspoint_args,kmeanspoint_params_tuning,metrics=kmeanspoint_metrics,compiler=compiler)
  print("-------------------------------------------------")
  #storing the results
  store_results("kmeanspoint.json","kmeansPoint","kmeanspoint.cu",kmeanspoint_params_tuning,kmeanspoint_size,results,env,top=3,objective="GFLOP/s")
  create_device_targets("kmeanspoint.h","kmeanspoint.json",objective="GFLOP/s")

if __name__ == "__main__":
  args = parse_args()
  tune(args)
