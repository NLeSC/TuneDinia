import sys
import numpy as np
import os
from kernel_tuner import tune_kernel,run_kernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]

def create_args(num_features, num_points, num_clusters):
  feature_flipped_d_list = np.zeros(num_features*num_points,dtype=np.float32)
  feature_flipped_d_list[0:(num_features*num_points)] = np.random.random(num_features*num_points)
  feature_flipped_d = np.array(feature_flipped_d_list,dtype=object)
  features = np.zeros(num_features*num_points,dtype=np.float32)
  features[0:(num_features*num_points)] = np.random.random(num_features*num_points)
  membership = np.full(num_points,-1,np.int32)
  clusters = np.zeros((num_clusters*num_features),np.float32)
  args = [num_features,num_points,num_clusters,features,membership,clusters,feature_flipped_d]
  return args
    
def tune(args):

  json_file = "kmeans.json"
  header_file = "kmeans.h"

  block_clusters_size = np.int32(args[2]*args[1]*args[0])
  block_deltas_size = np.int32(args[1])
  block_deltas = np.zeros(block_deltas_size,np.int32)
  block_clusters = np.zeros(block_clusters_size,np.float32)
  kmeanspoint_args = [args[3],args[4],args[5],block_deltas,block_clusters,args[6]]  #run the debug script to verify the corectness of the implementation
  kmeanspoint_size = args[1] * args[2]
  #flops in the kernel
  gflops = 3 * args[0] * args[1] * args[2] + args[1] + args[0]*args[2]
  # set of tunable parameters
  kmeanspoint_params_tuning = OrderedDict()
  kmeanspoint_params_tuning["block_size_x"] = [2**i for i in range(3,10)]
  kmeanspoint_params_tuning["block_size_y"] = [1] 
  kmeanspoint_params_tuning["NCLUSTERS"] = [args[2]]
  kmeanspoint_params_tuning["NPOINTS"] = [args[1]]
  kmeanspoint_params_tuning["NFEATURES"] = [args[0]]  
  kmeanspoint_params_tuning["read_only"] = [0,1]
  kmeanspoint_params_tuning["USE_CUB"] = [0,1]
  kmeanspoint_metrics = OrderedDict()
  kmeanspoint_metrics["GFLOP/s"] = lambda x: (gflops/1e9)/(x['time']/1e3)
  #tune the second kernel 
  print("-------- TUNING KMEANS POINT KERNEL ------------")
  results, env = tune_kernel("kmeansPoint","kmeanspoint.cu",kmeanspoint_size,kmeanspoint_args,kmeanspoint_params_tuning,metrics=kmeanspoint_metrics,compiler=cp)
  print("-------------------------------------------------")
  #storing the results
  store_results(json_file,"kmeansPoint","kmeanspoint.cu",kmeanspoint_params_tuning,kmeanspoint_size,results,env,top=3,objective="GFLOP/s")
  create_device_targets(header_file,json_file,objective="GFLOP/s")

if __name__ == "__main__":
  if len(sys.argv) > 3:
      points = int(sys.argv[1])
      features = int(sys.argv[2])
      clusters = int(sys.argv[3])
  else:
      points = int(100000)
      features = int(30)
      clusters = int(6)
  
  args = create_args(features, points, clusters)
  tune(args)
