import numpy as np
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.integration import store_results, create_device_targets
from kernel_tuner.kernelbuilder import PythonKernel
from kernel_tuner.observers import BenchmarkObserver
import os
from collections import OrderedDict

from bfs_helper import read_graph, generate_graph

cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]
class AccuracyObserver(BenchmarkObserver):

    def __init__(self, args):
        """ AccuracyObserver

        :param args: List of arguments to the kernel, corresponding to the number and types
            of arguments the kernel takes. Values that are not None are reset before
            the kernel starts. Use None for arguments that are not results nor need
            to be reset to obtain meaningful results to avoid needless data movement.
        :type args: list

        """
        self.args = args

    def before_start(self):
        for i, arg in enumerate(self.args):
            if not arg is None:
                self.dev.memcpy_htod(self.dev.allocations[i], arg)

    def get_results(self):
        return {}






def tune(args):
    starting, no_of_edges, graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost, num_of_nodes = args

    params1 = {"block_size_x" : 64}
    params2 = {"block_size_x" : 64, "threads_per_node": 1}
    problem_size = np.int32(args[7])

    first_kernel_results = run_kernel("Kernel","kernel_no_structs.cu",problem_size, args, params1)
    second_kernel_results = run_kernel("Kernel", "kernel_warps.cu", problem_size, args, params2)

    g_graph_mask_1 = first_kernel_results[3]
    g_updating_graph_mask_1 = first_kernel_results[4]
    g_cost_1 = first_kernel_results[6]
   
    g_graph_mask_2 = second_kernel_results[3]
    g_updating_graph_mask_2 = second_kernel_results[4]
    g_cost_2 = second_kernel_results[6]

    if np.allclose(g_graph_mask_1,g_graph_mask_2):
        print("g_graph_mask OK!")
    
    if np.allclose(g_updating_graph_mask_1,g_updating_graph_mask_2):
        print("g_updating_graph_mask OK!")

    if np.allclose(g_cost_1,g_cost_2):
        print("g_cost OK!")
  


if __name__ == "__main__":
    #args = create_args()
    #args = read_graph('graph4096.txt')
    args = generate_graph(1e6, 5, 1e5)
    tune(args)
