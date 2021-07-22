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

    params = {"block_size_x" : 64, "threads_per_node": 1}
    problem_size = np.int32(args[7])
    first_kernel_args = args

    first_kernel_results = run_kernel("Kernel", "kernel_warps.cu", problem_size, first_kernel_args, params)
    g_graph_mask = first_kernel_results[3]
    g_updating_graph_mask = first_kernel_results[4]
    g_cost = first_kernel_results[6]
    print(f"{g_graph_mask=}")
    print(f"{type(g_graph_mask)=}")

    #emulate kernel2
    g_graph_mask[g_updating_graph_mask] = True
    g_graph_visited[g_updating_graph_mask] = True
    g_updating_graph_mask[:] = False

    print(f"{g_graph_mask=}")
    print(f"{type(g_graph_mask)=}")

    #[starting, no_of_edges, graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost, num_of_nodes]
    args2 = [starting, no_of_edges, graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost, num_of_nodes]

    #g_graph_mask_ref, g_updating_graph_mask_ref, g_cost_ref = first_kernel(*args2)
    first_kernel_results = run_kernel("Kernel", "kernel_warps.cu", problem_size, args2, params)
    g_graph_mask_ref = first_kernel_results[3]
    g_updating_graph_mask_ref = first_kernel_results[4]
    g_cost_ref = first_kernel_results[6]


    answer = [None, None, None, g_graph_mask_ref, g_updating_graph_mask_ref, None, g_cost_ref, None]

    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda x: (problem_size/1e9)/(x['time']/1e3)
    refresh_args = [None for _ in args2]
    refresh_args[3] = g_graph_mask
    refresh_args[4] = g_updating_graph_mask
    refresh_args[6] = g_cost

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]
    tune_params["threads_per_node"] = [1, 2, 4, 8, 16, 32]
    grid_div_x = ["(block_size_x/threads_per_node)"]

    tune_results, env = tune_kernel("Kernel", "kernel_warps.cu", 
        problem_size, args2, tune_params, metrics=metrics, compiler_options=cp, answer=answer, grid_div_x=grid_div_x, verbose=True, observers=[AccuracyObserver(refresh_args)])
    
    store_results("bfs.json", "Kernel", "kernel_warps.cu", tune_params, problem_size, tune_results, env, top=3, objective="GFLOP/s")

    create_device_targets("bfs.h", "bfs.json", objective="GFLOP/s")


if __name__ == "__main__":
    #args = create_args()
    #args = read_graph('graph4096.txt')
    args = generate_graph(1e6, 5, 1e5)
    tune(args)
