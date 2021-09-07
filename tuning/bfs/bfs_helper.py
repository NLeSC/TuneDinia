import numpy as np
from kernel_tuner import tune_kernel
import os
from collections import OrderedDict

def create_mask_arrays(num_of_nodes, source):
    #initialize all bool arrays needed for the kernel
    g_graph_mask = np.full(num_of_nodes, False)
    g_updating_graph_mask = np.full(num_of_nodes, False)
    g_graph_visited = np.full(num_of_nodes, False)
    g_graph_mask[source] = True #set this 2 indexes to true
    g_graph_visited[source] = True
    #initialize output array
    g_cost = np.full(num_of_nodes, -1, dtype=np.int32)

    return g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost

def read_graph(filename):
    with open(filename, 'r') as fh:
        file_lines = fh.read().split('\n')
        num_of_nodes = np.int32(file_lines[0])
        nodes = np.array([[int(i) for i in node.split(" ")] for node in file_lines[1:num_of_nodes+1]])
        starting = np.array(nodes[:, 0]).astype(np.int32)
        no_of_edges = np.array(nodes[:, 1]).astype(np.int32)
        source = int(file_lines[num_of_nodes+2])
        edges = int(file_lines[num_of_nodes+4])
        graph_edges = np.array([int(node.split(" ")[0]) for node in file_lines[num_of_nodes+5:-1]])

    g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost = create_mask_arrays(num_of_nodes, source)
    args = [starting, no_of_edges, graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost, num_of_nodes]
    return args

def generate_graph(number_of_nodes, min_edges, max_edges):
    num_of_nodes = np.int32(number_of_nodes)

    #no_of_edges = np.random.uniform(low=min_edges, high=max_edges, size=num_of_nodes).astype(np.int32)
    no_of_edges = min_edges+(np.random.power(0.01, size=num_of_nodes)*max_edges).astype(np.int32)
    starting = np.cumsum(no_of_edges).astype(np.int32)

    source = np.argmax(no_of_edges)

    edges = starting[-1]
    #haven't thought about duplicates yet
    graph_edges = np.random.uniform(low=0, high=num_of_nodes, size=edges).astype(np.int32)

    g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost = create_mask_arrays(num_of_nodes, source)
    args = [starting, no_of_edges, graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost, num_of_nodes]
    return args



def create_args():
    source = -1
    new_line = 1
    with open('graph4096.txt','r') as file:    
        first_line = file.readline()
        num_of_nodes = np.int32(int(first_line))
        starting = np.empty(num_of_nodes,dtype=np.int32) #initialize int* starting array
        no_of_edges = np.empty(num_of_nodes,dtype=np.int32) #initialize int* no_of_edges array
        for i in range(num_of_nodes):    
            first_line = file.readline()
            starting[i] = first_line.split()[0]    #store values in the arrays
            no_of_edges[i] = first_line.split()[1]
        first_line = file.readline()
        while new_line > 0: #get rid of empty lines on the graph.txt file
            first_line = file.readline()
            if(first_line == "\n"):
                new_line = new_line - 1
            elif(new_line == 1):
                source = int(first_line)    #find head node
        first_line = file.readline()
        edges = int(first_line)
        graph_edges = np.empty(edges,dtype=np.int32)
        for i in range(edges):
            first_line = file.readline()
            graph_edges[i] = first_line.split()[0]
    #initialize all bool arrays needed for the kernel
    g_graph_mask = np.full(num_of_nodes,False)
    g_updating_graph_mask = np.full(num_of_nodes,False)
    g_graph_visited = np.full(num_of_nodes,False)
    g_graph_mask[source] = True #set this 2 indexes to true
    g_graph_visited[source] = True
    #initialize output array
    g_cost = np.empty(num_of_nodes,dtype=np.int32)
    for i in range(num_of_nodes):
        g_cost[i] = -1
    args = [starting, no_of_edges, graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_cost, num_of_nodes]
    return args

def tune(args):
    tune_params = dict()
    tune_params["block_size_x"] = [64, 128, 256, 512]
    problem_size = np.int32(args[7])
    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda x: (problem_size/1e9)/(x['time']/1e3)
    result = tune_kernel("Kernel", "kernel.cu", problem_size, args, tune_params, metrics=metrics, compiler_options=["-I" + os.getcwd()])

if __name__ == "__main__":
    read_graph('graph4096.txt')
   #args = create_args()
   #tune(args)
