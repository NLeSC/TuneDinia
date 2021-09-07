# main.py
import sys
import numpy as np
import os
from copy import deepcopy
from kernel_tuner import tune_kernel,run_kernel
from kernel_tuner.kernelbuilder import PythonKernel
from collections import OrderedDict
from kernel_tuner.integration import store_results, create_device_targets
# maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
MAX_PD = 3.0e6
# required precision in degrees	*/
PRECISION = 0.001
SPEC_HEAT_SI = 1.75e6
K_SI = 100
# capacitance fitting factor	*/
FACTOR_CHIP = 0.5
EXPAND_RATE = 2

BLOCK_SIZE  = 16
# chip parameters	*/
t_chip = 0.0005
chip_height = 0.016
chip_width = 0.016
# ambient temperature, assuming no package at all	*/
amb_temp = 80.0
cp = ["-I" + os.path.dirname(os.path.realpath(__file__))]

def parse_args():
    grid_rows = np.int32(sys.argv[1])
    grid_cols = np.int32(sys.argv[1])
    pyramid_height  = np.int32(sys.argv[2])
    total_iterations = np.int32(sys.argv[3])
    #store the temperatures from the file
    with open(sys.argv[4]) as f:
        all_file_list = f.read().strip().split('\n')  # make a list of lines
        temp_data_list = [[float(each_int) for each_int in line.split()] for line in all_file_list]  
        temp_data_src = np.array(temp_data_list,dtype=np.float32)
        temp_data_dst = deepcopy(temp_data_src)
        print("size of src_array -> ",len(temp_data_src))
        print("size of dst_array -> ",len(temp_data_dst))
    #store the power measurements from the file
    with open(sys.argv[5]) as f:
        all_file_list = f.read().strip().split('\n')  # make a list of lines
        power_data_list = [[float(each_int) for each_int in line.split()] for line in all_file_list] 
        power_data =  np.array(power_data_list).astype(np.float32)
    borderCols = np.int32((pyramid_height)*EXPAND_RATE/2)
    borderRows = np.int32((pyramid_height)*EXPAND_RATE/2)
    smallBlockCol = np.int32(BLOCK_SIZE-(pyramid_height)*EXPAND_RATE)
    smallBlockRow = np.int32(BLOCK_SIZE-(pyramid_height)*EXPAND_RATE)
    blockCols = np.int32(grid_cols/smallBlockCol)
    if (grid_cols%smallBlockCol != 0):
        blockCols = blockCols + 1    
    blockRows = np.int32(grid_rows/smallBlockRow)
    if(grid_rows%smallBlockRow != 0):
        blockRows = blockRows + 1
    problem_size = np.int32(blockCols*blockRows)
    #compute the constants
    grid_height = np.float32(chip_height/grid_rows)
    grid_width = np.float32(chip_width/grid_cols)
    Cap = np.float32(FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height)
    Rx = np.float32(grid_width / (2.0 * K_SI * t_chip * grid_height))
    Ry = np.float32(grid_height / (2.0 * K_SI * t_chip * grid_width))
    Rz = np.float32(t_chip / (K_SI * grid_height * grid_width))
    max_slope = np.float32(MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI))
    step = np.float32(PRECISION / max_slope)
    args = [total_iterations,power_data,temp_data_src,temp_data_dst,grid_cols,grid_rows,borderCols,borderRows,Cap,Rx,Ry,Rz,step]
    return args,problem_size

def tune(args,problem_size):
    
    params_naive = {"block_size_x" : 16,"block_size_y" : 16, "tile_size_y" : 2}

    hotspot_ref_output = run_kernel("calculate_temp","hotspot_original_kernel.cu",problem_size,args,params_naive)

    hotspot_output = run_kernel("calculate_temp","hotspot_kernel.cu",problem_size,args,params_naive)
    print("size of output -> ",len(hotspot_output[3]))
    print("size of output -> ",len(hotspot_output[3]))
    if np.allclose(hotspot_ref_output[3],hotspot_output[3]):
        print("TEST PASSED")
    else:
        print("TEST FAILED")
    col = 0
    row = 0
    with open(sys.argv[6], 'w') as f:
        for i in range(len(hotspot_output[3])):
            f.write("[" + str(row) + "," + str(col) + "]" + " " + str(hotspot_output[3][i]) + "\n")
            col = col + 1
            if col  == np.int32(sys.argv[1]):
                row = row + 1
                col = 0
                
    col = 0
    row = 0
    with open(sys.argv[7], 'w') as f:
        for i in range(len(hotspot_ref_output[3])):
            f.write("[" + str(row) + "," + str(col) + "]" + " " + str(hotspot_ref_output[3][i]) + "\n")
            col = col + 1
            if col == np.int32(sys.argv[1]):
                row = row + 1
                col = 0
               
            


if __name__ == "__main__":
    args,problem_size = parse_args()
    tune(args,problem_size)
    