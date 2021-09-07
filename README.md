## TuneDinia

To study the effect of different code optimizations and other tunable parameters on GPU energy efficiency it is crucial to have a representative set of tunable applications. Existing benchmark suites, e.g. PARSEC and Rodinia, are straightforward implementations that are not tunable, with hard coded parameters such as the number of threads.
Therefore, many auto-tuning studies only focus tuning on a single application. Studies that do consider multiple applications only tune clock frequencies, or thread block dimensions, because these can be parameterized without significant code modifications. We will create a highly-tunable, representative benchmark suite to be a catalyst for auto-tuning research in general, in addition to being a much more relevant benchmark suite for evaluating the performance of new hardware architectures.

## How to run Rodinia

In order to run the original Rodinia Benchmark suite do the following :   
	cd tunedia  
	make TARGET (TARGET = CUDA/OPENCL/OMP)  
	cd scripts  
	./run_TARGET.sh (TARGET = cuda/opencl/cpu)  

## Tuning

In order to tune the kernels do the following :   
	  cd tuning  
	  cd scripts  
	  ./run_tuner.sh  
 
 The run_tuner.sh script will create for each test case :     
 	1) A header file with device targets for compiling a kernel with different parameters on different devices (see https://benvanwerkhoven.github.io/kernel_tuner/user-api.html#kernel_tuner.create_device_targets)  
	2) A .json file that stores the top (3% by default) best kernel configurations (see https://benvanwerkhoven.github.io/kernel_tuner/user-api.html#kernel_tuner.store_results)  

## Verification

In order to verify the correctness of the optimized kernels do the following :  
	1. cd tests  
	2. cd TEST_CASE (backprop,bfs etc..)  
	3. python3 TEST_CASE.py  

The python script runs the original version of the kernel followed by the optimized one and compares the two outputs whether they are equal.

## End Goal

Several bugs may exist and not all Rodinia's test cases are tuned up to now. Our end goal is to optimize and tune each test case, so that the end user can use TuneDinia in order to :   
	1) Evaluate the performance of new hardware architectures  
	2) Assess the effect of different code optimizations and other tunable parameters on GPU energy efficiency  


## Documentation

Useful link : https://benvanwerkhoven.github.io/kernel_tuner/user-api.html
			   

