## TuneDinia

To study the effect of different code optimizations and other tunable parameters on GPU performance it is crucial to have a representative set of tunable applications. Existing benchmark suites, e.g. PARSEC and Rodinia, are straightforward implementations that are not tunable, with hard coded parameters such as the number of threads.

Therefore, many auto-tuning studies only focus tuning on a single application. Studies that do consider multiple applications only tune clock frequencies, or thread block dimensions, because these can be parameterized without significant code modifications. We will create a highly-tunable, representative benchmark suite to be a catalyst for auto-tuning research in general, in addition to being a much more relevant benchmark suite for evaluating the performance of new hardware architectures.

In TuneDinia, we optimized and inserted tunable parameters on the test cases of the Rodinia Benchmark suite. For the purpose of auto-tuning the Kernel Tuner[1] was used. In the first iteration of this project we selected the test cases present on Rodinia's original publication  ( https://www.cs.virginia.edu/~skadron/Papers/rodinia_iiswc09.pdf ).  In the next steps of this project more test cases will be included and more tunable parameters will be inserted on each of them. 

The existing version includes the following test cases : 

	 Backprop
	 Hotspot
	 BFS
	 Leukocyte Tracking
	 Streamcluster
	 Kmeans
	 Needleman Wunsch
	 Gaussian
	 SRAD

## Tuning

In order to tune the kernels do the following :   
 - cd tuning  
 - cd scripts 
 - ./run_tuner.sh
 
 The run_tuner.sh script will create for each test case :     
 	1) A header file with device targets for compiling a kernel with different parameters on different devices (see https://benvanwerkhoven.github.io/kernel_tuner/user-api.html#kernel_tuner.create_device_targets)  
	2) A .json file that stores the top (3% by default) best kernel configurations (see https://benvanwerkhoven.github.io/kernel_tuner/user-api.html#kernel_tuner.store_results)  
	
Note : the scripts are set to run on the DAS-5 supercomputer ( https://www.cs.vu.nl/das5/ ). Minor modifications need to be applied when these tests need to run on a different setup.

## End Goal

Several bugs may exist and not all Rodinia's test cases are tuned up to now. Our end goal is to optimize and tune each test case, so that the end user can use TuneDinia in order to :   
 - Evaluate the performance of new hardware architectures  
 - Assess the effect of different code optimizations and other tunable parameters on GPU energy efficiency and performance for a variety of applications belonging to different domains such as Data Mining, Graph Algorithms, Medical Imaging and others.


## Documentation

Useful link : https://benvanwerkhoven.github.io/kernel_tuner/user-api.html
			   
## Citation

<a id="1">[1]</a> 
article = kerneltuner \
author = Ben van Werkhoven \
title   = Kernel Tuner: A search-optimizing GPU code auto-tuner \
  journal = Future Generation Computer Systems \
  year = 2019 \
  volume  = 90 \
  pages = 347-358 \
  url = https://www.sciencedirect.com/science/article/pii/S0167739X18313359 \
  doi = https://doi.org/10.1016/j.future.2018.08.004

<a id="2">[2]</a>
article = willemsen2021bayesian \
  author = Willemsen, Floris-Jan and Van Nieuwpoort, Rob and Van Werkhoven, Ben \
  title = Bayesian Optimization for auto-tuning GPU kernels \
  journal = International Workshop on Performance Modeling, Benchmarking and Simulation
     of High Performance Computer Systems (PMBS) at Supercomputing (SC21) \
  year = 2021 \
  url = https://arxiv.org/abs/2111.14991
}

