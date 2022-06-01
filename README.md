# TelomericPatternSearch
Telomeric Pattern Searching in DNA sequences using parallel computing on Nvidia &amp; Intel GPU's

This project is done for course "High Performance Computing" under Dr. Zubair at ODU. The idea was to find telomeric patterns in given dataset of DNA sequences with large lengths (1300000 characters length considered). The datasets on which these codes are tested are 15GB, 25GB and 375GB. Each DNA sequence is processed on a different thread and 2 GB is being processed everytime in memory. 

Both CUDA and oneAPI codes are optimized codes and can run on any large size data.

Note:
If maximum read length of a DNA sequence is greater than 1300000 in a dataset, then need to update the new length in the codes.



How to run CUDA code:

Compile command:
nvcc -o cudaopt cuda_optimized.cu -arch=sm_70
Run command:
./cudaopt input_filename.fasta output_filename.fasta


How to run oneAPI code:

Compile command:
source /opt/intel/inteloneapi/setvars.sh
dpcpp oneAPI_optimized.cpp -o oneAPI_optimized

Run command:
./oneAPI_optimized
