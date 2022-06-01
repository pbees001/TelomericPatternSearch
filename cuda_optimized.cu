#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <time.h>
#include <vector>
#include <chrono>
#include <thread>
using namespace std;
//cuda_opt2 version - Final
#define maxPatternLength 1300000//Increase this limit if max read length of DNA sequence exceed this value in the dataset
#define maxEntries 1024
#define maxThreads 1024
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void kernel(char *d_c, int* d_match, char *d_pat, int *d_lps, int *d_numOfEntries, int *d_maxPattLength){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int maxPattlength = *d_maxPattLength;
    int n = *d_numOfEntries;
    char pat[] = "TTAGGGTTAGGGTTAGGGTTAGGG";
    if(idx < n){
        int M = 24;
        int N = 0;
        while(d_c[idx * maxPattlength + N]!='\0')
            N++;
        int pattern_found_count = 0;
		int rearEndRange = 20000;

        int i = 0, j =0;
        int lastMatchedIndex = -1;
        if(d_match[idx]!=0)
            return;
        while (i < N) {
            if (pat[j] == d_c[idx * maxPattlength + i]) {
                j++;
                i++;
            }
            if (j == M) {
                j = 0;
                pattern_found_count+=1;
                lastMatchedIndex = i;
            }
            else if (i < N && pat[j] != d_c[idx * maxPattlength + i]) {
                if (j != 0)
                    j = d_lps[j-1];
                else
                    i = i + 1;
            }
        }
        if(pattern_found_count > 1 && i-lastMatchedIndex < rearEndRange){
            d_match[idx]=1;
        }

    }
}

void computeLPSArray(char* pat, int M, int* lps)
{
    int len = 0;
    lps[0] = 0;
    int i = 1;
    while (i < M) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        }
        else
        {
            if (len != 0) {
                len = lps[len - 1];
            }
            else
            {
                lps[i] = 0;
                i++;
            }
        }
    }
}

int main( int argc, char **argv ){
    if( argc <= 2 ){
        cerr << "Usage is: "<<argv[0]<<" [infilename] [outfilename]" << std::endl;
        return -1;
    }

    ifstream input(argv[1]);
    ofstream output(argv[2]);
    if(!input.good()){
        cerr << "Error opening file "<<argv[1] << std::endl;
        return -1;
    }

    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;
    double dataLoadTime = 0.0, kernelComputionTime = 0.0;

    int N = 0, maxPattLength = maxPatternLength;
    int total_threadblocks = 0, total_threads = 0;
    string line;

    //fetching total no.of DNA pieces being processed
    while( getline( input, line ).good() ){
        if( line[0] == '>' ){
            N++;
        }
    }
    printf("Total DNA pieces in dataset : %d\n",N);

    t0 = clock();//Starting the data initialization time

    if(N > maxEntries)
        N = maxEntries;//limiting data allocation to N DNA pieces
    int j=0;
    int *match = new int[N];//flag array to segragate matched sequences
    for(int i=0;i<N;i++){
        match[i]= -1;
    }
    int pattLength = 24;
    char pattern[] = "TTAGGGTTAGGGTTAGGGTTAGGG";
    int *lps = new int[pattLength];
    computeLPSArray(pattern,pattLength,lps);//computing KMP hash map

    char (*seqs)[maxPatternLength] = new char [N][maxPatternLength];//DNA sequences array
    string names[N];//readID array
    char *d_c,*d_pat;
    int *d_match,*d_lps;
    int *d_numOfEntries, *d_maxPattLength;

    cudaMalloc((void**)&d_pat,pattLength*sizeof(char));
    cudaMalloc((void**)&d_lps,pattLength*sizeof(int));
    cudaMalloc((void**)&d_numOfEntries,sizeof(int));
    cudaMalloc((void**)&d_maxPattLength,sizeof(int));
    cudaMalloc((void**)&d_match,N * sizeof(int));//device memory allocation for flag array
    cudaMallocHost((int**)&match,N * sizeof(int));//pinned memory
    cudaMalloc((void**)&d_c,N*maxPatternLength*sizeof(char));//device memory allocation for DNA sequences
    cudaMallocHost((char**)&seqs,N*maxPatternLength*sizeof(char));//pinned memory
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_pat, pattern, pattLength*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lps, lps, pattLength*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numOfEntries, &N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxPattLength, &maxPattLength, sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");

    string name, content,tempString = "";
    input.clear();
    input.seekg(0);
    while( std::getline( input, line ).good() ){
        if( line.empty() || line[0] == '>' ){
            if( !name.empty() ){
                strcpy(seqs[j],content.c_str());
                names[j]=name;
                match[j]=0;
                j++;
                name.clear();

                if(j == N){//processing N DNA pieces at a time
                    cudaMemcpy(d_match, match, N*sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_c, &seqs[0][0], N*maxPatternLength*sizeof(char), cudaMemcpyHostToDevice);
                    cudaCheckErrors("cudaMemcpy H2D failure");

                    t1 = clock();//starting kernel computation time
                    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
                    dataLoadTime+=t1sum;

                    //Launching kernel for N DNA sequences KMP patten search processing
                    kernel <<< (N+maxThreads-1)/maxThreads,maxThreads >>> (d_c, d_match, d_pat, d_lps, d_numOfEntries, d_maxPattLength);
                    cudaCheckErrors("kernel launch failure");
                    total_threads+= maxThreads;//calculating total threads used
                    total_threadblocks+= (N+maxThreads-1)/maxThreads;//calculating total thread blocks used


                    t2 = clock();//calculating kernel computation end time
                    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
                    kernelComputionTime+=t2sum;

                    cudaMemcpy(match, d_match, N*sizeof(int), cudaMemcpyDeviceToHost);//write back device memory to host memory, flag array to find matched DNA sequences
                    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

                    //writing output to file
                    for(int k=0;k<N;k++){
                        if(match[k]== 1){
                            output<<'>'<<names[k]<<'\n';
                            output<<seqs[k]<<'\n';
                        }
                        match[k]= -1;
                        names[k].clear();
                        strcpy(seqs[k],tempString.c_str());
                    }
                    j = 0;
                    t0 = clock();
                }
            }
            if( !line.empty() ){
                name = line.substr(1);
            }
            content.clear();
        } else if( !name.empty() ){
            if( line.find(' ') != std::string::npos ){
                name.clear();
                content.clear();
            } else {
                content += line;
            }
        }
    }
    if( !name.empty() ){
        strcpy(seqs[j],content.c_str());
        names[j] = name;
        match[j] = 0;
        j++;
    }

    cudaMemcpy(d_match, match, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &seqs[0][0], N*maxPatternLength*sizeof(char), cudaMemcpyHostToDevice);//copying DNA seqs from host to device
    cudaCheckErrors("cudaMemcpy H2D failure");

    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    dataLoadTime+=t1sum;//calculating data initialization time

    //final kernel launch
    kernel <<< (N+maxThreads-1)/maxThreads,maxThreads >>> (d_c, d_match, d_pat, d_lps, d_numOfEntries, d_maxPattLength);
    cudaCheckErrors("kernel execution failure");
    total_threads+= maxThreads;
    total_threadblocks+= (N+maxThreads-1)/maxThreads;

    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    kernelComputionTime+=t2sum;
    cudaMemcpy(match, d_match, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    for(int i=0;i<N;i++){
        if(match[i]==1){
            output<<'>'<<names[i]<<'\n';
            output<<seqs[i]<<'\n';
        }
    }

    printf("Overall Data Initialization time: %f seconds\n",dataLoadTime);
    printf("Overall Kernel computation time: %f seconds\n",kernelComputionTime);
    printf("Total Threads used: %d\n", total_threads);
    printf("Total Thread-Blocks used: %d\n", total_threadblocks);
    input.close();
    output.close();

    cudaFree(d_c);
    cudaFree(d_pat);
    cudaFree(d_match);
    cudaFree(d_numOfEntries);
    cudaCheckErrors("Free memory failure");

    match = NULL;
    delete[] match;

    return 0;
}
