#include <fstream>
#include <chrono>
#include <CL/sycl.hpp>
#include <iostream>
#include <cstring>
#include <vector>


using namespace sycl;
using namespace std;
static const int maxEntries = 10000;
static const int maxPatternLength = 883000;//Increase this limit if max read length of DNA sequence exceed this value in the dataset

void computeLPSArray(char* pat, int M, int* lps)
{
    int len = 0;
    lps[0] = 0; // lps[0] is always 0
    int i = 1;
    while (i < M) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        }
        else // (pat[i] != pat[len])
        {
            if (len != 0) {
                len = lps[len - 1];
            }
            else // if (len == 0)
            {
                lps[i] = 0;
                i++;
            }
        }
    }
}

int main() {
    gpu_selector selector;//selecting GPU
    queue q(selector);
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    chrono::time_point<chrono::system_clock> start, end;

//#    ifstream input("/home/u126885/project/subset_na12878dataset_15GB.fasta");
    ifstream input("lab/subset_na12878dataset.fasta");//input file
    ofstream output("lab/final_sampleoutput.fasta");//output file
//#    ofstream output("lab/demo_output_15GB.fasta");
    if(!input.good()){
        std::cerr << "Error opening Input file" << std::endl;
        return -1;
    }

    string line;
    int N = 0;
    //counting DNA pieces in the Dataset
    while( getline( input, line ).good() ){
        if( line[0] == '>' ){ // Identifier marker
            N++;
        }
    }

    printf("Num of DNA pieces in Dataset : %d\n",N);
    start = chrono::system_clock::now();

    if(N > maxEntries)
        N = maxEntries;//limiting memory allocation to only N DNA pieces

    int j=0;
    char (*contents)[maxPatternLength] = new char [N][maxPatternLength];//DNA sequences
    int *matches = static_cast<int *>(malloc(N * sizeof(int)));//flag array to seperate matched DNA sequences
    for (int i = 0; i < N; i++) matches[i] = 0;
      //# Explicit USM allocation using malloc_device
    int *matches_device = malloc_device<int>(N, q);
    char* contents_device = malloc_device<char>(N*sizeof(char)*maxPatternLength,q);

    printf("Data allocated Successfully\n");

    int pattLength = 24;
    int *lps = new int[pattLength];
    char pattern[] = "TTAGGGTTAGGGTTAGGGTTAGGG";
    computeLPSArray(pattern,pattLength,lps);

    char *pat_device = malloc_device<char>(pattLength, q);
    int *lps_device = malloc_device<int>(pattLength, q);

    q.memcpy(pat_device, pattern, sizeof(char) * pattLength).wait();
    q.memcpy(lps_device, lps, sizeof(int) * pattLength).wait();

    string names[N];
    std::string name, content, tempString = "";
    input.clear();
    input.seekg(0);
    while( std::getline( input, line ).good() ){
        if( line.empty() || line[0] == '>' ){ // Identifier marker
            if( !name.empty() ){ // Print out what we read from the last entry
                names[j] = name;
                matches[j] = 0;
                strcpy(contents[j++],content.c_str());
                name.clear();
                if(j == N){//processing every N DNA pieces
                    q.memcpy(matches_device, matches, sizeof(int) * N).wait();
                    q.memcpy(contents_device, &contents[0][0], sizeof(char) * N * maxPatternLength).wait();//copying DNA pieces from host to device

                    //Kernel processing - KMP algorithm
                    q.parallel_for(range<1>(N), [=](id<1> index) {
                        int M = 24;
                        int nn = 0;
                        while(contents_device[index * maxPatternLength + nn]!='\0'){
                            nn++;
                        }
                        int pattern_found_count = 0;
                        int rearEndRange = 20000;
                        bool nearReadEnd = false;
                        int i = 0;
                        int j = 0;
                        if(matches_device[index] == 0){
                            while (i < nn) {
                                if (pat_device[j] == contents_device[index * maxPatternLength + i]) {
                                        j++;
                                        i++;
                                }
                                if (j == M) {
                                        j = 0;
                                        pattern_found_count+=1;
                                        if(nn - i < rearEndRange){
                                                nearReadEnd = true;
                                        }
                                }

                                else if (i < nn && pat_device[j] != contents_device[index * maxPatternLength + i]) {
                                        if (j != 0)
                                                j = lps_device[j-1];
                                        else
                                                i = i + 1;
                                }
                            }
                            if(pattern_found_count > 1 && nearReadEnd){
                                    matches_device[index] = 1;
                            }
                        }
                    }).wait();

                    //# copy mem from device to host
                    q.memcpy(matches, matches_device, sizeof(int) * N).wait();

                    for(int k=0;k<N;k++){
                        if(matches[k]==1){//writing output to file
                            output<<'>'<<names[k]<<'\n';
                            output<<contents[k]<<'\n';
                        }
                        matches[k]= -1;
                        names[k].clear();
                        strcpy(contents[k], tempString.c_str());
                    }
                    j = 0; //reset time
                }
            }
            if( !line.empty() ){
                name = line.substr(1);
            }
            content.clear();
        } else if( !name.empty() ){
            if( line.find(' ') != std::string::npos ){ // Invalid sequence--no spaces allowed
                name.clear();
                content.clear();
            } else {
                content += line;
            }
        }
    }
    if( !name.empty() ){ // Print out what we read from the last entry
        names[j] = name;
        matches[j] = 0;
        strcpy(contents[j++],content.c_str());
    }

    q.memcpy(matches_device, matches, sizeof(int) * N).wait();
    q.memcpy(contents_device, &contents[0][0], sizeof(char) * N * maxPatternLength).wait();

    //final kernel launch
    q.parallel_for(range<1>(N), [=](id<1> index) {
        int M = 24;
        int nn = 0;
        while(contents_device[index * maxPatternLength + nn]!='\0'){
            nn++;
        }
        char pat[] = "TTAGGGTTAGGGTTAGGGTTAGGG";
        int pattern_found_count = 0;
        int rearEndRange = 20000;
        bool nearReadEnd = false;
        int i = 0;
        int j = 0;
        if(matches_device[index] == 0){
            while (i < nn) {
                if (pat[j] == contents_device[index * maxPatternLength + i]) {
                        j++;
                        i++;
                }
                if (j == M) {
                        j = 0;
                        pattern_found_count+=1;
                        if(nn - i < rearEndRange){
                                nearReadEnd = true;
                        }
                }

                else if (i < nn && pat[j] != contents_device[index * maxPatternLength + i]) {
                        if (j != 0)
                                j = 0;
                        else
                                i = i + 1;
                }
            }
            if(pattern_found_count > 1 && nearReadEnd){
                    matches_device[index] = 1;
            }
        }
    }).wait();

    //# copy mem from device to host
    q.memcpy(matches, matches_device, sizeof(int) * N).wait();

    for(int k=0;k<N;k++){
        if(matches[k]==1){
            output<<'>'<<names[k]<<'\n';
            output<<contents[k]<<'\n';
        }
    }

    input.close();
    output.close();

    free(matches_device, q);
    free(matches);
    free(lps_device, q);
    free(lps);
    free(contents_device, q);
    free(contents);
    free(pat_device, q);

    end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    cout << "Elapsed CPU time: " << elapsed_seconds.count() << "s \n";

    return 0;
}
