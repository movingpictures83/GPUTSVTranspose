#include <emmintrin.h>
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "GPUTSVTransposePlugin.h"

void GPUTSVTransposePlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 N = atoi(parameters["N"].c_str());
 A = (double*) malloc(N*N*sizeof(double));
 C = (double*) malloc(N*N*sizeof(double));
 int M = N * N;
 std::ifstream myinput((std::string(PluginManager::prefix())+parameters["matrix"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < M; ++i) {
	int k;
	myinput >> k;
        A[i] = k;
 }
}




void GPUTSVTransposePlugin::run() {
	double *pA;
	double *pC;
cudaMalloc((void**)&pA, (N*N)*sizeof(double));
cudaMalloc((void**)&pC, (N*N)*sizeof(double));
cudaMemcpy(pA, A, (N*N)*sizeof(double), cudaMemcpyHostToDevice);
printf("***Transpose on %d x %d Matrix on GPU***\n",N,N);
MatTrans<<<N,N>>>(pA, pC, N);
cudaMemcpy(C, pC, (N*N)*sizeof(double), cudaMemcpyDeviceToHost);

cudaFree(pA);
cudaFree(pC);

}

void GPUTSVTransposePlugin::output(std::string file) {
	std::ofstream outfile(file.c_str(), std::ios::out);
        int i, j;
        for (i = 0; i < N; ++i){
            for (j = 0; j < N; ++j){
		outfile << C[i*N+j];//std::setprecision(0) << a[i*N+j];
		if (j != N-1)
			outfile << "\t";
		else
			outfile << "\n";
            }
	}
	free(A);
	free(C);
}



PluginProxy<GPUTSVTransposePlugin> GPUTSVTransposePluginProxy = PluginProxy<GPUTSVTransposePlugin>("GPUTSVTranspose", PluginManager::getInstance());


