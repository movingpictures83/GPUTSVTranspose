#ifndef GPUTSVTRANSPOSEPLUGIN_H
#define GPUTSVTRANSPOSEPLUGIN_H

#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUTSVTransposePlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
		double* A;
		double* C;
		int N;
                std::map<std::string, std::string> parameters;
};

__global__ void MatTrans(double* A, double* C, int N){
               int i = blockIdx.x;
               int j = threadIdx.x;
               C[j*N+i] = A[i*N+j];

}

#endif
