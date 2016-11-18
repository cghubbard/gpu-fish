/*
The MIT License (MIT)

Copyright (c) 2016 Charles Hubbard and Chinmay Hegde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/* Test GPUFish with Movielens dataset */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <math.h>
#include <stdexcept>
#include "gpu_fish_headers.h"

std::string training_data = "../Data_sets-Resutls/ml_1m_train1";
std::string testing_data = "../Data_sets-Resutls/ml_1m_test1";
#define RANK 30
#define CORES 200

//========== define derivative and regulizer for gradient updates ============//
__global__
void GradientUpdate(float* L, float* R, const int rank, float* dev_ratings, int* dev_offsets, int* dev_chunk_size, int round, float alpha, float average) {


	__shared__ float sh_L[RANK];
	__shared__ float sh_R[RANK];
	__shared__ float ijr[3];

	int idx = blockIdx.x;
	int tidx = threadIdx.x;
	int offset = dev_offsets[idx];
	int N = dev_chunk_size[idx]/3;

	float B = 4.95;
	float m_hat = 0;
	int i = 0;
	int j = 0;
	float m = 0;
	float deriv;

	for (int p=0;p<N;p++) {

		for (int k=(threadIdx.x); k<3; k+= blockDim.x){
			ijr[k] = dev_ratings[offset+p*3+k];
		}

		i = (int) ijr[0];
		j = (int) ijr[1];
		m = ijr[2];

		for (int k=(threadIdx.x); k<RANK; k+= blockDim.x) {

			sh_L[k] = L[i*RANK + k];
			sh_R[k] = R[j*RANK + k];

		}

		m_hat = 0;
		__syncthreads();

		//========== get current estimate of rating ============//
		#pragma unroll
		for (int k=0;k<RANK;k++) {

			m_hat += sh_L[k]*sh_R[k];

		}

		//========== calculate derivative: one-bit ============//
		if(m>average) {
			deriv = -1/(exp(m_hat) + 1);
		}
		else {
			deriv = exp(m_hat)/(exp(m_hat)+1);
		}

		/*

		Perform movielens analysis with squared error function
		//========== calculate derivative: squared error ============//

		deriv = 2*(m_hat-(m-average));

		*/

		for (int k=(threadIdx.x); k<RANK; k+= blockDim.x) {

			sh_L[k] = sh_L[k] - alpha*deriv*sh_R[k];
			sh_R[k] = sh_R[k] - alpha*deriv*sh_L[k];

		}

		__syncthreads();

		//========== regulizer ============//
		float normL = 0;
		float normR = 0;

		#pragma unroll
		for (int k=0;k<RANK;k++) {

			normL+= sh_L[k]*sh_L[k];
			normR+= sh_R[k]*sh_R[k];
		}


		if (normL>B){
		   	for (int k=(threadIdx.x); k<RANK; k+= blockDim.x) {

				sh_L[k] = sh_L[k]*sqrt(B)/sqrt(normL);
			}
		}

		if (normR>B){

			for (int k=(threadIdx.x); k<RANK; k+= blockDim.x) {

				sh_R[k] = sh_R[k]*sqrt(B)/sqrt(normR);

			}
		}

		for (int k=(threadIdx.x); k<RANK; k+= blockDim.x) {

			L[i*RANK + k] = sh_L[k];
			R[j*RANK + k] = sh_R[k];

		}

	}

}


struct dataSet;

int main() {


	const int cores = CORES;
	const int rank = RANK;


	std::cout << "==========================" << std::endl;

	std::vector<std::tuple<float,float,float>> host_all_ratings;

	//========== read training data ============//
	dataSet* data = readFile(training_data, &host_all_ratings);


	const int rows = data->rows;
	const int columns = data->columns;
	float* cpu_nums_R = new float[columns*rank];
	float* cpu_nums_L = new float[rows*rank];

	gpu_fish(&host_all_ratings, cpu_nums_L, cpu_nums_R, rows, columns, rank, cores, data->numRatings, data->average);


	//========== testing ============//
	std::fstream infile;
	infile.open(testing_data);
	if (!infile.is_open()) {
 		throw std::runtime_error("File not found!");
        }

	std::string line;

	int user;
	int movie;
	float rating;
	float m_hat;

	while (getline(infile,line,'\n')) {

		std::stringstream stream(line);

		m_hat = 0;
		stream >> user >> movie >> rating;

		//========== get estimate from L[i,:]*R[j,:] ============//
		for (int k=0;k<rank;k++) {
			m_hat += cpu_nums_L[(user-1)*rank + k]*cpu_nums_R[(movie-1)*rank + k];
		}

		//========== error metric ============//



	}


	infile.close();

	delete[] cpu_nums_R;
	delete[] cpu_nums_L;
	return 0;


}
