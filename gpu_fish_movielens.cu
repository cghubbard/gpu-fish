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

#define RANK 30
#define BLOCKS 40
std::string training_data = "";
std::string testing_data = "";
//========== define derivative and regulizer for gradient updates ============//
__global__ 
void GradientUpdate(float* L, float* R, const int rank, float* dev_ratings, int* dev_offsets, int* dev_chunk_size, int round, float alpha, float average) {
	
	
	__shared__ float sh_L[RANK];
	__shared__ float sh_R[RANK];
	__shared__ float ijr[3];
							
	int idx = blockIdx.x;
	int offset = dev_offsets[idx];
	int N = dev_chunk_size[idx]/3;
	if (dev_chunk_size[idx]<=0) { N=0; offset=0;}

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
		__syncthreads();
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

	
	const int blocks = BLOCKS;
	const int rank = RANK;


	std::string trainingFile = training_data;

	std::cout << "==========================" << std::endl;

	std::vector<std::tuple<float,float,float>> host_all_ratings;
	dataSet* data = readFile(trainingFile, &host_all_ratings);

	const int rows = data->rows;
	const int columns = data->columns;
	float* cpu_nums_R = new float[columns*rank];
	float* cpu_nums_L = new float[rows*rank];
	
	gpu_fish(&host_all_ratings, cpu_nums_L, cpu_nums_R, rows, columns, rank, blocks, data->numRatings, data->average);


	//========== testing ============// 
	std::fstream infile;

	infile.open(testing_data);
	if (!infile.is_open()) {
 		throw std::runtime_error("File not found!");
        }

	std::array<float,5> num_ratings;
	num_ratings.fill(0);

	std::array<float,5> num_correct;
	num_correct.fill(0);

	int idx = 0;
	int user;
	int movie;
	float rating;
	int total = 0;
	int total_correct = 0;
	double m_hat = 0;
	float c = data->average;
	std::string line;

	while (getline(infile,line,'\n')) {
		
		m_hat = 0;
		std::stringstream stream(line);
		
		stream >> user >> movie >> rating;
			
		for (int k=0;k<rank;k++) {
			m_hat += cpu_nums_L[(user-1)*rank + k]*cpu_nums_R[(movie-1)*rank + k];
		}
	
		idx = rating -1;
		num_ratings[idx]++;
		total++;
		
		m_hat = 1/(1+exp(-1*m_hat))-.5;

		if ((m_hat>=0) && (rating>=c)) { 
			num_correct[idx]++;			
			total_correct++;
		}

		if ((m_hat<0) && (rating<c))   { 
			num_correct[idx]++;			
			total_correct++;
		}

	
		
	}
	

	infile.close();
	
	delete[] cpu_nums_R;
	delete[] cpu_nums_L;


	for (int i=0; i<5; i++) {
		std::cout << "==================" << std::endl;
		std::cout << num_ratings[i] << std::endl;
		std::cout << ((float)num_correct[i])/num_ratings[i] << std::endl;
	}

	std::cout << "==================" << std::endl;
	std::cout << (float) total_correct/total <<std::endl;
	std::cout << "==========================" << std::endl;
	return 0;


}
