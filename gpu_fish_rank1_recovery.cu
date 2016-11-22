/*
The MIT License (MIT)

Copyright (c) 2016 Charles Hubbard and Chinmay Hegde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


/* Experiment to characterize the ability of GPUFish to recover a rank-1 matrix */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <math.h>
#include <stdexcept>
#include "gpu_fish_headers.h"

#define RANK 1
#define BLOCKS 20

//========== define derivative and regulizer for gradient updates ============//
__global__ 
void GradientUpdate(float* L, float* R, const int rank, float* dev_ratings, int* dev_offsets, int* dev_chunk_size, int round, float alpha, float average) {
	
	
	__shared__ float sh_L[RANK];
	__shared__ float sh_R[RANK];
	__shared__ float ijr[3];
							
	int idx = blockIdx.x;
	int offset = dev_offsets[idx];
	int N = dev_chunk_size[idx]/3;
	
	float B = 1;
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

	//========== define rank and number of cores to use ============//
	const int blocks = BLOCKS;
	const int rank = RANK;


	std::ofstream results;
	results.open("data/results.txt");

	//========== loop through 30 training sets ============//

	for (int j=0; j<30; j++){
		std::cout << "================================" << std::endl;

		std::vector<std::tuple<float,float,float>> host_all_ratings;

		std::ostringstream fstring;
		fstring << "data/" <<"train" << j+1 << "r" << 1;
		std::string fileString = fstring.str();

		dataSet* data = readFile(fileString, &host_all_ratings);


		const int rows = data->rows;
		const int columns = data->columns;
		float* cpu_nums_R = new float[columns*rank];
		float* cpu_nums_L = new float[rows*rank];

		//========== loop through 9 test sets ============//
		for (int i=0; i<9;i++) {

			gpu_fish(&host_all_ratings, cpu_nums_L, cpu_nums_R, rows, columns, rank, blocks, 
					data->numRatings, data->average);

			std::ostringstream tString;
			tString << "data/" <<"test" << i+1 << "r" << 1;
			std::string testString = tString.str();
	
			std::fstream infile;
			infile.open(testString);
	
			if (!infile.is_open()) {
 				std::cout << "file not found, exiting" << std::endl;
                 		return 0;
         		}

	
			int user;
			int movie;
			float rating;
		
			int count = 0;
			int correct = 0;
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

				if ((m_hat>=0) && (rating>=c)) { correct++; }
				if ((m_hat<0) && (rating<c))   { correct++; }
	
				count++;
		
			}
	
			double ratio = ((double) correct)/count;

			if (i==0) { 
				results << "Number of samples: " << data->numRatings << std::endl;
			 	std::cout << "Number of samples: " << data->numRatings << std::endl;
			}

			//========== write recovery percentage to file ============//
			results << ratio << "\t";
			infile.close();
		}

		results << std::endl;
		delete[] cpu_nums_R;
		delete[] cpu_nums_L;

	}

	results.close();
	return 0;
}
