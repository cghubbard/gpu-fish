/*
The MIT License (MIT)

Copyright (c) 2016 Charles Hubbard and Chinmay Hegde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <random>
#include <algorithm>
#include "gpu_fish_headers.h"
#include <thread>
#include <stdexcept>
#include <chrono>


//========== Error handling macro ============//
//  HandleError is provided by the NVIDIA Corporation and is available in //
//  Cuda by Example Copyright (c) 2011, NVIDIA Corporation //

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

//========== initialize random states for L and R ============//
__global__
void init(unsigned int seed, curandState_t* states, int maxthread) {

	int tidx = blockIdx.x*blockDim.x + threadIdx.x;

	if (tidx < maxthread) {
		curand_init((seed << 20) + tidx, tidx, 0, &states[tidx]);
	}

}

//========== fill L and R with random numbers ============//
__global__
void randoms(curandState_t* states, float* numbers, float scaling, int maxthread) {

	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	if (tidx < maxthread) {
		numbers[tidx] = (curand_uniform(&states[threadIdx.x])-0.5)/scaling;
	}

}


//========== read training data file ============//
dataSet* readFile(std::string file, std::vector<std::tuple<float,float,float>>* host_all_ratings) {
	float user,movie,rating;

	std::fstream infile;
	std::string line;
	infile.open(file);

	if (!infile.is_open()) {
		throw std::runtime_error("File not found!");
	}

	int count = 0;
	double average = 0;
	int maxRow = 0;
	int maxCol = 0;

	dataSet* data = new dataSet();

	getline(infile,line,'\n');
	std::stringstream stream(line);
	stream >> data->rows >> data->columns;

	while (getline(infile,line,'\n')) {


		std::stringstream stream(line);

		stream >> user >> movie >> rating;

		//========== subtract 1 to 0-index data ============//
		user = user - 1;
		movie = movie - 1;

		maxRow = max(maxRow,(int)user);
		maxCol = max(maxCol,(int)movie);

		auto temp = std::make_tuple(user,movie,rating);

		host_all_ratings->push_back(temp);
		average += rating;
		count++;

	}
	data->average = average/count;
	data->numRatings = count;
	infile.close();

	//  No need to modify data if users or items are indexed with gaps i.e. [0,3,....n] //
	//  If memory is a concern data needs to be re-index without gaps // 
	if (maxRow > data->rows) { data->rows = maxRow; }
	if (maxCol > data->columns) { data->columns = maxCol; }

	return data;

}

//========== chunk training data set ============//
void chunking(int* piRows, int* piCols, std::vector<std::tuple<float,float,float>>* host_all_ratings,
			std::vector<float>* host_ratings_chunked, int blocks, int rows, int columns, int omega) {



		float user,movie,rating;
		int a_index,b_index;
		float a,b;
		int half = host_all_ratings->size()/2;
		std::tuple<float,float,float> temp;

		unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count();
		auto randEng = std::default_random_engine(randSeed);

		std::shuffle (host_all_ratings->begin(),host_all_ratings->end(),randEng);

		std::shuffle (&piRows[0], &piRows[rows-1],randEng);
		std::shuffle (&piCols[0], &piCols[columns-1],randEng);


		for(std::vector<std::tuple<float,float,float>>::reverse_iterator it = host_all_ratings->rbegin();
		it != host_all_ratings->rend(); it++) {


			temp = *it;
			user = std::get<0>(temp);
			movie = std::get<1>(temp);
			rating = std::get<2>(temp);

			a = (((float) blocks)/rows)*(piRows[(int)user]) + 1;
			b = (((float) blocks)/columns)*(piCols[(int)movie]) + 1;

			a_index = (int) a-1;
			b_index = (int) b-1;

			host_ratings_chunked[blocks*a_index + b_index].push_back(user);
                	host_ratings_chunked[blocks*a_index + b_index].push_back(movie);
                	host_ratings_chunked[blocks*a_index + b_index].push_back(rating);

		}


}

//========== setup and execution of GPUFish ============//
void gpu_fish(std::vector<std::tuple<float,float,float>>* host_all_ratings, float *L, float *R, const int rows, const int columns, const int rank, const int blocks, int omega, double average) {



	std::vector<float>* host_ratings_chunked = new std::vector<float>[blocks*blocks];
	std::vector<float>* odd_host_ratings_chunked = new std::vector<float>[blocks*blocks];

	int* piRows = new int[rows];
	int* piCols = new int[columns];


	for(int i=0; i<rows; i++) { piRows[i] = i; }
	for(int i=0; i<columns; i++) { piCols[i] = i; }

	//========== Launch initial chunking on separate thread ============//
	std::thread initial_chunk (chunking, piRows, piCols, std::ref(host_all_ratings), host_ratings_chunked, blocks, rows, 
					columns,omega);

	//========== set up L and R ============//
	int maxDim = std::max(rows,columns);

	curandState_t* states;

	HANDLE_ERROR ( cudaMalloc ((void**) &states, (maxDim*rank*sizeof(curandState_t))));
	HANDLE_ERROR(cudaPeekAtLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

    	int tpblock = 128;
   	int randInitBlocks = (maxDim*rank+tpblock-1)/tpblock;
  	int maxthread = maxDim*rank;

   	init<<<randInitBlocks, tpblock>>>(time(0), states, maxthread);

    	HANDLE_ERROR(cudaPeekAtLastError());
    	HANDLE_ERROR(cudaDeviceSynchronize());

    	//========== fill L and R ============//
    	float* dev_L;
    	float* dev_R;

    	float scaling = sqrt(rows*columns);

	HANDLE_ERROR( cudaMalloc((void**) &dev_L, rows *rank* sizeof(float)));
	HANDLE_ERROR( cudaMalloc((void**) &dev_R, columns *rank* sizeof(float)));

	tpblock = 128;
    	randInitBlocks = (rows*rank+tpblock-1)/tpblock;
    	maxthread = rows*rank;

	randoms<<<randInitBlocks, tpblock>>>(states, dev_L, scaling, maxthread);

	HANDLE_ERROR(cudaPeekAtLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	tpblock = 128;
    	randInitBlocks = (columns*rank+tpblock-1)/tpblock;
    	maxthread = columns*rank;

	randoms<<<randInitBlocks, tpblock>>>(states, dev_R, scaling, maxthread);

	HANDLE_ERROR( cudaFree(states));

	//========== separate chunks into rounds ============//
	int rounds_chunks[blocks][blocks];

	int rounds = 0;
	int chunk = 0;
	int num = 0;

	for (int i=0;i<blocks*blocks;i++) {

		if (i%blocks ==0  && i != 0) { rounds++; num = 0; }
		rounds_chunks[rounds][num] = chunk;
		chunk += blocks+1;
		if (chunk > blocks*blocks) { chunk = chunk - blocks*blocks;}
		num++;
	}

	//========== set up variables needed in epochs ============//
	bool use_even = true;
	float* pinned_mem;
	float* dev_ratings;
	int offsets[blocks];
	int* dev_offsets;
	int chunk_size[blocks];
	int* dev_chunk_size;

	std::vector<float>* ratings_to_chunk;
	std::vector<float>* ratings_to_read;

	cudaStream_t s_mem, s_kernel;
	cudaStreamCreate(&s_mem); cudaStreamCreate(&s_kernel);

	initial_chunk.join();

	int epoch = 0;
	int numEpoch = 20;
	double alpha = 0.8;
 	std::thread thread;

	while (epoch < numEpoch) {

		//std::cout << epoch << std::endl;

		//========== choose which data to permute and which to train with ============//
		if (use_even) {

			ratings_to_chunk = odd_host_ratings_chunked;
			ratings_to_read = host_ratings_chunked;
		}
		else {

			ratings_to_chunk = host_ratings_chunked;
			ratings_to_read = odd_host_ratings_chunked;
		}

		//========== launch chunking for next epoch ============//
		if (epoch != numEpoch-1) {				
			thread = std::thread(chunking, piRows, piCols, std::ref(host_all_ratings), ratings_to_chunk, blocks, rows, 					columns,omega);
		}

		//========== get largest pinned mem size that will be needed ============//
		int idx = 0;
		int maxSize = 0;
		int roundSize;
		for (int k=0; k<blocks; k++) {
			roundSize=0;
			for (int l=0; l<blocks; l++) {
				idx = rounds_chunks[k][l];
				roundSize += ratings_to_read[idx].size();
			}
			maxSize = max(maxSize,roundSize);
		}

		size_t bytes = sizeof(float)*maxSize;
		if (bytes == 0) { std::cout << "error, empty round" <<std::endl;}
		HANDLE_ERROR( cudaMallocHost((void**)&pinned_mem,bytes) );


		//========== concatenate all chunks for round one data transfer to GPU ============//
		roundSize = 0;
		for (int l=0; l<blocks; l++) {

			idx = rounds_chunks[0][l];
	
			std::copy(ratings_to_read[idx].begin(),ratings_to_read[idx].end(),pinned_mem+roundSize);
		
			offsets[l] = roundSize;

			roundSize += ratings_to_read[idx].size();
			chunk_size[l] = ratings_to_read[idx].size();

		}

		//========== tranfer data for round 1 to GPU ============//
		HANDLE_ERROR( cudaMalloc((void**) &dev_ratings, bytes) );
		HANDLE_ERROR( cudaMalloc((void**) &dev_offsets, blocks*sizeof(int)) );
		HANDLE_ERROR( cudaMalloc((void**) &dev_chunk_size, blocks*sizeof(int)) );

		HANDLE_ERROR( cudaMemcpyAsync(dev_ratings, pinned_mem, bytes, cudaMemcpyHostToDevice,s_mem) );

		HANDLE_ERROR( cudaMemcpy(dev_offsets, offsets, blocks*sizeof(int), cudaMemcpyHostToDevice) );
		HANDLE_ERROR( cudaMemcpy(dev_chunk_size, chunk_size, blocks*sizeof(int), cudaMemcpyHostToDevice) );
		HANDLE_ERROR( cudaStreamSynchronize(s_mem));


		bool odd = false;
		float* odd_dev_ratings;
		int* odd_dev_offsets;
		int* odd_dev_chunk_size;

		float* dev_ratings_read; float* dev_ratings_write;
		int* dev_offsets_read; int* dev_offsets_write;
		int* dev_chunk_size_read; int* dev_chunk_size_write;

		HANDLE_ERROR( cudaMalloc((void**) &odd_dev_ratings, bytes) );
		HANDLE_ERROR( cudaMalloc((void**) &odd_dev_offsets, blocks*sizeof(int)) );
		HANDLE_ERROR( cudaMalloc((void**) &odd_dev_chunk_size, blocks*sizeof(int)) );

	    	for(int i=0; i < blocks; i++) {

			if (odd) {
				dev_ratings_read = odd_dev_ratings;
				dev_offsets_read = odd_dev_offsets;
				dev_chunk_size_read = odd_dev_chunk_size;

				dev_ratings_write = dev_ratings;
				dev_offsets_write = dev_offsets;
				dev_chunk_size_write = dev_chunk_size;
			}
			else {
				dev_ratings_read = dev_ratings;
				dev_offsets_read = dev_offsets;
				dev_chunk_size_read = dev_chunk_size;

				dev_ratings_write = odd_dev_ratings;
				dev_offsets_write = odd_dev_offsets;
				dev_chunk_size_write = odd_dev_chunk_size;
			}



			//========== launch graident updates ============//
			GradientUpdate<<<blocks,rank,0,s_kernel>>>
				(dev_L, dev_R, rank,dev_ratings_read,dev_offsets_read,dev_chunk_size_read,i,(float) alpha,
					(float) average);

			HANDLE_ERROR( cudaDeviceSynchronize() );
			//========== in parallel, begin data tranfer for next chunk ============//
			if (i+1 < blocks) {

				roundSize = 0;
				for (int l=0; l<blocks; l++) {

					idx = rounds_chunks[i+1][l];
					std::copy(ratings_to_read[idx].begin(),ratings_to_read[idx].end(),
							pinned_mem+roundSize);
					
					offsets[l] = roundSize;

					roundSize += ratings_to_read[idx].size();

					chunk_size[l] = ratings_to_read[idx].size();

				}

				HANDLE_ERROR( cudaMemcpyAsync(dev_ratings_write, pinned_mem, bytes,
						cudaMemcpyHostToDevice,s_mem));
				HANDLE_ERROR( cudaMemcpyAsync(dev_offsets_write, offsets, blocks*sizeof(int),
						cudaMemcpyHostToDevice,s_mem));
				HANDLE_ERROR( cudaMemcpyAsync(dev_chunk_size_write, chunk_size, blocks*sizeof(int),
						cudaMemcpyHostToDevice,s_mem));


				HANDLE_ERROR( cudaDeviceSynchronize() );
				odd = !odd;

			}
			else {
				
				HANDLE_ERROR( cudaFreeHost(pinned_mem) );
				HANDLE_ERROR( cudaFree(dev_ratings) ); HANDLE_ERROR( cudaFree(odd_dev_ratings) );
				HANDLE_ERROR( cudaFree(dev_offsets) ); HANDLE_ERROR( cudaFree(odd_dev_offsets) );
				HANDLE_ERROR( cudaFree(dev_chunk_size) ); HANDLE_ERROR( cudaFree(odd_dev_chunk_size) );
			}



	   	}

		//========== clear chunks used for training ============//
		for (int k=0; k<blocks*blocks; k++) { ratings_to_read[k].clear(); }

		alpha = pow(alpha,1.2);
		//alpha *= 0.83;
		if (epoch != numEpoch-1){ thread.join(); }
		epoch++;
		use_even = !use_even; 
		
   }


	//========== copy L and R to host ============//
	HANDLE_ERROR( cudaMemcpy(L, dev_L, rows*rank* sizeof(float), cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(R, dev_R, columns*rank* sizeof(float), cudaMemcpyDeviceToHost) );

	//========== free memory ============//
	delete[] piRows;
	delete[] piCols;
	HANDLE_ERROR( cudaStreamDestroy(s_mem) );
	HANDLE_ERROR( cudaStreamDestroy(s_kernel) );
	HANDLE_ERROR( cudaFree(dev_L) );
	HANDLE_ERROR( cudaFree(dev_R) );

}
