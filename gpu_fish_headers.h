/*
The MIT License (MIT)

Copyright (c) 2016 Charles Hubbard and Chinmay Hegde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef GPU_FISH_HEADERS_H_
#define GPU_FISH_HEADERS_H_

#include <curand.h>
#include <curand_kernel.h>

__global__
void init(unsigned int, curandState_t*, int);

__global__
void randoms(curandState_t*, float*, float, int);

struct dataSet {
	
	int rows;
	int columns;
	int numRatings;
	double average;
};

dataSet* readFile(std::string, std::vector<std::tuple<float,float,float>>*);

void chunking(int*, int*, std::vector<std::tuple<float,float,float>>*, std::vector<float>*, int, int, int, int);

void gpu_fish(std::vector<std::tuple<float,float,float>>*, float*, float*, const int, const int, const int, const int, int, double);

__global__ 
void GradientUpdate(float*, float*, const int, float*, int*, int*, int, float, float);


#endif
