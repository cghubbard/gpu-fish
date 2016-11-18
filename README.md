# GPUFish
GPUFish is designed to solve very large matrix completion problems in a modular and scalable fashion on workstations equipped with a graphics processing unit (GPU).

GPUFish builds upon the algorithmic techniques of Jellyfish<sup>[1](#footnote1)</sup>, a popular parallel stochastic gradient descent algorithm for matrix completion, but has been highly optimized to run efficiently on a GPU.  GPUFish enables the user to combine various loss functions and regularizers used in the matrix completion problem, making experimentation and testing quick and easy.  For details, please refer to the technical report:

C. Hubbard, C. Hegde, "GPUFish: A Parallel Computing Framework for Matrix Completion from a Few Observations", Iowa State University Technical Report, November 2016.

## Getting Started

These instructions will get you GPUFish setup and running on your local machine for development and testing purposes.

### Prerequisites

To use this software, you will need an NVIDIA graphics card with Compute Capability 2.0 or higher, and the NVIDIA CUDA Compiler (NVCC).  The relevant installation instructions for NVCC are part of the CUDA Toolkit and can be found [here](https://developer.nvidia.com/cuda-downloads).

### Installing

To begin, clone the repository using:
```
git clone https://github.com/cghubbard/gpu-fish.git
```
Next, navigate to the top folder of your repository and open:
```
gpu_fish_template.cu
```
in your favorite text editor to get started.

As training data, GPUFish accepts a list of *observed* entries of the unknown matrix. This list should be in the form of a file containing tab-delimited <row index,column index,value> tuples with one tuple per line. You can specify the name of the file on Line 24.

The first line of the file should contain the number of rows followed by the number of columns of the matrix. An example is provided below.

```
134 345             \\ 134 users and 345 movies
1   1   5           \\ (user 1) gives (movie 1) a rating of 5.
3   10  1
...
134 345 2           \\ tuples need not be sorted
```

GPUFish also expects your row and column indices to begin with 1. If they are 0-indexed a small change to the readFile function is required (line 105 of gpu_fish_functions.cu).

After entering the location of your training file, you will need to choose a target *rank* for your output matrix (Line 26). We have not experimented with this parameter in depth, but over-estimating the rank seems to have a smaller effect on accuracy than under-estimating the rank; so when in doubt, guess something sufficiently large. (However, as always be wary of overfitting!)

Finally, choose the number of blocks you would like to use (Line 27). The right parameter choice here will depend both on your GPU and the size of your dataset.  Your data set will be split into _blocks<sup>2</sup>_ (approximately) equal chunks; having empty or near empty chunks will decrease the performance of GPUFish. [This link](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)(table 13) displays the maximum number of blocks than can be run in parallel by a single multiprocessor of the GPU. Running more blocks than (max blocks per multiprocessor)x(number of multiprocessors) may hinder performance).  

GPUFish takes advantage of pinned memory to move data quickly to the GPU.  If you run a large data set with too few blocks a large amount of pinned memory will be allocated, reducing the amount of memory available to your system.  

Once you are done, save your changes to the template as a new file and compile with the command:
```
nvcc -std=c++11 -O2 gpu_fish_functions.cu <my_file.cu>
```
Execute GPUFish with:
```
./a.out
```
### Changing the Loss Function
GPUFish tries to optimize over a loss function that is user-specified.  
The function that defines the gradient updates performed by GPUFish is provided in every file containing a _main()_ method.  The experiments and template provided in this repository all note where the change(s) in _GradientUpdates_ should be made and provide an alternate loss function as an example.

## Experiments
This repository provides code to reproduce some of the experimental results presented in the accompanying technical report.

### Movielens
To reproduce the Movielens 100k and 1M experiments, you will need to download the data sets from [the grouplens website](http://grouplens.org/datasets/movielens/).  For our experiments, 5000 random ratings were used as test data points, and the remaining points were used to train GPUFish.  Once the data file is in the format described above, the code provided will perform *1-bit matrix completion<sup>[2](#footnote2)</sup>* as described in the technical report provided with this code.  It will output (to the console) the percentage of total like/dislike ratings correctly predicted, as well the percentage of ratings correctly predicted as a function of the correct rating.  The file _gpu_fish_movielens.cu_ will perform this experiment.

### Rank-1 Recovery
The training and test data sets used to measure the ability of GPUFish to recover a rank=1 matrix are available in the _data_ folder.
The file _gpu_fish_rank1_recovery.cu_ will perform this experiment.

To repeat either the Rank-1 recovery experiment or the Movielens experiment compile with the command:
```
nvcc -std=c++11 gpu_fish_functions.cu <either_experiment.cu>
```
and run the experiment with
```
./a.out
```

## Authors

* **Charles Hubbard**, Iowa State University
* **Chinmay Hegde**, Iowa State University

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Contact
Please contact hubbardc@iastate.edu with bugs, comments or questions.
## Acknowledgments

* Many thanks to the creators of Jellyfish<sup>[1](#footnote1)</sup> and  1-bit matrix completion<sup>[2](#footnote2)</sup> for their inspirational work!
* Thanks [PurpleBooth](https://github.com/PurpleBooth) for the template used to create this README.

<a name="footnote1">1</a>: Benjamin Recht and Christopher Re.  Parallel stocastic gradient algorithms for large-scale matrix completion.  _Mathematical Programming Computation_, 5(2):201-226, 2013.

<a name="footnote2">2</a>: Mark A Davenport, Yaniv Plan, Ewout van den Berg, and Mary Wootters. 1-bit matrix completion. Information and Inference, 3(3):189–223, 2014.

