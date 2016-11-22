FUNC_OBJ = gpu_fish_functions.o
ML_OBJ = gpu_fish_movielens.o $(FUNC_OBJ)
R1_OBJ = gpu_fish_rank1_recovery.o $(FUNC_OBJ)

NVCC = nvcc
LFLAGS = -std=c++11 -O2
CFLAGS = -std=c++11 -O2 -c
ARCH = 

movielens : $(ML_OBJ)
	$(NVCC) $(LFLAGS) $(ARCH) $(ML_OBJ) -o movies.out

rank1 : $(R1_OBJ)
	$(NVCC) $(LFLAGS) $(ARCH) $(R1_OBJ) -o rank1.out

gpu_fish_movielens.o : gpu_fish_headers.h
	$(NVCC) $(CFLAGS) gpu_fish_movielens.cu

gpu_fish_functions.o : gpu_fish_headers.h
	$(NVCC) $(CFLAGS) gpu_fish_functions.cu

gpu_fish_rank1_recovery.o : gpu_fish_headers.h
	$(NVCC) $(CFLAGS) gpu_fish_rank1_recovery.cu

clean:
	\rm *.o *.out 
