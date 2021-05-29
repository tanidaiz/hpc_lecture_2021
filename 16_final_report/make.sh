# module load cuda/10.2.89 openmpi gcc/8.3.0-cuda
nvcc -c final.cu -lmpi
g++ -c err_calc.cpp -fopenmp
g++ final.o err_calc.o -g -lmpi -fopenmp -L/usr/local/cuda/lib64 -lcuda -lcudart

