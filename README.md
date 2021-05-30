21M31217

# usage
1. modules.txtに記載されているmoduleをloadする
1. make.shを実行する
1. mpirun -n 4 ./a.out

# memo
- OpenMPIとCudaを使用して計算した。
- shared memoryを使用した。
- P100 1枚で 447 GFlops、P100 4枚で 922 GFlops の速度が出る
- OpenMPを使用して、verificationを高速化した (err_calc.cpp)
    - ファイルを分けたのは、OpenMP・OpenMPI・Cudaの全てを利用して同時にコンパイルするのが難しかったため
- MPIの通信用・計算用バッファのために、ダブルバッファリングを使用して、通信用と計算用バッファの間のコピーを不要にした

# hpc_lecture

|          | Topic                                | Sample code               |
| -------- | ------------------------------------ | ------------------------- |
| Class 1  | Introduction to parallel programming |                           |
| Class 2  | Shared memory parallelization        | 02_openmp                 |
| Class 3  | Distributed memory parallelization   | 03_mpi                    |
| Class 4  | SIMD parallelization                 | 04_simd                   |
| Class 5  | GPU programming 1                    | 05_openacc                |
| Class 6  | GPU programming 2                    | 06_cuda                   |
| Class 7  | Parallel programing models           | 07_starpu                 |
| Class 8  | Cache blocking                       | 08_cache_cpu,08_cache_gpu |
| Class 9  | High Performance Python              | 09_python                 |
| Class 10 | I/O libraries                        | 10_io                     |
| Class 11 | Parallel debugger                    | 11_debugger               |
| Class 12 | Parallel profiler                    | 12_profiler               |
| Class 13 | Containers                           |                           |
| Class 14 | Scientific computing                 | 14_pde                    |
| Class 15 | Deep Learning                        | 15_dl                     |
