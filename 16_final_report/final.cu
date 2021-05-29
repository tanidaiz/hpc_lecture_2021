#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

double err_calc(vector<float>,vector<float>,float*,int);

__global__ void kernel(float *A, float *B, float *C, int N, int offset, int width){
  //printf("block:%d/%d thread:%d/%d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
  //int i = blockIdx.y;
  //int j = threadIdx.x + blockDim.x * blockIdx.x;
  //int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i = 0;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("%d:%d\n", i, j);
  float sum = 0.0f;
  /*extern __shared__ float A_s[];
  for (int ks=0; ks<N; ks+=blockDim.x) {
    __syncthreads();
    A_s[threadIdx.x] = A[N*i+ks+threadIdx.x];
    __syncthreads();
    for (int k=ks; k<ks+blockDim.x; k++) {
      sum += A_s[k-ks] * B[width*k+j];
    }
  }
  C[N*i+j+offset] = sum;*/
  
  extern __shared__ float A_s[];
  for(int row=0;row<width;row++){
      __syncthreads();
      /*A_s[threadIdx.x] = A[row*N+threadIdx.x];
      A_s[threadIdx.x+width] = A[row*N+threadIdx.x+width];
      A_s[threadIdx.x+width*2] = A[row*N+threadIdx.x+width*2];
      A_s[threadIdx.x+width*3] = A[row*N+threadIdx.x+width*3];*/
      A_s[threadIdx.x*4] = A[row*N+threadIdx.x*4];
      A_s[threadIdx.x*4+1] = A[row*N+threadIdx.x*4+1];
      A_s[threadIdx.x*4+2] = A[row*N+threadIdx.x*4+2];
      A_s[threadIdx.x*4+3] = A[row*N+threadIdx.x*4+3];
      __syncthreads();
      
      float sum = 0.0;
      for(int k=0;k<N;k++){
          sum += A_s[k] * B[width*k+threadIdx.x];
      }
      C[row*N+threadIdx.x+offset] = sum;
  }
  
  /*for(int k=0;k<N;k++){
    sum += A[N*i+k]*B[width*k+j];
  }
  C[N*i+j+offset] = sum;// 65.6GFlops
  */
}

int main(int argc, char** argv) {
  int size, rank;
  int gpusize, gpurank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaGetDeviceCount(&gpusize);
  cudaSetDevice(rank % gpusize);
  cudaGetDevice(&gpurank);
  printf("MPI rank: %d/%d  GPU device: %d/%d\n", rank, size, gpurank, gpusize);

  const int N = 1024;
  const int M = 1024;
  vector<float> A(N*N);
  vector<float> B(N*N);
  float C[N*N];
  //float subA[N*N/size];
  //float subB[N*N/size*2]; // *2 for MPI send/recv buffer
  //float subC[N*N/size];
  float *subA, *subB, *subB_cpu, *subC;
  cudaMallocManaged(&subA, N*N/size*sizeof(float));
  //cudaMallocManaged(&subB, N*N/size*2);
  cudaMallocManaged(&subC, N*N/size*sizeof(float));
  subB_cpu = (float*)malloc(N*N/size*sizeof(float)*2); // *2 for MPI send/recv buffer
  cudaMalloc(&subB, N*N/size*sizeof(float));
  
  
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j]; // some rows
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB_cpu[N/size*i+j] = B[N*i+j+offset]; // some columns
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;
  printf("I'm %d  from %d  to %d\n", rank, recv_from, send_to);

  double comp_time = 0, comm_time = 0;
  int buffering_offset = N*N/size;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    
    offset = N/size*((rank+irank) % size);
    /*for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++)
        for (int k=0; k<N; k++)
          subC[N*i+j+offset] += subA[N*i+k] * subB[buffering_offset*(irank%2)+N/size*k+j];*/
      
    cudaMemcpy(subB, subB_cpu+buffering_offset*(irank%2), N*N/size*sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<N/size,N/size, N*sizeof(float)>>>(subA, subB, subC, N, offset, N/size);
    cudaDeviceSynchronize();
    
    
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    MPI_Request request[2];
    MPI_Isend(&subB_cpu[buffering_offset*(irank%2)], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&subB_cpu[buffering_offset*((irank+1)%2)], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
    /*for(int i=0;i<N/size;i++){
        for(int j=0;j<N/size;j++){
            
            //printf("%lf\n", C[i*N+j]);
            if(subC[i*N+j+offset]==0.0){
                
                printf("i:%d, j:%d  zero!\n", i, j);
                //goto hoge;
            }
        }
    }*/
  printf("here1\n");
  /*float *subC_cpu = (float*)malloc(N*N/size*sizeof(float));
  for(int i=0;i<N/size;i++){
      for(int j=0;j<N;j++){
          subC_cpu[i*N+j] = subC[i*N+j];
          if(subC_cpu[i*N+j]==0.0){
              printf("0!\n");
          }
      }
  }*/
  printf("here2\n");
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);

  
  // faster verification
  
  double err = err_calc(A,B,C,N);
  
  if(rank==0) {
    for(int i=99;i<100;i++){
        for(int j=0;j<N;j++){
            printf("[%d,%d] %lf\n", i, j, C[i*N+j]);
        }
    }
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  cudaFree(subA);
  cudaFree(subB);
  cudaFree(subC);
  MPI_Finalize();
}
