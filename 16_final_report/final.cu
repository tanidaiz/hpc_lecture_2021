// max 922.212659 GFlops with F_node P100x4

#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

double err_calc(vector<float>,vector<float>,float*,int);

__global__ void kernel(float *A, float *B, float *C, int N, int offset, int width, int size){
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  extern __shared__ float A_s[];
    
  __syncthreads();
  /*
  A_s[threadIdx.x] = A[N*i+threadIdx.x];
  A_s[threadIdx.x+blockDim.x] = A[N*i+threadIdx.x+blockDim.x];
  A_s[threadIdx.x+blockDim.x*2] = A[N*i+threadIdx.x+blockDim.x*2];
  A_s[threadIdx.x+blockDim.x*3] = A[N*i+threadIdx.x+blockDim.x*3];
  */
  /*A_s[threadIdx.x*4] = A[N*i+threadIdx.x*4];
  A_s[threadIdx.x*4+1] = A[N*i+threadIdx.x*4+1];
  A_s[threadIdx.x*4+2] = A[N*i+threadIdx.x*4+2];
  A_s[threadIdx.x*4+3] = A[N*i+threadIdx.x*4+3];
  */
  /*if(threadIdx.x==0 && threadIdx.y==0){
      for(int k=0;k<N;k++){
          A_s[k] = A[N*i+k];
      }
  }*/
  for(int k=0;k<size;k++){
      A_s[threadIdx.y*size+k] = A[N*i+threadIdx.y*size+k];
  }
  __syncthreads();
  
    
  for(int k=0;k<N;k++){
    sum += A_s[k] * B[k*width+j];
  }
  C[N*i+j+offset] = sum;
  
  
  /* the most simple code
  
  for(int k=0;k<N;k++){
    sum += A[N*i+k]*B[width*k+j];
  }
  C[N*i+j+offset] = sum;
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

  const int N = 2048;
  vector<float> A(N*N);
  vector<float> B(N*N);
  float C[N*N];
  float *subA, *subB, *subB_cpu, *subC;
  cudaMallocManaged(&subA, N*N/size*sizeof(float));
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
  //printf("I'm %d  from %d  to %d\n", rank, recv_from, send_to);

  double comp_time = 0, comm_time = 0;
  int buffering_offset = N*N/size;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    
    offset = N/size*((rank+irank) % size);
      
    cudaMemcpy(subB, subB_cpu+buffering_offset*(irank%2), N*N/size*sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<dim3(N/size, N/size/(N/size)),dim3(1,(N/size)), N*sizeof(float)>>>(subA, subB, subC, N, offset, N/size, size);
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
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);

  
    
  if(rank==0) {
    // faster verification using OpenMP
    double err = err_calc(A,B,C,N);
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
