#include <cmath>
#include <vector>
#include <cstdio>
using namespace std;

double err_calc(vector<float> A, vector<float> B, float *C, int N){
#pragma omp parallel for collapse(2)
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++){
            //float temp = 0.0;
            for (int k=0; k<N; k++){
                C[N*i+j] -= A[N*i+k] * B[N*k+j];
            }
        }
    double err = 0.0;
#pragma omp parallel for collapse(2) reduction(+:err)
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            err += fabs(C[N*i+j]);
    return err;
}