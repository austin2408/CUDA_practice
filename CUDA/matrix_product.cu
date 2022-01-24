#include <stdio.h>
#include <cuda.h>
#include<iostream>
#include <cuda_runtime.h>
#include <math.h>

#define A_row 5
#define A_col 2
#define B_row 2
#define B_col 4

__global__ void gpu_product(float* A, float* B, float* result){
    // 0~4
    int row = threadIdx.x;

    // 0~3
    int col = threadIdx.y;

    for (int i=0; i<A_col; i++){
        // printf("row : %d || col : %d || i : %d\n",row*B_col + col, A[row*A_col + i], B[i*B_col + col]);

        result[row*B_col + col] += A[row*A_col + i]*B[i*B_col + col];
    }

}

// time function
double ms_time(){
    return (double)clock()/CLOCKS_PER_SEC*1000.0;
}

int main(){

    float A_m[A_row][A_col], B_m[B_row][B_col], C_m[A_row][B_col];

    float* A, * B, * C;

    // set A matrix
    printf("a Matrix: \n");
	for (int i = 0; i < A_row; i++)
	{
		for (int j = 0; j < A_col; j++)
		{
			A_m[i][j] = (float)rand()/RAND_MAX*2-1;
            // A_m[i][j] = i;
			printf("%f   ", (float)A_m[i][j]);
		}
		printf("\n");
	}
    printf("\n");

    // Set B matrix
    printf("b Matrix: \n");
	for (int i = 0; i < B_row; i++)
	{
		for (int j = 0; j < B_col; j++)
		{
			B_m[i][j] = (float)rand()/RAND_MAX*2-1;
            // B_m[i][j] = j;
			printf("%f   ", (float)B_m[i][j]);
		}
		printf("\n");
	}
    printf("\n");

    // Define device's (GPU) memory
    cudaMalloc((void **)&A, A_row * A_col * sizeof(float));
    cudaMalloc((void **)&B, B_row * B_col * sizeof(float));
    cudaMalloc((void **)&C, A_row * B_col * sizeof(float));

    // pass data to device
    cudaMemcpy(A, A_m, A_row * A_col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_m, B_row * B_col * sizeof(float), cudaMemcpyHostToDevice);

    dim3 numBlock(A_row, B_col, 1);

    // implement kernel
    double t0 = ms_time();
    gpu_product<<<1, numBlock>>>(A, B, C);
    double device_time = ms_time() - t0;

    cudaDeviceSynchronize();

    // pass data to host
    cudaMemcpy(C_m, C, A_row * B_col * sizeof(float), cudaMemcpyDeviceToHost);

    // Show result
    printf("c Matrix: \n");
	for (int i = 0; i < A_row; i++)
	{
		for (int j = 0; j < B_col; j++)
		{
			printf("%f   ", (float)C_m[i][j]);
		}
		printf("\n");
	}
    printf("Time of gpu : %g\n", device_time);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;



}