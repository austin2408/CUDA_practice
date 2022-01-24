#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define A_row 5
#define A_col 2
#define B_row 2
#define B_col 4

__global__ void gpu_product(int *A, int *B, int *C){
    // 0~4
    int row = threadIdx.x;

    // 0~3
    int col = threadIdx.y;

    for (int i=0; i<A_col; i++){
        C[i*B_col + col] += A[row*A_col + i]*B[B_col + col];
    }
}

int main(){

    int A_m[A_row][A_col], B_m[B_row][B_col], C_m[A_col][B_row];

    int *A, *B, *C;

    printf("a Matrix: \n");
	for (int i = 0; i < A_row; i++)
	{
		for (int j = 0; j < A_col; j++)
		{
			A_m[i][j] = i;
			printf("%d   ", (int)A_m[i][j]);
		}
		printf("\n");
	}

    printf("b Matrix: \n");
	for (int i = 0; i < B_row; i++)
	{
		for (int j = 0; j < B_col; j++)
		{
			B_m[i][j] = i;
			printf("%d   ", (int)B_m[i][j]);
		}
		printf("\n");
	}

    cudaMalloc((void **)&A, A_row * A_col * sizeof(int));
    cudaMalloc((void **)&B, B_row * B_col * sizeof(int));
    cudaMalloc((void **)&C, A_col * B_row * sizeof(int));

    cudaMemcpy(A, A_m, A_row * A_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_m, B_row * B_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_m, A_col * B_row * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numBlock(A_row, B_col);

    gpu_product<<<1, numBlock>>>(A, B, C);

    cudaDeviceSynchronize();

    cudaMemcpy(C_m, C, A_col * B_row * sizeof(int), cudaMemcpyDeviceToHost);

    printf("c Matrix: \n");
	for (int i = 0; i < A_row; i++)
	{
		for (int j = 0; j < B_col; j++)
		{
			A_m[i][j] = i;
			printf("%d   ", (int)C_m[i][j]);
		}
		printf("\n");
	}

    free(A_m);
    free(B_m);
    free(C_m);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;



}