#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 512
#define BLOCK_SIZE 16

// Kernel of GPU
__global__ void MatAdd(float *A, float *B, float *C)
{
    // thread id of x, y according to designed CUDA model
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Linear index
    int idx = j * N + i;

    if (i < N && j < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    int i;

    // Define host's memory
    h_A = (float *)malloc(N * N * sizeof(float));
    h_B = (float *)malloc(N * N * sizeof(float));
    h_C = (float *)malloc(N * N * sizeof(float));

    // initial host's variable
    for (i = 0; i < (N * N); i++)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 0.0;
    }

    // Define device's (GPU) memory
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // pass data to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // implement kernel
    MatAdd<<<numBlock, blockSize>>>(d_A, d_B, d_C);

    // wait finishing computing
    cudaDeviceSynchronize();

    // pass data to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // evalution
    for (i = 0; i < (N * N); i++)
    {
        if (h_C[i] != 3.0)
        {
            printf("Error:%f, idx:%d\n", h_C[i], i);
            break;
        }
    }

    printf("PASS\n");

    // free memory

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}