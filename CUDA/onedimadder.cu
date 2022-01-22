#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// GPU Adder
__global__ void gpu_adder(float *sum, float *a, float *b, int N){
    for (int i=0; i<N; i++){
        sum[i] = a[i] + b[i];
    }
}

__global__ void gpu_adder_parallel(float *sum, float *a, float *b){
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    sum[id] = a[id] + b[id];
}

// HOST Adder
void host_adder(float *sum, float *a, float *b, int N){
    for (int i=0; i<N; i++){
        sum[i] = a[i] + b[i];
    }
}

// time function
double ms_time(){
    return (double)clock()/CLOCKS_PER_SEC*1000.0;
}

int main(){
    int n = 1024*1024;
    int size = n*sizeof(float);
    bool check = true;

    int gridim = 2048;
    int blockdim = n/gridim;

    // Memory for host
    float *a, *b, *c, *gpu_sum;
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    gpu_sum = (float*)malloc(size);

    // Memory for device
    float *ga, *gb, *gc;
    cudaMalloc((void**)&ga, size);
    cudaMalloc((void**)&gb, size);
    cudaMalloc((void**)&gc, size);

    // Randomly initial vector
    for (int i=0; i<n; i++){
        a[i] = (float)rand()/RAND_MAX*2-1;
        b[i] = (float)rand()/RAND_MAX*2-1;
    }

    cudaMemcpy(ga, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gb, b, size, cudaMemcpyHostToDevice);

    

    double t0 = ms_time();
    gpu_adder_parallel<<<gridim,blockdim>>>(gc, ga, gb);
    // gpu_adder<<<1,1>>>(gc, ga, gb, n);
    double device_time = ms_time() - t0;

    cudaMemcpy(gpu_sum, gc, size, cudaMemcpyDeviceToHost);

    double t1 = ms_time();
    host_adder(c, a, b, n);
    double host_time = ms_time() - t1;

    printf("Time of gpu : %g\n", device_time);
    printf("Time of host : %g\n", host_time);

    for (int i=0; i<n; i++){
        if (gpu_sum[i] != c[i]){
            printf("Error %d\n", i);
            printf("Error : gpu : %f || host : %f\n", gpu_sum[i], c[i]);
            check = false;
            break;
        }
    }

    if (check){
        printf("PASS\n");
    }

    free(a);
    free(b);
    free(c);
    free(gpu_sum);

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);

    return 0;

}

