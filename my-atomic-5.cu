#include <stdio.h>
#include <cuda.h>

#define N 1<<7
#define THREADS_PER_BLOCK 1024

__global__ void dot(float *a, float *b, float *c) {
    __shared__ float temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) return;
    temp[threadIdx.x] = a[index] * b[index];

    __syncthreads();
    if (0 == threadIdx.x) {
        float sum = 0.0;
        int max = THREADS_PER_BLOCK;
        if (N < max) max = N;

        for (int i = 0; i < max; i++) {
            sum += temp[i];
        }
        //c[0] = sum;
        atomicAdd(c, sum);
    }
}

void random_floats(float *a, float size)
{
    int i;
    for (i=0; i<size; i++)
        a[i] = i;
    return;
}

int main(void) {
    int i;
    float result;
    float *a, *b, *c; // host copies of a, b, c
    float *dev_a, *dev_b, *dev_c; // device copies of a, b, c
    int size = N * sizeof(float); // we need space for N floats
    // allocate device copies of a, b, c
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );
    cudaMalloc( (void**)&dev_c, sizeof(float) );
    a = (float*)malloc( size );
    b = (float*)malloc( size );
    c = (float*)malloc( sizeof(float) );

    random_floats( a, N );
    random_floats( b, N );
    /*
    printf("a = ");
    for (i=0; i<N; i++) printf("%d, ", a[i]);
    printf("\n");
    printf("b = ");
    for (i=0; i<N; i++) printf("%d, ", b[i]);
    printf("\n");
    */
    result = 0;
    for (i=0; i<N; i++) result += a[i] * b[i];
    *c = 0;

    // copy inputs to device
    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice);

    int blocks = (int)(N/THREADS_PER_BLOCK) + 1; // ceil(...)
    //if(blocks<1) blocks=1;

    // launch dot() kernel
    dot <<< blocks, THREADS_PER_BLOCK >>> (dev_a, dev_b, dev_c);

    // copy device result back to host copy of c
    cudaMemcpy(c, dev_c, sizeof(float) , cudaMemcpyDeviceToHost);

    printf("*c     = %.2f\n", *c);
    printf("result = %.2f\n", result);

    free(a); free(b); free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

