%%cuda

#include <stdio.h>
#define N 2048*2048


//kernel definition
__global__ void add(int *a, int *b, int *c) { // we put global to access it from CPU to GPU
    int i = blockIdx.x;
    c[i]= a[i] + b[i];
}

//host code
int main() {
    //declare variables
    int *h_a, *h_b, *h_c; //host
    int *d_a, *d_b, *d_c; //device
    int size = N*sizeof(int);

    //Allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    //Allocate memory and initialize values on host
    h_a = (int *)malloc(size);
        for (int i = 0; i < N; i++) {
            h_a[i] = rand() % 1000;} // rand numbers are taken from seed 1 by default, for every iteration
    h_b = (int *)malloc(size);
        for (int i = 0; i < N; i++) {
            h_b[i] = rand() % 1000;} // rand numbers are taken from seed 1 by default, for every iteration
    h_c = (int *)malloc(size);

    //Copy data from host to device memory
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //Kernel launch
    add<<<N,1>>>(d_a, d_b, d_c); // N represents the number of blocks, 1 represents the number of threads per block

    //Synchronize (wait for completion of kernel)
    cudaDeviceSynchronize(); // blocks host until kernel completion

    //Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //Print results
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d \n",h_a[i],h_b[i],h_c[i]);
    }

    //Free allocated memory
    free(h_a); free(h_b); free(h_c); // host memory free up
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); // device memory free up

    return 0;
}
