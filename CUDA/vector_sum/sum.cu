#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Tells the compiler this is a function to run in the GPU
__global__ void vectorAdd(int *a,int *b,int *c){
    
    int i = threadIdx.x; // Get the index of the thread
    c[i] = a[i] + b[i];

    return;
}


int main(){
    int a[] = {0,1,2,3,4,5,6};
    int b[] = {0,1,2,3,4,5,6};
    int c[sizeof(a)/sizeof(int)] = {0};

    // Create pointers into the GPU
    int *cudaA = 0; // Pointer to the GPU memory
    int *cudaB = 0;
    int *cudaC = 0;

    // Allocate memory in the GPU 
    cudaMalloc(&cudaA,sizeof(a));
    cudaMalloc(&cudaB,sizeof(b));
    cudaMalloc(&cudaC,sizeof(c));

    // Send data to the GPU 
    cudaMemcpy(&cudaA,a,sizeof(a),cudaMemcpyHostToDevice);
    cudaMemcpy(&cudaB,b,sizeof(b),cudaMemcpyHostToDevice);

    // Start computation in the GPU (I'm telling CUDA to generate a kernel with one block and N threads)
    // vectorAdd<<<GRID_SIZE,BLOCK_SIZE>>>
    vectorAdd <<< 1, sizeof(a) / sizeof(int) >>> (cudaA,cudaB,cudaC);

    // Get return value
    cudaMemcpy(c,cudaC,sizeof(c),cudaMemcpyDeviceToHost);

    printf("Results of the computation is the following vector:\n");
    for(int i = 0; i < (sizeof(c)/sizeof(int)); i++){
        printf("%d ",c[i]);
    }
    puts("");
    


    return 0;
}