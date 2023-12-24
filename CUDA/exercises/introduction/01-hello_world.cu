// Copyright 2014, Cranfield University
// All rights reserved
// Author: Michał Czapiński (mczapinski@gmail.com)
//
// Demonstrates the most basic CUDA concepts on the example
// of single precision AXPY operation.
// AXPY stands for y = y + alpha * x, where x, and y are vectors.

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#define VISUALIZATION_CONSTANT 10 // Constat used for disply elements in the vector for debugging

// With this implementation and 256 threads per block, works only for up to 16M. Why?
const int N = 15 * 1024 * 1024;
const dim3 BLOCK_DIM = 256;

// Simple CPU implementation of a single precision AXPY operation.
void CpuSaxpy(int n, float alpha, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

// Function to display vector values weel formatted
void printVector(float *h_y, int n){

  std::cout << "\t";

  for(int i = 0;i<n;++i){
    std::cout << h_y[i] << " ";
  }

  std::cout << "\n";
}


// GPU implementation of AXPY operation - one CUDA thread per vector element.
__global__ void GpuSaxpy(int n, float alpha, const float* x, float* y) {
  // Calculate the index of the vector element updated by this thread.
  // Assume 1D grid of blocks.
  int idx = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

  if (idx < n && idx >= 0) {
    y[idx] += alpha * x[idx];
  }

}

// GPU implementation of AXPY operation - CUDA thread updates multiple vector elements.
__global__ void GpuSaxpyMulti(int n, float alpha, const float* x, float* y) {
  // Implement CUDA kernel where threads update more than one vector element.
  // Assume 1D grid of blocks.
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_count = gridDim.x * blockDim.x;

  for (; idx < n; idx += thread_count) {
    y[idx] += alpha * x[idx];
  }


  // TODO(later) Check if it's faster than the original implementation.
}

int main(int argc, char** argv) {

  std::cout << "\n-------------------START-------------------]\n";
  std::cout << "[INFO] Starting AXPY computation...\n";

// ----------------------- Host memory initialisation ----------------------- //

  float* h_x = new float[N];
  float *h_y = new float[N];
  float* h_1_y = new float[N];
  float* h_2_y = new float[N];
  float timer_time = 0.0f;


  // Initialise vectors on the CPU.
  std::fill_n(h_x, N, 1.0f);  // Vector of ones

  for (int i = 0; i < N; ++i) {
    h_y[i] = 0.33f * (i + 1);
    h_1_y[i] = 0.33f * (i + 1);
    h_2_y[i] = 0.33f * (i + 1);
  }

  std::cout << "[INFO] First 10 elements of the first vector are:\n";
  printVector(h_1_y,VISUALIZATION_CONSTANT);
  std::cout << "[INFO] First 10 elements of the second vector are:\n";
  printVector(h_2_y,VISUALIZATION_CONSTANT);

// ---------------------- Device memory initialisation ---------------------- //


  // Allocate global memory on the GPU.
  float* d_x = 0;
  float* d_y = 0;
  cudaError_t cudaRet = cudaSuccess;

  cudaRet = cudaMalloc((void **)&d_x, N*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the first vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMalloc((void **)&d_y, N*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "[INFO] Allocation of memory on the GPU finished with succes!\n";
  fflush(stdout);


  // Copy vectors from the host (CPU) to the device (GPU).
  cudaRet = cudaSuccess;

  cudaRet = cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the first vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "[INFO] Memory correctly copied from HOST to GPU!\n";

// --------------------- Calculations for CPU implementation ---------------- //

  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  std::cout << "\n****************************************\n";
  std::cout << "[INFO] Starting CPU computation...\n";


  timer->reset();
  timer->start();
  CpuSaxpy(N, 0.25f, h_x, h_1_y);  // y = y + 0.25 * x;
  timer->stop();

  timer_time = timer->getTime();

  timer->reset();
  timer->start();
  CpuSaxpy(N, -10.5f, h_x, h_2_y);  // y = y - 10.5 * x;
  timer->stop();

  timer_time += timer->getTime();

  std::cout << "\n\tCPU time: " << timer_time << " ms." << std::endl << "\n";
  std::cout << "[INFO] CPU computation ended correctly\n";
  std::cout << "[INFO] CPU computation first 10 results for first computation:\n";
  printVector(h_1_y,VISUALIZATION_CONSTANT);
  std::cout << "[INFO] CPU computation first 10 results for second computation:\n";
  printVector(h_2_y,VISUALIZATION_CONSTANT);
  fflush(stdout);
  std::cout << "****************************************\n";

// --------------------- Calculations for GPU implementation ---------------- //

  // Calculate the number of required thread blocks (one thread per vector element).
  const dim3 GRID_DIM = N/256;
  float* device_result_vector_1_y = new float[N];
  float* device_result_vector_2_y = new float[N];



  std::cout << "\n****************************************\n";
  std::cout << "[INFO] Starting GPU kernels\n";


  cudaRet = cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  timer->reset();
  timer->start();
  GpuSaxpy<<<GRID_DIM,BLOCK_DIM>>>(N, 0.25f, d_x, d_y);  // # blocks, # thread per block
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();

  timer_time = timer->getTime();

  cudaRet = cudaSuccess;

  // Download the resulting vector d_y from the device and store it in h_1_y.
  cudaRet = cudaMemcpy(device_result_vector_1_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the DEVICE to the HOST for the first vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }


  cudaRet = cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  timer->reset();
  timer->start();
  GpuSaxpy<<<GRID_DIM,BLOCK_DIM>>>(N, -10.5f, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();

  timer_time += timer->getTime();


  cudaRet = cudaSuccess;

  // Download the resulting vector d_y from the device and store it in h_2_y.
  cudaRet = cudaMemcpy(device_result_vector_2_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the DEVICE to the HOST for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  // Kernel calls are asynchronous with respect to the host, i.e. control is returned to
  // the CPU immediately. It is possible that the second operation is submitted _before_
  // the first one is completed. However, CUDA driver will ensure that they will be
  // completed in FIFO order, one at a time.

  // CPU has to explicitly wait for the device to complete
  // in order to get meaningful time measurement.

  std::cout << "\n\tGPU time: " << timer_time << " ms." << std::endl << "\n";

  // Print result vectors
  std::cout << "[INFO] GPU computation first 10 results for first computation:\n";
  printVector(device_result_vector_1_y,VISUALIZATION_CONSTANT);
  std::cout << "[INFO] GPU computation first 10 results for second computation:\n";
  printVector(device_result_vector_2_y,VISUALIZATION_CONSTANT);

  float diff1 = 0.0f;
  float diff2 = 0.0f;
  for (int i = 0; i < N; ++i) {
    diff1 = std::max(diff1, std::abs(device_result_vector_1_y[i] - h_1_y[i]));
  }

  for (int i = 0; i < N; ++i) {
    diff2 = std::max(diff2, std::abs(device_result_vector_2_y[i] - h_2_y[i]));
  }



  cudaRet = cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }


  // Test using monodimensional grid
  timer_time = 0.0f;

  timer->reset();
  timer->start();
  GpuSaxpyMulti<<<N/256,BLOCK_DIM>>>(N, 0.25f, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();

  timer_time = timer->getTime();




  cudaRet = cudaSuccess;

  // Download the resulting vector d_y from the device and store it in device_return_vector_1 with multi flag.
  cudaRet = cudaMemcpy(device_result_vector_1_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the DEVICE to the HOST for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }


  cudaRet = cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  timer->reset();
  timer->start();
  GpuSaxpyMulti<<<N/256,BLOCK_DIM>>>(N, -10.5f, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();

  timer_time += timer->getTime();

  std::cout << "\n\tGPU time using multi flag: " << timer_time << " ms." << std::endl << "\n";

  cudaRet = cudaSuccess;

  // Download the resulting vector d_y from the device and store it in device_return_vector_1 with multi flag.
  cudaRet = cudaMemcpy(device_result_vector_2_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the DEVICE to the HOST for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }


  // Print result vectors
  std::cout << "[INFO] GPU computation first 10 results for first computation with multi flag:\n";
  printVector(device_result_vector_1_y,VISUALIZATION_CONSTANT);
  std::cout << "[INFO] GPU computation first 10 results for second computation with multi flag:\n";
  printVector(device_result_vector_2_y,VISUALIZATION_CONSTANT);


  std::cout << "[INFO] Memory correctly copied from DEVICE to HOST!\n";

  // cudaMemcpy is synchronous, i.e. it will wait for any computation on the GPU to
  // complete before any data is copied (as if cudaDeviceSynchronize() was called before).


  std::cout << "\n[INFO] GPU computation completed\n";
  std::cout << "****************************************\n";

  // Now let's check if the results are the same.

  std::cout << "\n\tFirst vector max diff = " << diff1 << std::endl << "\n";  // Should be (very close to) zero.
  std::cout << "\n\tSecond vector max diff = " << diff2 << std::endl << "\n";  // Should be (very close to) zero.


  diff1 = 0.0f;
  diff2 = 0.0f;

  for (int i = 0; i < N; ++i) {
    diff1 = std::max(diff1, std::abs(device_result_vector_1_y[i] - h_1_y[i]));
  }
  std::cout << "\n\tFirst vector max diff = " << diff1 << " using multi flag\n" << std::endl;  // Should be (very close to) zero.


  for (int i = 0; i < N; ++i) {
    diff2 = std::max(diff2, std::abs(device_result_vector_2_y[i] - h_2_y[i]));
  }
  std::cout << "\n\tSecond vector max diff = " << diff2 << " using multi flag\n" << std::endl;  // Should be (very close to) zero.


// ------------------------------- Cleaning up ------------------------------ //

  delete timer;
  delete[] h_x;
  delete[] h_1_y;
  delete[] h_2_y;
  delete[] device_result_vector_1_y;
  delete[] device_result_vector_2_y;

  // Don't forget to free host and device memory!

  cudaRet = cudaSuccess;

  cudaRet = cudaFree(d_x);

  if(cudaRet == cudaErrorInvalidValue){
    std::cout << "[ERROR] Error in clean up code due to cudaFree for the first vector, error returned code: " << cudaRet << "\n";
    return -1;
  }

  cudaRet = cudaFree(d_y);

  if(cudaRet == cudaErrorInvalidValue){
    std::cout << "[ERROR] Error in clean up code due to cudaFree for the second vector, error returned code: " << cudaRet << "\n";
    return -1;
  }

  std::cout << "[INFO] Correctly cleaned up environment\n";
  std::cout << "[INFO] Computation finished\n";
end:
  std::cout << "[-------------------END-------------------]\n\n";

  return 0;
}
