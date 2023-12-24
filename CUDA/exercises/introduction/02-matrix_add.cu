// Copyright 2014, Cranfield University
// All rights reserved
// Author: Michał Czapiński (mczapinski@gmail.com)
//
// Adds two matrices on the GPU. Matrices are stored in linear memory in row-major order,
// i.e. A[i, j] is stored in i * COLS + j element of the vector.

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#define VISUALIZATION_CONSTANT 16 // Constat used for disply elements in the vector for debugging


// Matrix dimensions. Can you make these input arguments?
const int ROWS = 4096;
const int COLS = 4096;

// TODO(later) Play a bit with the block size. Is 16x16 setup the fastest possible?
// Note: For meaningful time measurements you need sufficiently large matrix.
const dim3 BLOCK_DIM(16, 16);

// Simple CPU implementation of matrix addition.
void CpuMatrixAdd(int rows, int cols, const float* A, const float* B, float* C) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int idx = row * cols + col;
      C[idx] = A[idx] + B[idx];
    }
  }
}

// GPU implementation of matrix add using one CUDA thread per vector element.
__global__ void GpuMatrixAdd(int rows, int cols, const float* A, const float* B, float* C) {
  // Calculate indices of matrix elements added by this thread. Assume 2D grid of blocks.
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_count_row = gridDim.y * blockDim.y;
  int thread_count_col = gridDim.x * blockDim.x;

  // TODO(later) Does it matter if you index rows with x or y dimension of threadIdx and blockIdx?

  // Calculate the element index in the global memory and add the values.
  for (; row < rows; row += thread_count_row) {
    for (; col < cols; col += thread_count_col){
      int idx = row * cols + col;
      C[idx] = A[idx] + B[idx];
    }
  }

  // TODO Make sure that no threads access memory outside the allocated area.
}


void printMatrix(const float *h_A,int ROWS,int COLS){
  std::cout << "________________________________________________________________________________________________________________________\n";
  for (int row = 0; row < VISUALIZATION_CONSTANT; ++row) {
    std::cout << "| ";
    for (int col = 0; col < VISUALIZATION_CONSTANT; ++col) {
      int idx = row * COLS + col;
      std::cout << h_A[idx] << " ";
    }
    std::cout << "\t|\n";
  }
  std::cout << "________________________________________________________________________________________________________________________\n\n";
}


int main(int argc, char** argv) {

// ----------------------- Host memory initialisation ----------------------- //

  // INIT MATIX
  float* h_A = new float[ROWS * COLS];
  float* h_B = new float[ROWS * COLS];
  float* h_C = new float[ROWS * COLS];


  // CPU MATIX
  float* h_c_A = new float[ROWS * COLS];
  float* h_c_B = new float[ROWS * COLS];
  float* h_c_C = new float[ROWS * COLS];

  // GPU MATIX
  float* h_g_A = new float[ROWS * COLS];
  float* h_g_B = new float[ROWS * COLS];
  float* h_g_C = new float[ROWS * COLS];



  std::cout << "\n------------------------------------------------------------------------[START]------------------------------------------------------------------------\n";
  std::cout << "[INFO] String matrix add computation...\n";

  srand(time(0));
  for (int row = 0; row < ROWS; ++row) {
    for (int col = 0; col < COLS; ++col) {
      int idx = row * COLS + col;
      h_A[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
      h_B[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;

      h_c_A[idx] = h_A[idx];
      h_c_B[idx] = h_B[idx];

    }
  }

  std::cout << "[INFO] Matrix correctly initialized on HOST memory\n";

  std::cout << "[INFO] Printing sub-matrix of the first " << VISUALIZATION_CONSTANT << " elemetns\n";
  std::cout << "\nMATRIX A:\n";
  printMatrix(h_A,ROWS,COLS);
  std::cout << "MATRIX B:\n";
  printMatrix(h_B,ROWS,COLS);


// ---------------------- Device memory initialisation ---------------------- //

  // Allocate global memory on the GPU.
  float *d_A, *d_B, *d_C;
  cudaError_t cudaRet = cudaSuccess;

  cudaRet = cudaMalloc((void **)&d_A, ROWS*COLS*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the first matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMalloc((void **)&d_B, ROWS*COLS*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the second matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMalloc((void **)&d_C, ROWS*COLS*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the third matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "[INFO] Memory correctly allocated on DEVICE\n";

  // Copy matrices from the host (CPU) to the device (GPU).
  cudaRet = cudaMemcpy(d_A, h_A, ROWS*COLS*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the first matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMemcpy(d_B, h_B, ROWS*COLS*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the second matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }


  std::cout << "[INFO] Data correctly copied on DEVICE\n";



// ------------------------ Calculations on the CPU ------------------------- //

  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  std::cout << "\n*************************************************************************[CPU]*************************************************************************\n";
  std::cout << "[INFO] CPU computation started\n";

  timer->reset();
  timer->start();
  CpuMatrixAdd(ROWS, COLS, h_c_A, h_c_B, h_c_C);
  timer->stop();

  std::cout << "[INFO] CPU computation finished\n";
  std::cout << "\n\tCPU time: " << timer->getTime() << " ms.\n\n";

  std::cout << "[INFO] Printing sub-matrix of the first " << VISUALIZATION_CONSTANT << " elemetns in the result matrix for the CPU computation\n";
  std::cout << "\nCPU COMPUTED MATRIX C:\n";
  printMatrix(h_c_C,ROWS,COLS);


  std::cout << "\n******************************************************************************************************************************************************\n";




// ------------------------ Calculations on the GPU ------------------------- //

  // Calculate the dimension of the grid of blocks (2D).
  const dim3 GRID_DIM ( 256, 256 );
  std::cout << "\n*************************************************************************[GPU]*************************************************************************\n";



  timer->reset();
  timer->start();
  GpuMatrixAdd<<<GRID_DIM, BLOCK_DIM>>>(ROWS, COLS, d_A, d_B, d_C);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  std::cout << "\n\tGPU time: " << timer->getTime() << " ms.\n\n";

  // Download the resulting matrix d_C from the device and store it in h_A.
  cudaRet = cudaMemcpy(h_g_C, d_C, ROWS*COLS*sizeof(float), cudaMemcpyDeviceToHost);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the DEVICE to the HOST for the third matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }


  std::cout << "[INFO] Data correctly returned on HOST\n";

  std::cout << "[INFO] Printing sub-matrix of the first " << VISUALIZATION_CONSTANT << " elemetns in the result matrix for the GPU computation\n";
  std::cout << "\nGPU COMPUTED MATRIX C:\n";
  printMatrix(h_g_C,ROWS,COLS);



  std::cout << "\n******************************************************************************************************************************************************\n";

  std::cout << "\n***********************************************************************[RESULTS]***********************************************************************\n";

  // Now let's check if the results are the same.
  float diff = 0.0f;
  for (int row = 0; row < ROWS; ++row) {
    for (int col = 0; col < COLS; ++col) {
      int idx = row * COLS + col;
      diff = std::max(diff, std::abs(h_A[idx] + h_B[idx] - h_c_C[idx]));
    }
  }
  std::cout << "\n\tMax diff CPU = " << diff << "\n\n";  // Should be (very close to) zero.

  diff = 0.0f;
  for (int row = 0; row < ROWS; ++row) {
    for (int col = 0; col < COLS; ++col) {
      int idx = row * COLS + col;
      diff = std::max(diff, std::abs(h_A[idx] + h_B[idx] - h_g_C[idx]));
    }
  }
  std::cout << "\n\tMax diff GPU = " << diff << "\n\n";  // Should be (very close to) zero.


  std::cout << "\n******************************************************************************************************************************************************\n";


// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  std::cout << "[INFO] Memory de-allocation correctly completed\n";
  std::cout << "\n-------------------------------------------------------------------------[END]-------------------------------------------------------------------------\n\n";

  return 0;
}
