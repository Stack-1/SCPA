//
// Author: Salvatore Filippone salvatore.filippone@cranfield.ac.uk
//

// Computes matrix-vector product. Matrix A is in row-major order
// i.e. A[i, j] is stored in i * COLS + j element of the vector.
//

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#define VISUALIZATION_CONSTANT 16 // Constat used for disply elements in the vector for debugging

#define MATRIX_SIZE 4096

// Ottimizzazione considerando la quantità dei processori per la grandezza dei blocchi
// Cuda Malloc con potenze diverse da due, può generare un disalineamento in memoria (andrebbe utilizzato cuda malloc 2d e cua memcpy 2d)

// Matrix dimensions.
const int ROWS = MATRIX_SIZE;
const int COLS = MATRIX_SIZE;



// TODO(later) Play a bit with the block size.
// Note: For meaningful time measurements you need sufficiently large matrix.
dim3 GRID_DIM(4, 1, 1);
// Calculate the dimension of the grid of blocks (2D).
dim3 BLOCK_DIM(1024, 1, 1); // (col, row, z)


// Simple CPU implementation of matrix-vector product.
void CpuMatrixVector(int rows, int cols, const float* A, const float* x, float* y) {
  for (int row = 0; row < rows; ++row) {
    float t=0.0;
    for (int col = 0; col < cols; ++col) {
      int idx = row * cols + col;
      t += A[idx] * x[col];
    }
    y[row] = t;
  }
}

// GPU implementation of matrix_vector product: see if you can use
// one thread per row. You'll need to get the addressing right!
__global__ void gpuMatrixVector(int rows, int cols, const float* A, const float* x, float* y) {
  // With this computation, max diff = 2/3 is it normal?
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = 0;

  if(row < rows) {
    float t=0.0;
    for (; col < cols; ++col) {
      t += A[col + row*cols] * x[col];
    }
    y[row] = t;
  }
}



// Function to display vector values weel formatted
void printVector(float *h_y, int n){

  std::cout << "\t";

  for(int i = 0;i<n;++i){
    std::cout << h_y[i] << " ";
  }

  std::cout << "\n\n";
}

void printMatrix(const float *h_A,int ROWS,int COLS){
  std::cout << "______________________________________________________________________________________________________________________________________________________\n";
  for (int row = 0; row < ROWS; ++row) {
    std::cout << "| ";
    for (int col = 0; col < COLS; ++col) {
      int idx = row * COLS + col;
      std::cout << h_A[idx] << " ";
    }
    std::cout << "\t\t|\n";
  }
  std::cout << "______________________________________________________________________________________________________________________________________________________\n\n";
}


int main(int argc, char** argv) {

// ----------------------- Host memory initialisation ----------------------- //

  float* h_A = new float[ROWS * COLS];
  float* h_x = new float[COLS];
  float* h_y = new float[ROWS];
  float* h_y_d = new float[ROWS];



  std::cout << "\n------------------------------------------------------------------------[START]------------------------------------------------------------------------\n";
  std::cout << "[INFO] String matrix add computation...\n";


  srand(time(0));
  for (int row = 0; row < ROWS; ++row) {
    for (int col = 0; col < COLS; ++col) {
      int idx = row * COLS + col;
      h_A[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
    }
    h_x[row] = 100.0f * static_cast<float>(rand()) / RAND_MAX;

    h_y[row] = 0;
    h_y_d[row] = 0;
  }

  std::cout << "[INFO] Correctly initialized on HOST memory\n";

  std::cout << "[INFO] Printing sub-matrix of the first " << VISUALIZATION_CONSTANT << " elemetns\n";
  std::cout << "\nMATRIX A:\n";
  printMatrix(h_A,VISUALIZATION_CONSTANT,VISUALIZATION_CONSTANT);
  std::cout << "VECTOR X:\n";
  printVector(h_x,VISUALIZATION_CONSTANT);
  std::cout << "VECTOR Y:\n";
  printVector(h_y,VISUALIZATION_CONSTANT);
  std::cout << "VECTOR Y_D:\n";
  printVector(h_y_d,VISUALIZATION_CONSTANT);


// ---------------------- Device memory initialisation ---------------------- //

  // Allocate global memory on the GPU.
  float *d_A, *d_x, *d_y;
  cudaError_t cudaRet = cudaSuccess;


  cudaRet = cudaMalloc((void **)&d_A, ROWS*COLS*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMalloc((void **)&d_x, COLS*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the first vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMalloc((void **)&d_y, ROWS*sizeof(float));

  if(cudaRet == cudaErrorMemoryAllocation){
    std::cout << "[ERROR] Error occured allocating memory on the GPU for the second vector, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "[INFO] Memory correctly allocated on DEVICE\n";


  // Copy matrices from the host (CPU) to the device (GPU).
  cudaRet = cudaMemcpy(d_A, h_A, ROWS*COLS*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  cudaRet = cudaMemcpy(d_x, h_x, COLS*sizeof(float), cudaMemcpyHostToDevice);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the HOST to the GPU for the first vector, error return code is: " << cudaRet << "\n";
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
  CpuMatrixVector(ROWS, COLS, h_A, h_x, h_y);
  timer->stop();

  std::cout << "[INFO] CPU computation finished\n";
  std::cout << "\n\tCPU time: " << timer->getTime() << " ms.\n\n";

  std::cout << "[INFO] Printing result vector first " << VISUALIZATION_CONSTANT << " elemetns for the CPU computation\n";
  std::cout << "\nCPU COMPUTED VECTOR Y:\n";
  printVector(h_y,VISUALIZATION_CONSTANT);

  std::cout << "******************************************************************************************************************************************************\n";


// ------------------------ Calculations on the GPU ------------------------- //


  std::cout << "\n*************************************************************************[GPU]*************************************************************************\n";


  std::cout << "[INFO] GPU computation started\n";
  timer->reset();
  timer->start();
  gpuMatrixVector<<<GRID_DIM, BLOCK_DIM>>>(ROWS, COLS, d_A, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();
  std::cout << "[INFO] GPU computation finished\n";


  std::cout << "\n\tGPU time: " << timer->getTime() << " ms.\n\n";

  // Download the resulting vector d_y from the device and store it in h_y_d.
  cudaRet = cudaMemcpy(h_y_d, d_y, ROWS*sizeof(float), cudaMemcpyDeviceToHost);

  if(cudaRet == cudaErrorInvalidValue || cudaRet == cudaErrorInvalidDevicePointer || cudaRet == cudaErrorInvalidMemcpyDirection){
    std::cout << "[ERROR] Error occured during the copy from the DEVICE to the HOST for the third matrix, error return code is: " << cudaRet << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "[INFO] Data correctly returned on HOST\n";

  std::cout << "[INFO] Printing result vector first " << VISUALIZATION_CONSTANT << " elemetns for the GPU computation\n";
  std::cout << "\nGPU COMPUTED VECTOR Y:\n";
  printVector(h_y_d,VISUALIZATION_CONSTANT);

  std::cout << "******************************************************************************************************************************************************\n";

  std::cout << "\n***********************************************************************[RESULTS]***********************************************************************\n";

  // Now let's check if the results are the same.
  float diff = 0.0f;
  for (int row = 0; row < ROWS; ++row) {
    diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
  }
  std::cout << "\n\tMax diff = " << diff << "\n\n";  // Should be (very close to) zero.

  std::cout << "\n******************************************************************************************************************************************************\n";

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

  delete[] h_A;
  delete[] h_x;
  delete[] h_y;
  delete[] h_y_d;


  std::cout << "[INFO] Memory de-allocation correctly completed\n";
  std::cout << "\n-------------------------------------------------------------------------[END]-------------------------------------------------------------------------\n\n";

  return 0;
}

