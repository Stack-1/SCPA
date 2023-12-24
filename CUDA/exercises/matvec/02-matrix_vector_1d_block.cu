//
// Author: Salvatore Filippone salvatore.filippone@cranfield.ac.uk
//

// Computes matrix-vector product. Matrix A is in row-major order
// i.e. A[i, j] is stored in i * ncols + j element of the vector.
//

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

// Simple 1-D thread block
// What is the best block dimension? Smaller? Larger?
#define BD 32
#define VISUALIZATION_CONSTANT 16


const dim3 BLOCK_DIM(BD);

// Simple CPU implementation of matrix addition.
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

// GPU implementation of matrix_vector product using a block of threads for
// each row.
__global__ void gpuMatrixVector(int rows, int cols, const float* A,
				const float* x, float* y) {
  // TODO This will require:
  // 1. Use of shared memory
  // 2. Implementation of a reduction operation
  // Pay attention to the indexing: which row is the current block supposed
  // to act upon?


}


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


  if (argc < 3) {
    fprintf(stderr,"Usage: %s  rows cols\n",argv[0]);
  }
  int nrows=atoi(argv[1]);
  int ncols=atoi(argv[2]);

  std::cout << "\n------------------------------------------------------------------------[START]------------------------------------------------------------------------\n";
  std::cout << "[INFO] String matrix add computation...\n";

  // ----------------------- Host memory initialisation ----------------------- //

  float* h_A = new float[nrows * ncols];
  float* h_x = new float[ncols];
  float* h_y = new float[nrows];
  float* h_y_d = new float[nrows];

  srand(123456);
  for (int row = 0; row < nrows; ++row) {
    for (int col = 0; col < ncols; ++col) {
      int idx = row * ncols + col;
      h_A[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
    }
    h_y[row] = 0.0f;
    h_y_d[row] = 0.0f;
  }
  for (int col = 0; col < ncols; ++col) {
    h_x[col] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
  }

  std::cout << "[INFO] Memory correctly initialized on HOST\n";

  std::cout << "[INFO] Printing sub-matrix of the first " << VISUALIZATION_CONSTANT << " elemetns\n";
  std::cout << "\nMATRIX A:\n";
  printMatrix(h_A,VISUALIZATION_CONSTANT,VISUALIZATION_CONSTANT);
  std::cout << "VECTOR X:\n";
  printVector(h_x,VISUALIZATION_CONSTANT);
  std::cout << "VECTOR Y:\n";
  printVector(h_y,VISUALIZATION_CONSTANT);
  std::cout << "VECTOR Y_D:\n";
  printVector(h_y_d,VISUALIZATION_CONSTANT);

  std::cout << "\n\tTest case: " << nrows  << " x " << ncols << "\n\n";
// ---------------------- Device memory initialisation ---------------------- //

  float *d_A, *d_x, *d_y;

  checkCudaErrors(cudaMalloc((void**) &d_A, nrows * ncols * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_x, ncols * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_y, nrows * sizeof(float)));

  // Copy matrices from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(d_A, h_A, nrows * ncols * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x, h_x,  ncols * sizeof(float), cudaMemcpyHostToDevice));


  std::cout << "[INFO] Memory correctly initialized on DEVICE\n";

  // ------------------------ Calculations on the CPU ------------------------- //
  float flopcnt=2.e-6*nrows*ncols;

  std::cout << "\n*************************************************************************[CPU]*************************************************************************\n";
  std::cout << "[INFO] CPU computation started\n";


  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);
  std::cout << "[INFO] CPU computation finished\n";

  timer->reset();
  timer->start();
  CpuMatrixVector(nrows, ncols, h_A, h_x, h_y);
  timer->stop();
  float cpuflops=flopcnt/ timer->getTime();
  std::cout << "\n\tCPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << "\n\n";


  std::cout << "******************************************************************************************************************************************************\n";


// ------------------------ Calculations on the GPU ------------------------- //

  // TODO Calculate the dimension of the grid of blocks (1D) needed to cover all
  // entries in the matrix and output vector
  const dim3 GRID_DIM(1);

  std::cout << "\n*************************************************************************[GPU]*************************************************************************\n";
  std::cout << "[INFO] GPU computation started\n";

  timer->reset();
  timer->start();
  gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>>(nrows, ncols, d_A, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "[INFO] GPU computation finished\n";

  timer->stop();
  float gpuflops=flopcnt/ timer->getTime();
  std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

  // Download the resulting vector d_y from the device and store it in h_y_d.
  checkCudaErrors(cudaMemcpy(h_y_d, d_y, nrows*sizeof(float),cudaMemcpyDeviceToHost));
  std::cout << "[INFO] Data correctly returned on HOST\n";

  std::cout << "[INFO] Printing result vector first " << VISUALIZATION_CONSTANT << " elemetns for the GPU computation\n";
  std::cout << "\nGPU COMPUTED VECTOR Y:\n";
  printVector(h_y_d,VISUALIZATION_CONSTANT);

  std::cout << "******************************************************************************************************************************************************\n";

  std::cout << "\n***********************************************************************[RESULTS]***********************************************************************\n";


  // Now let's check if the results are the same.
  float reldiff = 0.0f;
  float diff = 0.0f;

  for (int row = 0; row < nrows; ++row) {
    float maxabs = std::max(std::abs(h_y[row]),std::abs(h_y_d[row]));
    if (maxabs == 0.0) maxabs=1.0;
    reldiff = std::max(reldiff, std::abs(h_y[row] - h_y_d[row])/maxabs);
    diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
  }
  std::cout << "\n\tMax diff = " << diff << "  Max rel diff = " << reldiff << "\n\n";
  // Rel diff should be as close as possible to unit roundoff; float
  // corresponds to IEEE single precision, so unit roundoff is
  // 1.19e-07
  //

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
