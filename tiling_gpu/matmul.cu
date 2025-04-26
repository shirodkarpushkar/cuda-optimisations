#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define TILE_DIM 2
#define N 8


/* naive CUDA kernel */
__global__ void matmul_kernel(int *A, int *B, int *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    int sum = 0;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

/* tiled CUDA kernel */
__global__ void tiled_matmul_kernel(int *A, int *B, int *C) {
  __shared__ int As[TILE_DIM][TILE_DIM + 1];
  __shared__ int Bs[TILE_DIM][TILE_DIM + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int sum = 0;

  for (int tile = 0; tile < N / TILE_DIM; ++tile) {
    /* load data into shared memory */
    As[ty][tx] = A[row * N + (tile * TILE_DIM + tx)];
    Bs[ty][tx] = B[(tile * TILE_DIM + ty) * N + col];
    __syncthreads();

    /* compute mat mul */
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  /* write result to global memory */
  C[row * N + col] = sum;
}

__global__ void tiledpadded_matmul_kernel(int *A, int *B, int *C) {
  // Add +1 to avoid bank conflicts
  __shared__ int As[TILE_DIM][TILE_DIM + 1];
  __shared__ int Bs[TILE_DIM][TILE_DIM ];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_DIM + ty;
  int col = blockIdx.x * TILE_DIM + tx;

  int sum = 0;

  for (int tile = 0; tile < N / TILE_DIM; ++tile) {
    /* load data into shared memory */
    As[ty][tx] = A[row * N + (tile * TILE_DIM + tx)];
    Bs[ty][tx] = B[(tile * TILE_DIM + ty) * N + col];
    __syncthreads();

    /* compute mat mul */
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  /* write result to global memory */
  C[row * N + col] = sum;

  
}
/* generate random matrix of type Matrix */
int *generate_random_matrix() {
  int rows = N;
  int cols = N;
  int *matrix = new int [rows * cols];
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i * cols + j] = rand() % 10; // Random number between 0 and 9
    }
  }
  return matrix;
}

int *matmul(int *A, int *B) {
  int rowsA = N, colsA = N;
  int rowsB = N, colsB = N;

  if (colsA != rowsB) {
    std::cerr << "Matrix dimensions do not match for multiplication."
              << std::endl;
    return nullptr;
  }

  int *C = new int[rowsA * colsB];
  for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsB; ++j) {
      C[i * colsB + j] = 0;
      int sum = 0;
      for (int k = 0; k < colsA; ++k) {
        sum += A[i * colsA + k] * B[k * colsB + j];
      }
      C[i * colsB + j] = sum;
    }
  }

  return C;
}

void print_matrix(int *matrix) {
  int rows = N;
  int cols = N;

  printf("[\n");
  for (int i = 0; i < rows; ++i) {
    printf("  [");
    for (int j = 0; j < cols; ++j) {
      printf("%3d", matrix[i * cols + j]);
      if (j < cols - 1)
        printf(", ");
    }
    printf("]");
    if (i < rows - 1)
      printf(",\n");
    else
      printf("\n");
  }
  printf("]\n");
}

void free_matrix(int *matrix) {
  delete[] matrix;
}


int main() {
  /* cuda timers */
  cudaEvent_t start, stop;
  float elapsedTime;
  float elapsedTimeTiled;
  float elapsedTimeTiledPadded;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Generate random matrices
  int *A = generate_random_matrix();
  int *B = generate_random_matrix();

  // Allocate device memory
  int *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, N * N * sizeof(int));
  cudaMalloc((void**)&d_B, N * N * sizeof(int));
  cudaMalloc((void**)&d_C, N * N * sizeof(int));
  // Copy matrices from host to device
  cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

  /* grid and block dimension for naive kernel */
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
  // Start timer
  cudaEventRecord(start, 0);
  // Launch naive kernel
  matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);
  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  // Copy result back to host
  int *C = new int[N * N];
  cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  // Print result
  printf("Result of naive kernel GPU:\n");
  print_matrix(C);

  // Free device memory
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free_matrix(C);


  /* tiled calculation */
  dim3 blockDimTiled(TILE_DIM, TILE_DIM);
  dim3 gridDimTiled(N / TILE_DIM, N / TILE_DIM);
  /* create timer */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Start timer
  cudaEventRecord(start, 0);
  // Launch tiled kernel
  tiled_matmul_kernel<<<gridDimTiled, blockDimTiled>>>(d_A, d_B, d_C);
  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTimeTiled, start, stop);
  /* allocate mem for C */
  C = new int[N * N];
  // Copy result back to host
  cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  // Print result
  printf("Result of tiled kernel GPU:\n");
  print_matrix(C);

  // Free device memory
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free_matrix(C);

  /* bank conflict free */
  dim3 blockDimTiledPadded(TILE_DIM, TILE_DIM);
  dim3 gridDimTiledPadded(N / TILE_DIM, N / TILE_DIM);
  /* create timer */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Start timer
  cudaEventRecord(start, 0);
  // Launch tiled kernel
  tiledpadded_matmul_kernel<<<gridDimTiledPadded, blockDimTiledPadded>>>(d_A, d_B, d_C);
  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTimeTiledPadded, start, stop);
  /* allocate mem for C */
  C = new int[N * N];
  // Copy result back to host
  cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  // Print result
  printf("Result of tiled padded kernel GPU:\n");
  print_matrix(C);

  /* print execution time and speedup  */
  printf("Execution time of naive kernel: %f ms\n", elapsedTime);
  printf("Execution time of tiled kernel: %f ms\n", elapsedTimeTiled);
  printf("Execution time of tiled padded kernel: %f ms\n",
         elapsedTimeTiledPadded);
  
  printf("Speedup of tiled kernel: %f\n", elapsedTime / elapsedTimeTiled);
  printf("Speedup of tiled padded kernel: %f\n",
         elapsedTime / elapsedTimeTiledPadded);
  printf("Speedup of tiled padded kernel with respect to tiled kernel: %f\n",
         elapsedTimeTiled / elapsedTimeTiledPadded);
  
  




  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  free_matrix(C);
  free_matrix(A);
  free_matrix(B);


  
  
  
  return 0;
}