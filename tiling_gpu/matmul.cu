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
  __shared__ int As[TILE_DIM][TILE_DIM];
  __shared__ int Bs[TILE_DIM][TILE_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int sum = 0;

  for (int t = 0; t < N / TILE_DIM; ++t) {
    /* load data into shared memory */
    As[ty][tx] = A[row * N + (t * TILE_DIM + tx)];
    Bs[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  C[row * N + col] = sum;
}

/* generate random matrix of type Matrix */
int *generate_random_matrix(int N) {
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

void print_matrix(int *matrix, int N) {
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

void free_matrix(int *matrix, int rows) {
  delete[] matrix;
}


int main() {
  // Define matrix dimensions
  int N = 8;
  /* cuda timers */
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Generate random matrices
  int *A = generate_random_matrix(N);
  int *B = generate_random_matrix(N);

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
  printf("Naive kernel execution time: %f ms\n", elapsedTime);
  // Copy result back to host
  int *C = new int[N * N];
  cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  // Print result
  printf("Result of naive kernel:\n");
  print_matrix(C, N);
  
  // Free allocated memory
  free_matrix(A, N);
  free_matrix(B, N);
  free_matrix(C, N);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}