#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>

#define TILE_DIM 2

/* generate random matrix of type Matrix */
int **generate_random_matrix(int N) {
  int rows = N;
  int cols = N;
  int **matrix = new int *[rows];
  for (int i = 0; i < rows; ++i) {
    matrix[i] = new int[cols];
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = rand() % 10; // Random number between 0 and 9
    }
  }
  return matrix;
}

void print_matrix(int **matrix, int N) {
  int rows = N;
  int cols = N;

  printf("[\n");
  for (int i = 0; i < rows; ++i) {
    printf("  [");
    for (int j = 0; j < cols; ++j) {
      printf("%3d", matrix[i][j]);
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

void free_matrix(int **matrix, int rows) {
  for (int i = 0; i < rows; ++i) {
    delete[] matrix[i];
  }
  delete[] matrix;
}

int **matmul(int **A, int **B, int N) {
  int rowsA = N, colsA = N;
  int rowsB = N, colsB = N;

  if (colsA != rowsB) {
    std::cerr << "Matrix dimensions do not match for multiplication."
              << std::endl;
    return nullptr;
  }

  int **C = new int *[rowsA];
  for (int i = 0; i < rowsA; ++i) {
    /* create a new row for c */
    C[i] = new int[colsB];
    for (int j = 0; j < colsB; ++j) {
      C[i][j] = 0;
      int sum = 0;

      for (int k = 0; k < colsA; ++k) {
        sum += A[i][k] * B[k][j];
      }

      C[i][j] = sum;
    }
  }
  return C;
}

/* tiled matrix multiplication */
int** tiled_matmul(int **A, int **B, int N) {
  int rowsA = N, colsA = N;
  int rowsB = N, colsB = N;

  if (colsA != rowsB) {
    std::cerr << "Matrix dimensions do not match for multiplication."
              << std::endl;
    return nullptr;
  }

  /* initialise matrix C */
  int **C = new int *[rowsA];
  for (int i = 0; i < rowsA; ++i) {
    C[i] = new int[colsB];
    for (int j = 0; j < colsB; ++j) {
      C[i][j] = 0;
    }
  }

  for (int i = 0; i < rowsA; i += TILE_DIM) {
    for (int j = 0; j < colsB; j += TILE_DIM) {

      /* tiled matrix multiply */
      for(int ii = i; ii < i + TILE_DIM ; ++ii) {
        for(int jj = j; jj < j + TILE_DIM ; ++jj) {
          int sum = 0;

          for (int k = 0; k < colsA; k += TILE_DIM) {
            for (int kk = k; kk < k + TILE_DIM; ++kk) {
              sum += A[ii][kk] * B[kk][jj];
            }
          }

          C[ii][jj] = sum;
        }
      }
      
    }
  }

  return C;
}

int main() {
  // Define matrix dimensions
  int N = 8;
  /* timers  for recording matrix mul precision ms */


  // Generate random matrices
  int **A = generate_random_matrix(N);
  int **B = generate_random_matrix(N);

  /* start time */
  auto start = std::chrono::high_resolution_clock::now();
  int **C = matmul(A, B, N);
  auto end = std::chrono::high_resolution_clock::now();
  /* precision in ms */
  std::chrono::duration<double, std::milli> matmul_time = end - start;

  // Print matrices
  std::cout << "Matrix A:" << std::endl;
  print_matrix(A, N);
  std::cout << "Matrix B:" << std::endl;
  print_matrix(B, N);

  std::cout << "Matrix C (A * B):" << std::endl;
  print_matrix(C, N);

  /* tiled matmul*/
  start = std::chrono::high_resolution_clock::now();
  int **C_tiled = tiled_matmul(A, B, N);
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> tiled_matmul_time = end - start;

  std::cout << "Matrix C_tiled (A * B):" << std::endl;
  print_matrix(C_tiled, N);

  // Print execution times
  printf("Matmul Time: %.6f ms\n", matmul_time.count());
  printf("Tiled Matmul Time: %.6f ms\n", tiled_matmul_time.count());  

  // Free allocated memory
  free_matrix(A, N);
  free_matrix(B, N);

  return 0;
}