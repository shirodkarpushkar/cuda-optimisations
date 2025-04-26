#include <cuda.h>
#include <iostream>
#include <vector>
#define SIZE 1024
#define THREADS_PER_BLOCK 32

__global__ void compute_kernel(int *data) {
  // Declare shared memory for each block
  __shared__ int sharedData[THREADS_PER_BLOCK];

  // Calculate thread index within the block
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize shared memory elements to the thread index (each block's shared
  // memory is independent)
  if (idx < SIZE) {
    sharedData[tid] = idx;
  }
  __syncthreads();

  /* now move data from shared memory to global memory */
  if (idx < SIZE) {
    data[idx] = sharedData[tid];
  }

  __syncthreads();
}

int main() {
  std::vector<int> data(SIZE);
  /* let say we handle this problem using  */
  int num_blocks = SIZE / THREADS_PER_BLOCK;
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);

  /* copy data to device */
  int *d_data;
  cudaMalloc((void **)&d_data, SIZE * sizeof(int));
  cudaMemcpy(d_data, data.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  /* launch kernel */
  compute_kernel<<<grid, block>>>(d_data);

  /* copy data from device to host */
  cudaMemcpy(data.data(), d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  /* free device memory */
  cudaFree(d_data);
  /* print data */
  printf("data[] = {");
  for (int i = 0; i < SIZE; i++) {
    printf("%d", data[i]);
    if (i < SIZE - 1) {
      printf(", ");
    }
  }
  printf("}\n");

  return 0;
}
