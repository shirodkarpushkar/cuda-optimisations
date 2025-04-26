# makefile
NVCC := nvcc
CFLAGS := -O2

all: matmul 

matmul: matmul.cu
	$(NVCC) $(CFLAGS) matmul.cu -o matmul

run: clean matmul
	./matmul

clean:
	rm -rf matmul 

