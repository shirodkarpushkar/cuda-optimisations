# makefile
NVCC := g++
CFLAGS := -O2

all: matmul 

matmul: matmul.cc
	$(NVCC) $(CFLAGS) matmul.cc -o matmul

run: clean matmul
	./matmul

clean:
	rm -rf matmul 

