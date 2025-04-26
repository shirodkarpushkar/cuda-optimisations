NVCC := nvcc
CFLAGS := -O2

all: sharedMem 

sharedMem: sharedMem.cu
	$(NVCC) $(CFLAGS) sharedMem.cu -o sharedMem

clean:
	rm -f sharedMem 

run: clean sharedMem
	./sharedMem