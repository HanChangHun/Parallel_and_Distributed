CC=g++
NVCC=nvcc

OBJS = matrixmul.o matrixmul_gold.o

# Write inc directory of Target Platform
IFLAG = -I /usr/local/cuda/samples/common/inc
CUDAFLAG = 

all: matrix_cuda

matrix_cuda: $(OBJS)
	$(NVCC) -o $@ $^

matrixmul.o: matrixmul.cu assist.h matrixmul.h
	$(NVCC) -c -o $@ $< $(IFLAG) $(CUDAFLAG) 

matrixmul_gold.o: matrixmul_gold.cpp matrixmul.h
	$(CC) -c -o $@ $<

clean:
	rm $(OBJS) matrix_cuda
