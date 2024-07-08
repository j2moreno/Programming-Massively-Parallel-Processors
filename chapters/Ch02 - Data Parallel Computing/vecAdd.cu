#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


inline void checkError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(err);
    }
}

#define CHECK_ERROR(call) checkError((call), __FILE__, __LINE__)

// compute vector sum C = A+B
// each thread performs one pair-wise addition
__global__ // executed on the device, only callable from the host
void vecAddKernel(float *A, float *B, float *C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

int main(void) {

	// create and host vectors
	float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;

	int n = 2000000;
    int size = n*sizeof(float);

	// allocate memory for host vectors
	A_h = (float*)malloc(sizeof(float)*n);
	B_h = (float*)malloc(sizeof(float)*n);
	C_h = (float*)malloc(sizeof(float)*n);
	
	// fill A and B host vectors with random values
	for (int i = 0; i < n; i++) {
		A_h[i] = ((((float)rand() / (float)(RAND_MAX)) * 100));
		B_h[i] = ((((float)rand() / (float)(RAND_MAX)) * 100));
	}

	//1. Allocate global memory on the device for A, B and C
	CHECK_ERROR(cudaMalloc((void**)&A_d, size));
	CHECK_ERROR(cudaMalloc((void**)&B_d, size));
	CHECK_ERROR(cudaMalloc((void**)&C_d, size));

	// copy A and B to device memory
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	//2. Kernel launch code - to have the device to perform the actual vector addition
	// Kernel invocation with 256 threads
	dim3 dimGrid(ceil(n / 256.0),1,1);
	dim3 dimBlock((256.0),1,1);
	vecAddKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);

	//3. copy C from the device memory
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	// Free device vectors
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	// Free host memory
	free(A_h);
	free(B_h);
	free(C_h);

	return 0;
}