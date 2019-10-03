
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "helper_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIZE_M (512 * 2)
#define SIZE_N (512 * 4)
#define SIZE_K (512 * 2)
#define BLOCK_SIZE 16
#define ID2INDX(_row, _col, _width) (((_row)*(_width))+(_col))

void generateValues(float** p, int size) {
	for (int i = 0; i < size; i++) {
		*((*p) + i) = rand() % 10 + rand() % 100 / 100.0;
	}
}

__global__ void MatMul(float* matA, float* matB, float* matC, int m, int n, int k) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	if (row >= m || col >= n) return;
	float val = 0;
	for (int i = 0; i < k; i++) {
		val += matA[ID2INDX(row, i, k)] * matB[ID2INDX(i, col, n)];
	}
	matC[ID2INDX(row, col, n)] = val;
}

void compareMatrix(float* a, float* b, int size) {
	bool isPass = true;
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			isPass = false;
			//break;
			//printf("%d != %d\n", a[i], b[i]);
		}
	}
	if (isPass) printf("CPU and GPU result are same.\n");
	else printf("The results are not matched!!!!\n");
}

int main(int argc, char* argv[]) {

	// Matrix size
	int m, n, k;
	m = SIZE_M; n = SIZE_N; k = SIZE_K;
	printf("Matrix A = [%d by %d]\nMatrix B = [%d by %d]\nMatrix C = [%d by %d]\n",
		m, k, k, n, m, n);

	int sizeA, sizeB, sizeC;
	sizeA = m * k; sizeB = n * k; sizeC = m * n;

	// initialize matrix A and B
	float* A = NULL, * B = NULL;
	A = new float[sizeA];
	B = new float[sizeB];
	memset(A, 0, sizeof(float) * sizeA);
	memset(B, 0, sizeof(float) * sizeB);

	// initialize matrix cpuC and gpuC
	float* cpuC = NULL, * gpuC = NULL;
	cpuC = new float[sizeC];
	gpuC = new float[sizeC];
	memset(cpuC, 0, sizeof(float) * sizeC);
	memset(gpuC, 0, sizeof(float) * sizeC);

	// input values matrix A and B
	generateValues(&A, sizeA);
	generateValues(&B, sizeB);

	printf("CPU val: %f %f\n", A[0], B[0]);

	// CPU running
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int index = ID2INDX(row, col, n);
			for (int i = 0; i < k; i++) {
				cpuC[index] += A[ID2INDX(row, i, k)] * B[ID2INDX(i, col, n)];
			}
		}
	}
	printf("CPU result %f\n", cpuC[0]);
	printf("CPU is finished\n");

	// GPU
	printf("GPU start\n");
	float* dA, * dB, * dC;
	cudaMalloc(&dA, sizeA * sizeof(float));
	cudaMalloc(&dB, sizeB * sizeof(float));
	cudaMalloc(&dC, sizeC * sizeof(float));
	cudaMemset(&dA, 0, sizeA * sizeof(float));
	cudaMemset(&dB, 0, sizeB * sizeof(float));
	cudaMemset(&dC, 0, sizeC * sizeof(float));

	cudaMemcpy(dA, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
	printf("finished copy data from host to device\n");

	dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	MatMul << <gridDim, blockDim >> > (dA, dB, dC, m, n, k);
	cudaDeviceSynchronize();

	cudaMemcpy(gpuC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
	printf("GPU result %f\n", gpuC[0]);
	compareMatrix(cpuC, gpuC, sizeC);

	// 1. GPU global memory

	return 0;
}

*/