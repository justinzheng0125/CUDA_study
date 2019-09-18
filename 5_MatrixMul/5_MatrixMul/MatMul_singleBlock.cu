#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW_SIZE (32)
#define COL_SIZE (32)
#define K_SIZE (128)

#define WORK_LOAD (1024)
#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)

float A[ROW_SIZE][K_SIZE];
float B[K_SIZE][COL_SIZE];
float hostC[ROW_SIZE][COL_SIZE];
float deviceC[ROW_SIZE][COL_SIZE];

#define memSetZero(P, type, size) memset(P, 0, sizeof(type)*size)
//#define dMemAlloc(P, type, size) cudaMemAlloc(&P, sizeof(type)*size)

__global__ void mult_kernel(float* _A, float* _B, float* _C) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDim.x + col;
	if (row >= blockDim.y || col >= blockDim.x) {
		return;
	}
	for (int k = 0; k < K_SIZE; k++) {
		for (int i = 0; i < WORK_LOAD; i++) {
			_C[index] += _A[row * K_SIZE + k] * _B[k * COL_SIZE + col];
		}
	}
	
}

void main(void) {
	float* dA, * dB, * dC;
	dA = dB = dC = NULL;

	memSetZero(A, float, MAT_SIZE_A);
	memSetZero(B, float, MAT_SIZE_B);
	memSetZero(hostC, float, MAT_SIZE_C);
	memSetZero(deviceC, float, MAT_SIZE_C);

	cudaMalloc(&dA, sizeof(float) * MAT_SIZE_A);
	cudaMalloc(&dB, sizeof(float) * MAT_SIZE_B);
	cudaMalloc(&dC, sizeof(float) * MAT_SIZE_C);

	// Generate Data
	for (int i = 0; i < ROW_SIZE; i++) {
		for (int j = 0; j < K_SIZE; j++) {
			A[i][j] = rand() % 100;
		}
	}
	for (int i = 0; i < K_SIZE; i++) {
		for (int j = 0; j < COL_SIZE; j++) {
			B[i][j] = rand() % 100;
		}
	}

	// Calculate Data - CPU
	for (int r = 0; r < ROW_SIZE; r++) {
		for (int c = 0; c < COL_SIZE; c++) {
			for (int k = 0; k < K_SIZE; k++) {
				for (int i = 0; i < WORK_LOAD; i++) {
					hostC[r][c] += A[r][k] * B[k][c];
				}
			}
		}
	}

	// copy Data to GPU
	cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);

	dim3 blockDim(COL_SIZE, ROW_SIZE);
	mult_kernel << <1, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);

	bool isCorrect = true;
	for (int i = 0; i < ROW_SIZE; i++) {
		for (int j = 0; j < COL_SIZE; j++) {
			if (hostC[i][j] != deviceC[i][j]) {
				isCorrect = false;
				break;
			}
		}
	}
	if (isCorrect) printf("working Correct!!\n");
	else printf("result is WRONG!!\n");
}

