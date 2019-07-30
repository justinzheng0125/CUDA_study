#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// define sizes
#define BLOCK_SIZE 1024
#define GRID_X     4
#define GRID_Y     3
#define GRID_Z     2
#define NUM_DATA BLOCK_SIZE * GRID_X * GRID_Y * GRID_Z

// Get Thead ID definitions
/********************************************************************
	int block_1d = threadIdx.x;
	int block_2d = blockDim.x * threadIdx.y + block_1d;
	int block_3d = blockDim.x * blockDim.y * threadIdx.z + block_2d;
	int grid_1d = BLOCK_SIZE * blockIdx.x + block_3d;
	int grid_2d = BLOCK_SIZE * gridDim.x * blockIdx.y + grid_1d;
	int grid_3d = BLOCK_SIZE * gridDim.x * gridDim.y * blockIdx.z + grid_2d;
	***************************************************************/

__global__ void largeVecAdd(int* a, int* b, int* c)
{

	int block_1d = threadIdx.x;
	int block_2d = blockDim.x * threadIdx.y + block_1d;
	int block_3d = blockDim.x * blockDim.y * threadIdx.z + block_2d;
	int grid_1d = BLOCK_SIZE * blockIdx.x + block_3d;
	int grid_2d = BLOCK_SIZE * gridDim.x * blockIdx.y + grid_1d;
	int grid_3d = BLOCK_SIZE * gridDim.x * gridDim.y * blockIdx.z + grid_2d;

	int tID = grid_3d;
	if (tID < NUM_DATA)
	{
		c[tID] = a[tID] + b[tID];
		printf("tID:%d  %d + %d = %d\n", tID, a[tID], b[tID], c[tID]);
	}
}


int main(void)
{
	int* a, * b, * c, * host_c;
	int* dev_a, * dev_b, * dev_c;

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, %d bytes memSize\n", NUM_DATA, memSize);

	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	host_c = new int[NUM_DATA]; memset(host_c, 0, memSize);

	for (int i = 0; i < NUM_DATA; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	for (int i = 0; i < NUM_DATA; i++)
	{
		c[i] = a[i] + b[i];
	}

	cudaMalloc(&dev_a, memSize);
	cudaMalloc(&dev_b, memSize);
	cudaMalloc(&dev_c, memSize);

	// Copy CPU data to GPU
	cudaMemcpy(dev_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, memSize, cudaMemcpyHostToDevice);

	// GPU computing
	dim3 block(256, 2,2);
	dim3 grid(GRID_X, GRID_Y, GRID_Z);
	largeVecAdd << <grid, block >> > (dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();

	// Copy result from GPU to CPU
	cudaMemcpy(host_c, dev_c, memSize, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < NUM_DATA; i++) printf("%d\n", host_c[i]);

	bool res = true;
	for (int i = 0; i < NUM_DATA; i++)
	{
		if (host_c[i] != c[i])
		{
			printf("[%d] the result %d != %d\n", i, c[i], host_c[i]);
			res = false;
		}
	}
	if (res) printf("\nGREAT! GPU works well\n");
	cudaFree(dev_a); 
	cudaFree(dev_b); 
	cudaFree(dev_c);
	delete[] a; delete[] b; delete[] c;

	return 0;
}