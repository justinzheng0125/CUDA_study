#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DS_timer.h"

// define sizes
// (1024 * 1024) ~ (1024 * 1024 * 128)
#define NUM_DATA   1000 * 100

#define BLOCK_X    500
#define BLOCK_Y    2
#define BLOCK_Z    1
#define BLOCK_SIZE BLOCK_X * BLOCK_Y * BLOCK_Z
#define GRID_X     ceil((float)NUM_DATA / BLOCK_X)
#define GRID_Y     1
#define GRID_Z     1

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
		//printf("tID:%d  %d + %d = %d\n", tID, a[tID], b[tID], c[tID]);
	}
}

typedef enum {
	GpuTotal,
	GpuComputation,
	GpuHost2Device,
	GpuDevice2Host,
	CpuTotal
}TIME_TYPE;


int main(void)
{
	//Set Timer
	DS_timer timer(5);
	timer.setTimerName(GpuTotal, "Device(GPU) Total");
	timer.setTimerName(GpuComputation, "Computation Device Kernel");
	timer.setTimerName(GpuHost2Device, "Data Transfer : Host -> Device");
	timer.setTimerName(GpuDevice2Host, "Data Transfer : Device -> Host");
	timer.setTimerName(CpuTotal, "\nHost(CPU) Total");
	timer.initTimers();

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

	timer.onTimer(CpuTotal);
	for (int i = 0; i < NUM_DATA; i++)
	{
		c[i] = a[i] + b[i];
	}
	timer.offTimer(CpuTotal);

	cudaMalloc(&dev_a, memSize);
	cudaMalloc(&dev_b, memSize);
	cudaMalloc(&dev_c, memSize);

	timer.onTimer(GpuTotal);

	timer.onTimer(GpuHost2Device);
	// Copy CPU data to GPU
	cudaMemcpy(dev_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, memSize, cudaMemcpyHostToDevice);
	timer.offTimer(GpuHost2Device);

	// GPU computing
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
	dim3 grid(GRID_X, GRID_Y, GRID_Z);

	timer.onTimer(GpuComputation);
	largeVecAdd << <grid, block >> > (dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	timer.offTimer(GpuComputation);

	timer.onTimer(GpuDevice2Host);
	// Copy result from GPU to CPU
	cudaMemcpy(host_c, dev_c, memSize, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < NUM_DATA; i++) printf("%d\n", host_c[i]);
	timer.offTimer(GpuDevice2Host);

	timer.offTimer(GpuTotal);
	timer.printTimer();

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

	delete[] a; delete[] b; delete[] c; delete[] host_c;

	return 0;
}