#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
//#include <stblib.h>

// define size of the vector
#define NUM_DATA 512 //10240

__global__ void vecAdd(int* a, int* b, int* c)
{
	int tID = threadIdx.x;
	c[tID] = a[tID] + b[tID];
	//printf("%d + %d = %d\n", a[tID], b[tID], c[tID]);
}

__global__ void printing(int* a)
{
	int tID = threadIdx.x;
	printf("%d\n", a[tID]);
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
		a[i] = /*rand() % 10*/ i;
		b[i] = /*rand() % 10*/ i;
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
	//printing << <1, NUM_DATA >> > (dev_a);
	cudaMemcpy(dev_b, b, memSize, cudaMemcpyHostToDevice);

	// GPU computing
	vecAdd << <1, NUM_DATA >> > (dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	//printing << <1, NUM_DATA >> > (dev_c);

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
	if (res) printf("GPU works well\n");
	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
	delete[] a; delete[] b; delete[] c;

	return 0;
}