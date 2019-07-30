#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void checkIdx(void)
{
	printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) BlockSize: (%d, %d, %d) GridSize: (%d, %d, %d)\n"
			, threadIdx.x, threadIdx.y, threadIdx.z
			, blockIdx.x,  blockIdx.y,  blockIdx.z
			, blockDim.x,  blockDim.y,  blockDim.z
			, gridDim.x,   gridDim.y,   gridDim.z
		);
}

int CheckDim(void)
//int main(void)
{
	int num_elem = 50;

	dim3 block(10);
	dim3 grid((num_elem + block.x - 1) / block.x);

	printf("print from CPU\n");
	printf("block.x = %d, block.y = %d, block.z = %d\n", block.x, block.y, block.z);
	printf(" grid.x = %d,  grid.y = %d,  grid.z = %d\n", grid.x, grid.y, grid.z);

	printf("\nprintf from GPU\n");
	checkIdx << <grid, block >> > ();

	return 0;
}

