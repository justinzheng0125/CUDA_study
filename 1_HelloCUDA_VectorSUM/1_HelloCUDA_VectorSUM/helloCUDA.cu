#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// __global__ : device code (gpu code)
// running from GPU
__global__ void helloCUDA(void) 
{
	printf("Hello CUDA from GPU\n");
}

int helloCUDAmain(void)
//int main(void) 
{
	printf("Hellow GPU from CPU\n");
	// 9 thread from gpu
	helloCUDA <<< 1, 9 >>>();
	return 0;
}