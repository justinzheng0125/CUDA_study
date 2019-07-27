#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void helloCUDA(void) {

	printf("Hello CUDA from GPU\n");
}

int main2(void) {
	printf("Hellow GPU from CPU\n");
	helloCUDA <<< 1, 10 >>>();
	return 0;
}