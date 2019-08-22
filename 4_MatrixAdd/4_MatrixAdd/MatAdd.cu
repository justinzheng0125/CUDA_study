#include "kernelCall.h"

__global__ void MatAdd_G2D_B2D(float* MatA, float* MatB, float* MatC, int row, int col) {
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * col + ix;
	if (ix < col && iy < row) {
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

bool kernelCall(float* MatA, float* MatB, float* MatC
	, int Row, int Col, int layout, dim3 gridDims, dim3 blockDims) {

	switch(layout) {
	
	
	
	case G1D_B1D:
	case G2D_B1D:
	case G2D_B2D:
		MatAdd_G2D_B2D << <gridDims, blockDims >> > (MatA, MatB, MatC, Row, Col);
		break;
	default:
		printf("Not Supporting type %d layout\n", layout);
		return false;
	}

	return true;
}