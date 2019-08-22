#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum ThreadLayout {
	G1D_B1D,
	G1D_B2D,
	G2D_B1D,
	G2D_B2D
};

bool kernelCall(float* MatA, float* MatB, float* MatC
	, int Row, int Col, int layout, dim3 gridDim, dim3 blockDim);