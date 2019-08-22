#include "kernelCall.h"
#include "DS_timer.h"

#define ROW_SIZE (8192)
#define COL_SIZE (8192)
#define MAT_SIZE (ROW_SIZE*COL_SIZE)

#define ID2INDEX(row, col) (row*COL_SIZE+col)

bool MatAddGPU_2D2D(float* da, float* db, float* dc) {
	dim3 blockDim(32, 32);
	dim3 gridDim(ceil((float)COL_SIZE / blockDim.x), ceil((float)ROW_SIZE / blockDim.y)); 
	return kernelCall(da, db, dc, ROW_SIZE, COL_SIZE, G2D_B2D, gridDim, blockDim);
}

int main(void) {
	float* A, * B, * C, * hC;
	float* dA, * dB, * dC;

	A = new float[MAT_SIZE]; memset(A, 0, sizeof(float) * MAT_SIZE);
	B = new float[MAT_SIZE]; memset(B, 0, sizeof(float) * MAT_SIZE);
	C = new float[MAT_SIZE]; memset(C, 0, sizeof(float) * MAT_SIZE);
	hC = new float[MAT_SIZE]; memset(hC, 0, sizeof(float) * MAT_SIZE);

	cudaMalloc(&dA, sizeof(float) * MAT_SIZE); cudaMemset(dA, 0, sizeof(float) * MAT_SIZE);
	cudaMalloc(&dB, sizeof(float) * MAT_SIZE); cudaMemset(dB, 0, sizeof(float) * MAT_SIZE);
	cudaMalloc(&dC, sizeof(float) * MAT_SIZE); cudaMemset(dC, 0, sizeof(float) * MAT_SIZE);

	for (int i = 0; i < MAT_SIZE; i++) {
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

	for (int i = 0; i < ROW_SIZE; i++) {
		for (int j = 0; j < COL_SIZE; j++) {
			hC[ID2INDEX(i, j)] = A[ID2INDEX(i, j)] + B[ID2INDEX(i, j)];
		}
	}

	cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE, cudaMemcpyHostToDevice);

	MatAddGPU_2D2D(dA, dB, dC); 
	cudaDeviceSynchronize();

	cudaMemcpy(C, dC, sizeof(float) * MAT_SIZE, cudaMemcpyDeviceToHost);

	bool isCorrect = true;
	for (int i = 0; i < MAT_SIZE; i++) {
		//printf("hC[%d]=%f   C[%d]=%f\n", i, hC[i], i, C[i]);
		if (hC[i] != C[i]) {
			isCorrect = false;
			break;
		}
	}

	if (isCorrect) printf("GPU works well!\n");
	else printf("GPU fail to calculate.\n");

	return 0;
}