
#include <cstdlib>
#include <cstdio>

#include "cuda_runtime.h"

#include <nvtx3/nvToolsExt.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

__global__ void gpu_matmul(float* xout, float* x, float* w, int n, int d) {
	int insertPos = blockIdx.x * blockDim.x + threadIdx.x;

	if (insertPos < d) {
		float val = 0.0f;
        for (int readPos = 0; readPos < n; readPos++) {
            val += w[insertPos * n + readPos] * x[readPos];
        }
        xout[insertPos] = val;
	}
}
/*
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}
*/

extern "C" void cuda_matmul(float* xout, float* x, float* w, int n, int d) {
    const int threadsPerBlock[1] = { 256 };

    const int outputVectorSize = d * sizeof(float);
	const int matrixSize = n * d * sizeof(float);
	const int inputVectorSize = n * sizeof(float);

    float* gpu_out, * gpu_x, * gpu_w;

    nvtxRangePushA("1. Alloc");    
    CHECK(cudaMalloc((void**)&gpu_out, outputVectorSize));
	CHECK(cudaMalloc((void**)&gpu_x, inputVectorSize));
	CHECK(cudaMalloc((void**)&gpu_w, matrixSize));
    nvtxRangePop();
    
    nvtxRangePushA("2. Transfer to GPU");
    CHECK(cudaMemcpy(gpu_x, x, inputVectorSize,
		cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(gpu_w, w, matrixSize,
		cudaMemcpyHostToDevice));
    nvtxRangePop();

    dim3 grid( ( (d + threadsPerBlock[0] - 1) / threadsPerBlock[0] ));
    dim3 tpb(threadsPerBlock[0], 1);

    nvtxRangePushA("3. Compute");
    gpu_matmul << <grid, tpb >> > (gpu_out, gpu_x, gpu_w, n, d);

    CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CHECK(cudaGetLastError());

    nvtxRangePushA("4. Transfer from GPU");
    CHECK(cudaMemcpy(xout, gpu_out, outputVectorSize, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    nvtxRangePushA("5. CUDA Free");
    CHECK(cudaFree(gpu_x)); CHECK(cudaFree(gpu_w)); CHECK(cudaFree(gpu_out));
    nvtxRangePop();
}