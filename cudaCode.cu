
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

//
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

extern "C" float* cuda_upload_matrix(float* matrixPointer, size_t byteSize){
    float* gpuPointer;
    CHECK(cudaMalloc((void**)&gpuPointer, byteSize));
    CHECK(cudaMemcpy(gpuPointer, matrixPointer, byteSize, cudaMemcpyHostToDevice));
    return gpuPointer;
}

extern "C" void cuda_free_matrix(float* gpuPointer){
    CHECK(cudaFree((void**)&gpuPointer));
}

//xout and x are still cpu pointers, but instead of w we now have gpu_w passed directly
// as a gpu pointer
extern "C" void cuda_matmul(float* xout, float* x, float* gpu_w, int n, int d) {
    const int threadsPerBlock[1] = { 256 };

    const int outputVectorSize = d * sizeof(float);
	const int inputVectorSize = n * sizeof(float);

    float* gpu_out, * gpu_x;

    nvtxRangePushA("1. Alloc");
    CHECK(cudaMalloc((void**)&gpu_out, outputVectorSize));
	CHECK(cudaMalloc((void**)&gpu_x, inputVectorSize));
    nvtxRangePop();

    nvtxRangePushA("2. Transfer to GPU");
    CHECK(cudaMemcpy(gpu_x, x, inputVectorSize,
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
    CHECK(cudaFree(gpu_x)); CHECK(cudaFree(gpu_out));
    nvtxRangePop();
}