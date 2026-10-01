// matmul.cu - naive vs. shared-memory-tiled vs. cuBLAS square matmul, timed
// with CUDA events. Square C[N,N] = A[N,N] * B[N,N], row-major, float32.
//
// Build:  make            (produces ./matmul_bench, see Makefile for -arch)
// Run:    ./matmul_bench <naive|tiled|cublas> <N> <iters>
// Output: one CSV-ish line to stdout: version,N,median_ms,gflops
//         (parsed by run_matmul.sh - keep this format if you edit the file)
//
// Profile a single run with Nsight Compute, e.g.:
//   ncu --set full -o tiled_1024 ./matmul_bench tiled 1024 5
// Look at "Compute (SM) Throughput" / "Memory Throughput" to see whether a
// version is compute- or memory-bound (this is the roofline question from
// phase 1, now on the GPU).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,        \
                   __LINE__, (int)st__);                                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

constexpr int TILE =
    32; // 32x32 tile: one thread per output element, matches a warp width

// Naive: each thread computes one C[row,col], reading A's row and B's column
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n)
    return;
  float acc = 0.0f;
  for (int k = 0; k < n; ++k)
    acc += A[row * n + k] * B[k * n + col];
  C[row * n + col] = acc;
}

// Tiled: cooperatively load a TILE x TILE block of A and B into shared memory
// per step, so each global-memory element is reused TILE times by the block
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < n && col < n)
    C[row * n + col] = acc;
}

static void fill_random(std::vector<float> &v, unsigned seed) {
  srand(seed);
  for (auto &x : v)
    x = (float)rand() / RAND_MAX - 0.5f;
}

static double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : (v[m - 1] + v[m]) / 2.0;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <naive|tiled|cublas> <N> <iters>\n",
                 argv[0]);
    return 1;
  }
  std::string version = argv[1];
  int n = std::atoi(argv[2]);
  int iters = std::atoi(argv[3]);
  const int warmup = std::max(3, iters / 5);

  size_t bytes = (size_t)n * n * sizeof(float);
  std::vector<float> hA(n * n), hB(n * n), hC(n * n);
  fill_random(hA, 1);
  fill_random(hB, 2);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

  cublasHandle_t handle = nullptr;
  const float alpha = 1.0f, beta = 0.0f;
  if (version == "cublas")
    CUBLAS_CHECK(cublasCreate(&handle));

  auto launch = [&]() {
    if (version == "naive") {
      matmul_naive<<<grid, block>>>(dA, dB, dC, n);
    } else if (version == "tiled") {
      matmul_tiled<<<grid, block>>>(dA, dB, dC, n);
    } else if (version == "cublas") {
      // cuBLAS is column-major; computing B^T * A^T = (A*B)^T with row-major
      // buffers gives the same bytes as C = A*B row-major. Standard trick,
      // avoids a real transpose.
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                               &alpha, dB, n, dA, n, &beta, dC, n));
    } else {
      std::fprintf(stderr, "unknown version '%s'\n", version.c_str());
      std::exit(1);
    }
  };

  for (int i = 0; i < warmup; ++i)
    launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> times_ms;
  times_ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times_ms.push_back(ms);
  }

  double med_ms = median(times_ms);
  double gflops =
      2.0 * n * n * n / (med_ms / 1000.0) / 1e9; // 2*N^3 FLOPs for NxN*NxN

  // Correctness spot-check against a naive CPU reference on a few elements,
  // so a broken kernel shows up as garbage GFLOPS-with-wrong-answer rather
  // than silently passing.
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
  int checks = std::min(5, n);
  double max_abs_err = 0.0;
  for (int i = 0; i < checks; ++i) {
    int r = (i * 37) % n, c = (i * 53) % n;
    float ref = 0.0f;
    for (int k = 0; k < n; ++k)
      ref += hA[r * n + k] * hB[k * n + c];
    max_abs_err = std::max(max_abs_err, (double)std::abs(ref - hC[r * n + c]));
  }
  if (max_abs_err > 1e-1) {
    std::fprintf(stderr,
                 "warning: max abs error vs. CPU reference = %.4f (check "
                 "kernel correctness)\n",
                 max_abs_err);
  }

  // version,N,median_ms,gflops  <- run_matmul.sh parses this exact order
  std::printf("%s,%d,%.4f,%.2f\n", version.c_str(), n, med_ms, gflops);

  if (handle)
    cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
