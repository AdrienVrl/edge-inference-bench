// kernel.c - INT8 dot product: naive scalar vs. compiler auto-vectorized vs.
// hand-written NEON vs. NEON dot-product (vdotq_s32, Armv8.2+).
//
// This is the AArch64 counterpart to phase 2's CUDA matmul: same idea of
// "how much does explicit SIMD buy you over scalar / over letting the
// compiler try", just at a different level of the memory hierarchy.
//
// Build (baseline, portable to any AArch64 core):
//   make
// Build the dotprod variant too (needs Armv8.2+ with the dot-product
// extension - Graviton2/3, Ampere Altra, and similar all have it; check
// with `grep asimddp /proc/cpuinfo` first):
//   make dotprod
//
// Run:
//   ./neon_bench <naive|autovec|neon> <N> <iters>
//   ./neon_bench_dotprod dotprod <N> <iters>
// Output (parsed by run_neon.sh - keep this format if you edit the file):
//   version,N,median_ms,gops
//
// All versions compute the same int8 dot product with an int32 accumulator.
// Integer arithmetic is exact (no rounding), so every version's result is
// checked against the naive scalar version's result and must match exactly -
// any mismatch means a real bug in the SIMD code, not just noise.

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Force this function to stay genuinely scalar regardless of -O3, so
// "naive" is a true baseline and "autovec" (identical loop, no attribute)
// shows exactly what the compiler's auto-vectorizer buys you for free.
#if defined(__clang__)
#define NOVEC __attribute__((optnone))
#else
#define NOVEC __attribute__((optimize("O0")))
#endif

static int8_t* alloc_random_vec(int n, unsigned seed) {
  int8_t* v = malloc((size_t)n * sizeof(int8_t));
  srand(seed);
  for (int i = 0; i < n; ++i) v[i] = (int8_t)((rand() % 256) - 128);
  return v;
}

NOVEC
static int32_t dot_naive(const int8_t* a, const int8_t* b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; ++i) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

// Identical loop to dot_naive, but compiled at the file's real -O3, so GCC/
// Clang's auto-vectorizer is free to do whatever it can with plain scalar C.
static int32_t dot_autovec(const int8_t* a, const int8_t* b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; ++i) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

// Hand-written NEON: widen 8x int8 -> 8x int16 via a widening multiply
// (vmull_s8), then pairwise-widen-and-accumulate into a 4-lane int32
// accumulator (vpadalq_s16). Available on every AArch64 core - no extra
// arch flag needed.
static int32_t dot_neon(const int8_t* a, const int8_t* b, int n) {
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    int8x8_t va = vld1_s8(a + i);
    int8x8_t vb = vld1_s8(b + i);
    int16x8_t prod = vmull_s8(va, vb);       // 8x (int8*int8 -> int16), exact
    acc = vpadalq_s16(acc, prod);            // pairwise widen-add into int32x4
  }
  int32_t acc_s = vaddvq_s32(acc);           // horizontal sum across the 4 lanes
  for (; i < n; ++i) acc_s += (int32_t)a[i] * (int32_t)b[i];  // scalar tail
  return acc_s;
}

#ifdef ENABLE_DOTPROD
// vdotq_s32: 4-way dot product per lane, so one instruction covers 16 int8
// MACs at once (4 lanes x 4-wide dot each) vs. dot_neon's 8-per-iteration
// widen+add. Needs Armv8.2+ dot-product extension (build with `make
// dotprod`) and a CPU that supports it at runtime, or this will SIGILL.
static int32_t dot_neon_dotprod(const int8_t* a, const int8_t* b, int n) {
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);
    acc = vdotq_s32(acc, va, vb);
  }
  int32_t acc_s = vaddvq_s32(acc);
  for (; i < n; ++i) acc_s += (int32_t)a[i] * (int32_t)b[i];  // scalar tail
  return acc_s;
}
#endif

static double median(double* v, int n) {
  // simple insertion sort - n is small (iters, typically <= a few hundred)
  for (int i = 1; i < n; ++i) {
    double key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
    v[j + 1] = key;
  }
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <naive|autovec|neon|dotprod> <N> <iters>\n", argv[0]);
    return 1;
  }
  const char* version = argv[1];
  int n = atoi(argv[2]);
  int iters = atoi(argv[3]);
  int warmup = iters / 5 > 3 ? iters / 5 : 3;

  int8_t* a = alloc_random_vec(n, 1);
  int8_t* b = alloc_random_vec(n, 2);
  int32_t ref = dot_naive(a, b, n);  // reference result, every version must match exactly

  int32_t (*fn)(const int8_t*, const int8_t*, int) = NULL;
  if (strcmp(version, "naive") == 0) fn = dot_naive;
  else if (strcmp(version, "autovec") == 0) fn = dot_autovec;
  else if (strcmp(version, "neon") == 0) fn = dot_neon;
#ifdef ENABLE_DOTPROD
  else if (strcmp(version, "dotprod") == 0) fn = dot_neon_dotprod;
#endif
  else {
    fprintf(stderr, "unknown or unbuilt version '%s' (dotprod needs `make dotprod`)\n", version);
    return 1;
  }

  int32_t result = fn(a, b, n);
  if (result != ref) {
    fprintf(stderr, "WARNING: %s result %d != naive reference %d (real bug, not rounding - "
                     "integer dot products must match exactly)\n", version, result, ref);
  }

  for (int i = 0; i < warmup; ++i) fn(a, b, n);

  double* times = malloc((size_t)iters * sizeof(double));
  for (int i = 0; i < iters; ++i) {
    double t0 = now_ms();
    fn(a, b, n);
    times[i] = now_ms() - t0;
  }
  double med_ms = median(times, iters);
  double gops = 2.0 * n / (med_ms / 1000.0) / 1e9;  // 1 MAC = 2 ops (multiply + add)

  // version,N,median_ms,gops  <- run_neon.sh parses this exact order
  printf("%s,%d,%.6f,%.3f\n", version, n, med_ms, gops);

  free(a);
  free(b);
  free(times);
  return 0;
}
