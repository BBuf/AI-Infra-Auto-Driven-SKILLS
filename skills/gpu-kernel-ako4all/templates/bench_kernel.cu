// Microbench skeleton for a CUDA C++ AKO4ALL kernel iteration.
//
// Copy this file into AKO4ALL/<your-task>/bench/bench_<kernel>.cu and adapt:
//   1. Replace kernel_baseline / kernel_candidate with launchers from
//      input/<kernel>.cu and solution/<kernel>.cu.
//   2. Replace reference_impl with the trusted reference (e.g. cuBLAS for
//      GEMM, or a CPU recompute for small problems).
//   3. Fill kShapes with at least one hot shape and at least one tail shape.
//
// Build:
//   nvcc -O3 -lineinfo --ptxas-options=-v bench_<kernel>.cu \
//       input/<kernel>.cu solution/<kernel>.cu -o bench_<kernel>
//
// Run:
//   ./bench_<kernel> --target baseline
//   ./bench_<kernel> --target candidate

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,    \
                   __LINE__, cudaGetErrorString(err__));                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct ShapeSpec {
  const char *tag;
  int M, N, K; // adapt to your kernel signature
};

// TODO: production-representative shapes.
static const std::vector<ShapeSpec> kShapes = {
    // {"hot",  4096, 4096, 1024},
    // {"tail",   16, 4096, 1024},
};

constexpr int kWarmup = 20;
constexpr int kIters = 100;
constexpr float kAtol = 1e-2f;
constexpr float kRtol = 1e-2f;

// TODO: implement these wrappers in input/<kernel>.cu and solution/<kernel>.cu.
extern "C" void reference_impl(const ShapeSpec &s, void *ws);
extern "C" void kernel_baseline(const ShapeSpec &s, void *ws);
extern "C" void kernel_candidate(const ShapeSpec &s, void *ws);

// TODO: shape-specific workspace allocator + correctness check that compares
// the output buffer against the reference. Return max_abs and max_rel.
struct CorrectnessReport {
  float max_abs;
  float max_rel;
};
extern "C" CorrectnessReport check_against_reference(const ShapeSpec &s,
                                                     void *ws);
extern "C" void *alloc_workspace(const ShapeSpec &s);
extern "C" void free_workspace(void *ws);

static double median(std::vector<float> &v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double percentile(std::vector<float> &v, double p) {
  size_t idx = static_cast<size_t>(p * v.size());
  if (idx >= v.size())
    idx = v.size() - 1;
  return v[idx];
}

static void run_target(const char *target) {
  void (*fn)(const ShapeSpec &, void *) = nullptr;
  if (std::strcmp(target, "baseline") == 0)
    fn = kernel_baseline;
  else if (std::strcmp(target, "candidate") == 0)
    fn = kernel_candidate;
  else {
    std::fprintf(stderr, "unknown target %s\n", target);
    std::exit(1);
  }

  std::printf("%-8s %12s %12s %12s %10s %10s\n", "shape", "median_us", "p95_us",
              "min_us", "max_abs", "max_rel");

  for (const auto &s : kShapes) {
    void *ws = alloc_workspace(s);

    // Correctness first.
    reference_impl(s, ws);
    fn(s, ws);
    CorrectnessReport rep = check_against_reference(s, ws);
    if (rep.max_abs > kAtol || rep.max_rel > kRtol) {
      std::printf("[FAIL] %-8s max_abs=%.3e max_rel=%.3e\n", s.tag, rep.max_abs,
                  rep.max_rel);
      free_workspace(ws);
      std::exit(2);
    }

    // Warmup.
    for (int i = 0; i < kWarmup; ++i)
      fn(s, ws);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations.
    std::vector<cudaEvent_t> starts(kIters), ends(kIters);
    for (int i = 0; i < kIters; ++i) {
      CUDA_CHECK(cudaEventCreate(&starts[i]));
      CUDA_CHECK(cudaEventCreate(&ends[i]));
    }
    for (int i = 0; i < kIters; ++i) {
      CUDA_CHECK(cudaEventRecord(starts[i]));
      fn(s, ws);
      CUDA_CHECK(cudaEventRecord(ends[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times_ms(kIters);
    for (int i = 0; i < kIters; ++i) {
      CUDA_CHECK(cudaEventElapsedTime(&times_ms[i], starts[i], ends[i]));
    }
    std::sort(times_ms.begin(), times_ms.end());
    double med_us = median(times_ms) * 1e3;
    double p95_us = percentile(times_ms, 0.95) * 1e3;
    double min_us = times_ms.front() * 1e3;

    std::printf("%-8s %12.2f %12.2f %12.2f %10.2e %10.2e\n", s.tag, med_us,
                p95_us, min_us, rep.max_abs, rep.max_rel);

    for (int i = 0; i < kIters; ++i) {
      CUDA_CHECK(cudaEventDestroy(starts[i]));
      CUDA_CHECK(cudaEventDestroy(ends[i]));
    }
    free_workspace(ws);
  }
}

int main(int argc, char **argv) {
  const char *target = "candidate";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--target") == 0)
      target = argv[i + 1];
  }
  run_target(target);
  return 0;
}
