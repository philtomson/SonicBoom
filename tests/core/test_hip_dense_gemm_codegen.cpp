// Compiler-generated dense bf16 GEMM on the ROCm backend (Phase 3).
//
// planner::glue::launch_dense_gemm's CUDA path runs the generated tensor-core
// kernel (nvgpu ldmatrix/mma, cp.async). Those ops have no ROCDL lowering (and
// RDNA has no MFMA), so the AMD path emits compiler/GemmCodegen.cpp's WMMA
// body instead: one BK slab per K step staged in shared memory, then a
// 16x16x16 bf16 -> f32 footprint per wave32 wavefront, one whole tile per warp
// slot. Same kernel ABI, same Y[token][row] output contract, same scale
// epilogue.
//
// This checks that HSACO end to end: the fp64 reference over the same
// bf16-rounded operands bounds the error, the rocBLAS GemmEx path is the
// comparison model when its (weak) entry points are loaded, the epilogue must
// be exactly alpha * the unscaled output, and a token's row must be bitwise
// independent of the batch size and of its block coordinate (the property the
// planner relies on to drop the fixed 128-column blocking).
//
// Shapes: a full 64x64x64 tile, a tail shape where M/K/N all miss the tile
// multiples, then every autotune candidate tile -- the tuner JITs all of them
// on a cold shape class, and they span 1..8 WMMA tiles per warp.
//
// Skips cleanly (exit 0) when no HIP device is present.

#include "../../core/src/planner/glue_jit.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

// Last: the HIP host headers define __noinline__ away, which breaks the
// __gnu__-namespaced attribute in libstdc++'s <format> if it has not been
// parsed yet (the include trace reaches it through <filesystem>). hip_blas.h
// pulls in <hip/hip_runtime.h>, so it stays below the standard headers too.
#include "../../core/src/nn/hip_blas.h"

namespace {

int g_failures = 0;
void check(bool ok, const char *what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool have_hip() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

int gfx_arch() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 0;
  int arch = 0;
  const char *g = std::strstr(prop.gcnArchName, "gfx");
  if (!g) return 0;
  for (g += 3; *g >= '0' && *g <= '9'; ++g) arch = arch * 10 + (*g - '0');
  return arch;
}

// Round to bfloat16 (round-to-nearest-even), as the kernel's operands are.
float bf16r(float x) {
  uint32_t u = std::bit_cast<uint32_t>(x);
  const uint32_t sign = u & 0x80000000u;
  u &= 0x7fffffffu;
  if (u >= 0x7f800000u) return x;  // inf / nan pass through
  u += 0x7fff + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  return std::bit_cast<float>(sign | u);
}

uint16_t bf16_bits(float x) {
  return uint16_t(std::bit_cast<uint32_t>(bf16r(x)) >> 16);
}

// fp64 reference over the bf16-rounded operands.
void reference(const std::vector<uint16_t> &x, const std::vector<uint16_t> &w,
               int m, int k, int n, std::vector<double> &out) {
  out.assign(size_t(n) * m, 0.0);
  std::vector<double> xf(size_t(n) * k), wf(size_t(m) * k);
  for (size_t i = 0; i < xf.size(); ++i)
    xf[i] = double(std::bit_cast<float>(uint32_t(x[i]) << 16));
  for (size_t i = 0; i < wf.size(); ++i)
    wf[i] = double(std::bit_cast<float>(uint32_t(w[i]) << 16));
  for (int t = 0; t < n; ++t)
    for (int r = 0; r < m; ++r) {
      double acc = 0.0;
      for (int i = 0; i < k; ++i)
        acc += xf[size_t(t) * k + i] * wf[size_t(r) * k + i];
      out[size_t(t) * m + r] = acc;
    }
}

struct DeviceBufs {
  uint16_t *x = nullptr;
  uint16_t *w = nullptr;
  float *y = nullptr;
  ~DeviceBufs() {
    if (x) hipFree(x);
    if (w) hipFree(w);
    if (y) hipFree(y);
  }
};

bool alloc_bufs(int m, int k, int nmax, DeviceBufs &b) {
  return hipMalloc(&b.x, size_t(nmax) * k * 2) == hipSuccess &&
         hipMalloc(&b.w, size_t(m) * k * 2) == hipSuccess &&
         hipMalloc(&b.y, size_t(nmax) * m * 4) == hipSuccess;
}

bool run(const sonicboom::compiler::DenseGemmShape &shape, DeviceBufs &b,
         int m, int k, int n, size_t x_off, size_t y_off) {
  return sonicboom::planner::glue::launch_dense_gemm(
      shape, b.x + x_off, b.w, b.y + y_off, m, k, n);
}

// The generated HSACO is written only after a successful generation and module
// load, so its presence proves the codegen arm ran. The ELF symbol table
// carries the kernel name (`gemm\0`).
bool codegen_cache_hit(const char *region) {
  const char *home = std::getenv("HOME");
  const int arch = gfx_arch();
  if (!home || arch == 0) return false;
  const std::filesystem::path p =
      std::filesystem::path(home) / ".cache" / "sonicboom" / "kernels" /
      (std::string(region) + "-gfx" + std::to_string(arch) + ".hsaco");
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  const std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  return body.find(std::string("gemm\0", 5)) != std::string::npos;
}

double max_rel(const std::vector<float> &got, const std::vector<double> &ref) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    worst = std::max(worst, std::fabs(double(got[i]) - ref[i]) /
                                std::max(1.0, std::fabs(ref[i])));
  return worst;
}

} // namespace

int main() {
  using sonicboom::compiler::DenseGemmShape;
  if (!have_hip()) {
    std::cout << "test_hip_dense_gemm_codegen SKIP (no HIP device)\n";
    return 0;
  }
  const DenseGemmShape shape{64, 64, 64};
  const int M = 1024, K = 1024, NMAX = 512;
  DeviceBufs b;
  if (!alloc_bufs(M, K, NMAX, b)) {
    std::cerr << "test_hip_dense_gemm_codegen: hipMalloc failed\n";
    return 1;
  }
  std::mt19937 rng(6006);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> xh(size_t(NMAX) * K), wh(size_t(M) * K);
  for (auto &v : xh) v = bf16_bits(nd(rng));
  for (auto &v : wh) v = bf16_bits(nd(rng) / std::sqrt(float(K)));
  check(hipMemcpy(b.x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(b.w, wh.data(), wh.size() * 2, hipMemcpyHostToDevice) ==
                hipSuccess,
        "operand upload");

  std::vector<float> got(size_t(NMAX) * M);
  const auto compare = [&](int n, const char *what) {
    std::vector<double> want;
    reference(xh, wh, M, K, n, want);
    check(hipMemcpy(got.data(), b.y, size_t(n) * M * 4,
                    hipMemcpyDeviceToHost) == hipSuccess,
          "result download");
    const double rel = max_rel(got, want);
    if (rel > 1e-3) std::cerr << "    n=" << n << " maxrel=" << rel << "\n";
    std::printf("  n=%d maxrel=%.3e\n", n, rel);
    check(rel <= 1e-3, what);
  };

  check(run(shape, b, M, K, 64, 0, 0), "generated dense GEMM launches");
  check(hipDeviceSynchronize() == hipSuccess, "kernel completes");
  compare(64, "generated GEMM matches fp64 on full tiles");

  // ---- 1b. accuracy beside the rocBLAS bf16 path it replaces ----
  // Same rule as the attention kernels: the replacement's fp64 error must not
  // exceed the replaced kernel's, on the same bf16 operands. hipBLAS is
  // reached through weak symbols (hip_blas.h); when the runtime has not
  // loaded it, the comparison has no baseline and is skipped.
  if (hipblasCreate != nullptr && hipblasGemmEx != nullptr) {
    std::vector<double> want;
    reference(xh, wh, M, K, 64, want);
    hipblasHandle_t h = nullptr;
    if (hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS) {
      float *ylt = nullptr;
      if (hipMalloc(&ylt, size_t(64) * M * 4) == hipSuccess) {
        const float alpha = 1.0f, beta = 0.0f;
        const hipblasStatus_t st = hipblasGemmEx(
            h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, 64, K, &alpha, b.w, HIP_R_16BF,
            K, b.x, HIP_R_16BF, K, &beta, ylt, HIP_R_32F, M,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        check(st == HIPBLAS_STATUS_SUCCESS, "hipBLAS bf16 GEMM");
        check(hipDeviceSynchronize() == hipSuccess, "hipBLAS GEMM completes");
        std::vector<float> yv(size_t(64) * M);
        check(hipMemcpy(yv.data(), ylt, yv.size() * 4,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "hipBLAS result download");
        const double cg_rel = max_rel(got, want);
        const double lt_rel = max_rel(yv, want);
        std::printf("  fp64 error: generated=%.3e hipblas=%.3e ratio=%.3f\n",
                    cg_rel, lt_rel, cg_rel / lt_rel);
        check(cg_rel <= lt_rel,
              "generated GEMM's fp64 error is not above hipBLAS's");
        hipFree(ylt);
      }
      hipblasDestroy(h);
    }
  } else {
    std::printf("  (hipBLAS not loaded: baseline comparison skipped)\n");
  }

  // ---- 2. tail shape: M, K and N all miss the tile multiples ----
  const int MT = 1000, KT = 1000, NT = 7;
  DeviceBufs bt;
  if (!alloc_bufs(MT, KT, NT, bt)) {
    std::cerr << "test_hip_dense_gemm_codegen: hipMalloc failed\n";
    return 1;
  }
  std::vector<uint16_t> xt(size_t(NT) * KT), wt(size_t(MT) * KT);
  for (auto &v : xt) v = bf16_bits(nd(rng));
  for (auto &v : wt) v = bf16_bits(nd(rng) / std::sqrt(float(KT)));
  check(hipMemcpy(bt.x, xt.data(), xt.size() * 2, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(bt.w, wt.data(), wt.size() * 2,
                      hipMemcpyHostToDevice) == hipSuccess,
        "tail operand upload");
  {
    const DenseGemmShape tail{32, 64, 32};
    check(run(tail, bt, MT, KT, NT, 0, 0), "tail-shape GEMM launches");
    check(hipDeviceSynchronize() == hipSuccess, "tail kernel completes");
    std::vector<double> want;
    reference(xt, wt, MT, KT, NT, want);
    std::vector<float> g(size_t(NT) * MT);
    check(hipMemcpy(g.data(), bt.y, g.size() * 4, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "tail result download");
    const double rel = max_rel(g, want);
    if (rel > 1e-3) std::cerr << "    tail maxrel=" << rel << "\n";
    check(rel <= 1e-3, "generated GEMM matches fp64 on a tail shape");
    std::printf("  tail m=%d k=%d n=%d maxrel=%.3e\n", MT, KT, NT, rel);
  }

  // ---- 2b. every autotune candidate tile ----
  // bm*bn/256 gives the register tile: 2, 4, 8 or 16 tokens per thread. The
  // dense tuner JITs all of them on a cold shape class, so each must compile
  // and be correct.
  for (const DenseGemmShape &s : {DenseGemmShape{64, 64, 64},
                                  DenseGemmShape{32, 64, 64},
                                  DenseGemmShape{64, 32, 64},
                                  DenseGemmShape{128, 32, 32},
                                  DenseGemmShape{32, 128, 32},
                                  DenseGemmShape{128, 64, 32},
                                  DenseGemmShape{128, 64, 64},
                                  DenseGemmShape{128, 128, 32},
                                  DenseGemmShape{128, 128, 64},
                                  DenseGemmShape{64, 128, 32}}) {
    check(run(s, b, M, K, 64, 0, 0), "candidate-shape GEMM launches");
    check(hipDeviceSynchronize() == hipSuccess, "candidate kernel completes");
    std::vector<double> want;
    reference(xh, wh, M, K, 64, want);
    std::vector<float> y(size_t(64) * M);
    check(hipMemcpy(y.data(), b.y, y.size() * 4, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "candidate result download");
    const double rel = max_rel(y, want);
    std::printf("  m%d n%d k%d maxrel=%.3e\n", s.bm, s.bn, s.bk, rel);
    check(rel <= 1e-3, "candidate tile matches fp64");
  }

  // ---- 3. batch invariance (bitwise) ----
  // Token 0 through the same kernel at N = 1, 7, 64 and 512.
  std::vector<uint32_t> row0;
  for (int n : {1, 7, 64, NMAX}) {
    check(run(shape, b, M, K, n, 0, 0), "invariance launch");
    check(hipDeviceSynchronize() == hipSuccess, "invariance kernel completes");
    std::vector<float> y(size_t(M), 0.0f);
    check(hipMemcpy(y.data(), b.y, y.size() * 4, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "invariance download");
    if (n == 1) {
      row0.resize(M);
      for (int r = 0; r < M; ++r) row0[r] = std::bit_cast<uint32_t>(y[r]);
      continue;
    }
    size_t diff = 0;
    for (int r = 0; r < M; ++r)
      if (std::bit_cast<uint32_t>(y[r]) != row0[r]) ++diff;
    if (diff)
      std::cerr << "    n=" << n << ": " << diff << "/" << M
                << " row-0 elements differ from n=1\n";
    check(diff == 0, "token 0 row is bitwise independent of N");
  }

  // ---- 3a. scalar-scale epilogue ----
  // The planner folds a projection's scale into the kernel, so its output must
  // be exactly alpha * (the unscaled kernel's output) in fp32.
  {
    const float alpha = 3.25f;
    const int n = 7;
    check(run(shape, b, M, K, n, 0, 0), "unscaled launch");
    check(hipDeviceSynchronize() == hipSuccess, "unscaled kernel completes");
    std::vector<float> y1(size_t(n) * M);
    check(hipMemcpy(y1.data(), b.y, y1.size() * 4, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "unscaled download");
    check(sonicboom::planner::glue::launch_dense_gemm(shape, b.x, b.w, b.y, M, K,
                                                     n, alpha),
          "scaled launch");
    check(hipDeviceSynchronize() == hipSuccess, "scaled kernel completes");
    std::vector<float> ya(size_t(n) * M);
    check(hipMemcpy(ya.data(), b.y, ya.size() * 4, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "scaled download");
    size_t diff = 0;
    for (size_t i = 0; i < ya.size(); ++i)
      if (std::bit_cast<uint32_t>(ya[i]) !=
          std::bit_cast<uint32_t>(alpha * y1[i]))
        ++diff;
    if (diff)
      std::cerr << "    epilogue: " << diff << "/" << ya.size()
                << " elements differ from alpha*y\n";
    check(diff == 0, "epilogue writes exactly alpha * the unscaled output");
  }

  // Tokens 0..6 between the 7-token and the 512-token call, and tokens 64..511
  // through a call whose base pointer is shifted by 64 tokens (same values,
  // different block coordinate).
  {
    check(run(shape, b, M, K, 7, 0, 0), "7-token launch");
    check(hipDeviceSynchronize() == hipSuccess, "7-token kernel completes");
    std::vector<float> y7(size_t(7) * M);
    check(hipMemcpy(y7.data(), b.y, y7.size() * 4, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "7-token download");
    check(run(shape, b, M, K, NMAX, 0, 0), "full-stream launch");
    check(hipDeviceSynchronize() == hipSuccess, "full-stream kernel completes");
    std::vector<float> yfull(size_t(NMAX) * M);
    check(hipMemcpy(yfull.data(), b.y, yfull.size() * 4,
                    hipMemcpyDeviceToHost) == hipSuccess,
          "full-stream download");
    size_t diff = 0;
    for (size_t i = 0; i < y7.size(); ++i)
      if (std::bit_cast<uint32_t>(y7[i]) != std::bit_cast<uint32_t>(yfull[i]))
        ++diff;
    check(diff == 0, "7-token rows == first 7 rows of the 512-token call");

    check(run(shape, b, M, K, NMAX - 64, size_t(64) * K, size_t(64) * M),
          "shifted-base launch");
    check(hipDeviceSynchronize() == hipSuccess, "shifted kernel completes");
    std::vector<float> yshift(size_t(NMAX - 64) * M);
    check(hipMemcpy(yshift.data(), b.y + size_t(64) * M, yshift.size() * 4,
                    hipMemcpyDeviceToHost) == hipSuccess,
          "shifted download");
    diff = 0;
    for (size_t i = 0; i < yshift.size(); ++i)
      if (std::bit_cast<uint32_t>(yshift[i]) !=
          std::bit_cast<uint32_t>(yfull[size_t(64) * M + i]))
        ++diff;
    if (diff)
      std::cerr << "    shifted: " << diff << "/" << yshift.size()
                << " elements differ\n";
    check(diff == 0, "token row does not depend on its block coordinate");
  }

  // ---- 4. the JIT kept the generated kernel on disk ----
  const char *cache_env = std::getenv("SONICBOOM_GLUE_CACHE");
  if (cache_env == nullptr || std::string_view(cache_env) != "0") {
    check(codegen_cache_hit("gemm-m64n64k64"),
          "the generated dense GEMM was compiled and cached");
    check(codegen_cache_hit("gemm-m32n64k32"),
          "the tail-shape dense GEMM was compiled and cached");
  }

  if (g_failures == 0) {
    std::cout << "test_hip_dense_gemm_codegen OK\n";
    return 0;
  }
  std::cerr << "test_hip_dense_gemm_codegen FAILED: " << g_failures
            << " check(s)\n";
  return 1;
}
