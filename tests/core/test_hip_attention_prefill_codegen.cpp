// Compiler-generated flash attention (prefill) on the ROCm backend (Phase 3).
//
// nn::cuda::flash_attention_chunk_dev_pos (the HIP twin's namespace) routes a
// shape class through the MLIR-generated `fa` kernel when the implementation
// switch selects codegen. On a CUDA build that kernel is the tensor-core FA-2
// body (nvgpu ldmatrix/mma, nvvm.cp.async). That body has no ROCDL lowering, so
// the AMD path emits compiler/AttentionCodegen.cpp's scalar warp-per-row
// variant instead: one warp per query row, lane-strided f32 fma chains with a
// butterfly (xor) q·k reduction, no shared memory and no barriers, same kernel
// ABI and mask semantics.
//
// This checks that HSACO end to end: the fp64 reference bounds the error, the
// hand-written warp kernel (the "own" arm, now reachable for every HIP shape)
// is the comparison model, and the disk cache proves the generated kernel ran
// rather than the fallback.
//
// Shapes: head_dim 64 (Qwen2.5 GQA ratio), 256 (Gemma sliding-window and
// causal) and 512 (Gemma global), with a short segment, a full window, and a
// slot-count smaller than the context (ring wrap-around).
//
// Skips cleanly (exit 0) when no HIP device is present.

#include "../../core/src/nn/cuda_resident.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Last: the HIP host headers define __noinline__ away, which breaks the
// __gnu__-namespaced attribute in libstdc++'s <format> if it has not been
// parsed yet (the include trace reaches it through <filesystem>).
#include <hip/hip_runtime.h>

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

struct Case {
  const char *name;
  int d;
  int nhkv;
  int gqa;
  uint64_t start;    // absolute position of the segment's first query row
  int n;             // query rows in this segment
  uint64_t sliding;  // 0 = causal
  uint64_t nslots;
  int br;            // generator default tile, for the cache-name check
  int bc;
  int dps;
};

// The generated HSACO is written only after a successful generation and module
// load, so its presence proves the codegen arm ran. The ELF symbol table
// carries the kernel name (`fa\0`).
bool codegen_cache_hit(const Case &c) {
  const char *home = std::getenv("HOME");
  const int arch = gfx_arch();
  if (!home || arch == 0) return false;
  const std::string mask = c.sliding != 0 ? "-window" : "-causal";
  const std::string region =
      "attn-d" + std::to_string(c.d) + "-g" + std::to_string(c.gqa) + mask +
      "-b" + std::to_string(c.br) + "x" + std::to_string(c.bc) + "-dps" +
      std::to_string(c.dps);
  const std::filesystem::path p =
      std::filesystem::path(home) / ".cache" / "sonicboom" / "kernels" /
      (region + "-gfx" + std::to_string(arch) + ".hsaco");
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  const std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  return body.find(std::string("fa\0", 3)) != std::string::npos;
}

// fp64 softmax over keys [lo..p] for every query row. Key j lives in the batch
// part when j >= start (index j - start) and in the ring slot j % nslots
// otherwise.
void reference(const Case &c, const std::vector<float> &q,
               const std::vector<float> &kb, const std::vector<float> &vb,
               const std::vector<float> &kc, const std::vector<float> &vc,
               float scale, std::vector<double> &out) {
  const int nhq = c.nhkv * c.gqa;
  const size_t d = size_t(c.d);
  out.assign(size_t(c.n) * nhq * d, 0.0);
  for (int r = 0; r < c.n; ++r) {
    const long long p = (long long)c.start + r;
    const long long lo =
        (c.sliding != 0 && p >= (long long)c.sliding)
            ? p - (long long)c.sliding + 1
            : 0;
    for (int h = 0; h < nhq; ++h) {
      const int hk = h / c.gqa;
      std::vector<double> s(size_t(p - lo + 1));
      double mx = -1e30;
      for (long long j = lo; j <= p; ++j) {
        const float *krow;
        if (j < (long long)c.start)
          krow = kc.data() +
                 (size_t(uint64_t(j) % c.nslots) * c.nhkv + hk) * d;
        else
          krow = kb.data() + (size_t(j - (long long)c.start) * c.nhkv + hk) * d;
        const float *qrow = q.data() + (size_t(r) * nhq + h) * d;
        double acc = 0;
        for (int i = 0; i < c.d; ++i)
          acc += double(qrow[i]) * double(krow[i]);
        s[size_t(j - lo)] = acc * scale;
        mx = std::max(mx, s[size_t(j - lo)]);
      }
      double sum = 0;
      for (auto &x : s) {
        x = std::exp(x - mx);
        sum += x;
      }
      for (int i = 0; i < c.d; ++i) {
        double acc = 0;
        for (long long j = lo; j <= p; ++j) {
          const float *vrow;
          if (j < (long long)c.start)
            vrow = vc.data() +
                   (size_t(uint64_t(j) % c.nslots) * c.nhkv + hk) * d;
          else
            vrow =
                vb.data() + (size_t(j - (long long)c.start) * c.nhkv + hk) * d;
          acc += s[size_t(j - lo)] * double(vrow[i]);
        }
        out[size_t(r) * nhq * d + size_t(h) * d + i] = acc / sum;
      }
    }
  }
}

double max_rel(const std::vector<float> &got, const std::vector<double> &ref) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    // Relative for normal values, absolute below 1e-3 (see the decode test).
    const double denom = std::abs(ref[i]) + 1e-3;
    worst = std::max(worst, std::abs(double(got[i]) - ref[i]) / denom);
  }
  return worst;
}

double max_abs(const std::vector<float> &got, const std::vector<double> &ref) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    worst = std::max(worst, std::abs(double(got[i]) - ref[i]));
  return worst;
}

struct DeviceBufs {
  float *q = nullptr, *kb = nullptr, *vb = nullptr, *kc = nullptr, *vc = nullptr;
  float *out = nullptr;
  uint64_t *start = nullptr;
  ~DeviceBufs() {
    for (float *p : {q, kb, vb, kc, vc, out})
      if (p) hipFree(p);
    if (start) hipFree(start);
  }
};

bool run(const Case &c, const std::vector<float> &q,
         const std::vector<float> &kb, const std::vector<float> &vb,
         const std::vector<float> &kc, const std::vector<float> &vc,
         const char *impl, std::vector<float> &got) {
  const int nhq = c.nhkv * c.gqa;
  const size_t d = size_t(c.d);
  DeviceBufs b;
  if (hipMalloc(&b.q, q.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&b.kb, kb.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&b.vb, vb.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&b.kc, kc.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&b.vc, vc.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&b.out, size_t(c.n) * nhq * d * sizeof(float)) != hipSuccess ||
      hipMalloc(&b.start, sizeof(uint64_t)) != hipSuccess)
    return false;
  hipMemcpy(b.q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(b.kb, kb.data(), kb.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(b.vb, vb.data(), vb.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(b.kc, kc.data(), kc.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(b.vc, vc.data(), vc.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(b.start, &c.start, sizeof(uint64_t), hipMemcpyHostToDevice);

  setenv("SONICBOOM_ATTN_IMPL", impl, 1);
  unsetenv("SONICBOOM_ATTN_CODEGEN");
  const float scale = 1.0f / std::sqrt(float(c.d));
  const bool ok = sonicboom::nn::cuda::flash_attention_chunk_dev_pos(
      b.q, b.kb, b.vb, b.kc, b.vc, b.start, c.nslots, c.n, nhq, c.nhkv, c.d,
      c.sliding, scale, b.out);
  if (!ok) return false;
  if (hipDeviceSynchronize() != hipSuccess) return false;
  got.assign(size_t(c.n) * nhq * d, 0.0f);
  hipMemcpy(got.data(), b.out, got.size() * sizeof(float),
            hipMemcpyDeviceToHost);
  return true;
}

} // namespace

int main() {
  if (!have_hip()) {
    std::cout << "SKIP: no HIP device\n";
    return 0;
  }

  const Case cases[] = {
      // Gemma sliding-window head: short segment inside a full window.
      {"d256-window-short", 256, 2, 4, 37, 5, 512, 2048, 64, 16, 4},
      // Gemma sliding-window head starting beyond the window.
      {"d256-window-full", 256, 2, 4, 700, 4, 512, 2048, 64, 16, 4},
      // Gemma global head: causal context.
      {"d256-causal", 256, 2, 4, 300, 6, 0, 2048, 64, 16, 4},
      // Ring wrap-around: context longer than the cache slot count.
      {"d256-causal-wrap", 256, 2, 4, 300, 4, 0, 128, 64, 16, 4},
      // Gemma global head at head_dim 512.
      {"d512-causal", 512, 2, 4, 700, 4, 0, 2048, 32, 16, 8},
      // Qwen2.5 head shape.
      {"d64-causal-gqa7", 64, 2, 7, 200, 6, 0, 2048, 64, 64, 1},
  };

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (const Case &c : cases) {
    std::cout << c.name << ":\n";
    const int nhq = c.nhkv * c.gqa;
    const size_t d = size_t(c.d);
    std::vector<float> q(size_t(c.n) * nhq * d),
        kb(size_t(c.n) * c.nhkv * d), vb(size_t(c.n) * c.nhkv * d),
        kc(size_t(c.nslots) * c.nhkv * d), vc(size_t(c.nslots) * c.nhkv * d);
    for (auto &x : q) x = dist(rng);
    for (auto &x : kb) x = dist(rng);
    for (auto &x : vb) x = dist(rng);
    // The cache holds the positions before the segment's start.
    for (auto &x : kc) x = dist(rng);
    for (auto &x : vc) x = dist(rng);

    std::vector<double> ref;
    reference(c, q, kb, vb, kc, vc, 1.0f / std::sqrt(float(c.d)), ref);

    std::vector<float> own, cg, cg2;
    check(run(c, q, kb, vb, kc, vc, "own", own), "own kernel ran");
    check(run(c, q, kb, vb, kc, vc, "codegen", cg), "codegen kernel ran");
    check(run(c, q, kb, vb, kc, vc, "codegen", cg2),
          "codegen kernel ran (repeat)");
    check(codegen_cache_hit(c), "generated HSACO was written to the disk cache");

    if (own.size() == ref.size() && cg.size() == ref.size()) {
      const double own_err = max_rel(own, ref);
      const double cg_err = max_rel(cg, ref);
      std::cout << "  max rel error: own=" << own_err
                << " codegen=" << cg_err << "  (abs: own=" << max_abs(own, ref)
                << " codegen=" << max_abs(cg, ref) << ")\n";
      if (cg_err > 1e-3) {
        for (int i = 0; i < 6; ++i)
          std::cerr << "    [0][" << i << "] ref=" << ref[size_t(i)]
                    << " own=" << own[size_t(i)] << " cg=" << cg[size_t(i)]
                    << "\n";
      }
      check(cg_err <= 1e-3, "codegen within fp64 tolerance");
      // The scalar body reorders the softmax rescale relative to the
      // hand-written per-tile kernel (per key instead of per tile), so the
      // outputs agree only to f32 tolerance -- but must not be worse.
      check(cg_err <= std::max(2.0 * own_err, 1e-4),
            "codegen error not worse than the hand-written kernel");
    } else {
      check(false, "own and codegen outputs have the same size");
    }
    check(cg.size() == cg2.size() &&
              std::equal(cg.begin(), cg.end(), cg2.begin()),
          "codegen is deterministic (bit-identical repeat)");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "OK\n";
  return 0;
}
