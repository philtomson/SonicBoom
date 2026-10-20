// Compiler-generated glue kernel test, ROCm/AMDGPU twin (Phase 2).
//
// The MLIR -> ROCDL -> HSACO pipeline (compiler::generate_glue_hsaco) and the
// HIP module JIT (planner::glue::launch) must produce a loadable HSA code
// object for every glue region and run it with the region's ABI. The CUDA twin
// (test_glue_codegen.cpp) byte-compares the generated kernels with the
// hand-written CUDA kernels; here the regions are compiled, loaded through
// hipModuleLoadData, launched with valid buffers, and (for the arithmetic the
// CPU can reproduce) checked within an fp32 tolerance. The q8_1 regions are
// checked structurally: a positive block scale and a saturated ±127 quant.
//
// Skips cleanly (exit 0) when no AMD device is present.

#include <sonicboom/compiler/GlueCodegen.h>

#include "../../core/src/planner/glue_jit.h"

// The standard headers first: HIP's headers define macros that clash with
// libstdc++'s attribute spellings (the <format> in <iostream>), so <hip/...>
// has to come last.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

namespace {

namespace compiler = sonicboom::compiler;
namespace glue = sonicboom::planner::glue;

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool have_rocm() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

int gfx_arch() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 0;
  const char* g = std::strstr(prop.gcnArchName, "gfx");
  if (g == nullptr) return 0;
  int arch = 0;
  for (g += 3; *g >= '0' && *g <= '9'; ++g) arch = arch * 10 + (*g - '0');
  return arch;
}

float half_roundtrip(float v) {
  return static_cast<float>(static_cast<_Float16>(v));
}

// GGML_GELU_FP16 tanh-GELU, the reference the region and gelu_fp16_kernel share.
float gelu_ref(float x) {
  if (x <= -10.0f) return 0.0f;
  if (x >= 10.0f) return x;
  const float v = half_roundtrip(x);
  constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876f;
  constexpr float kCoefA = 0.044715f;
  const float g =
      0.5f * v * (1.0f + std::tanh(kSqrt2OverPi * v * (1.0f + kCoefA * v * v)));
  return half_roundtrip(g);
}

// Worst relative error against max(1, |ref|).
double rel_err(double got, double ref) {
  return std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
}

template <class T>
struct DeviceBuf {
  T* ptr = nullptr;
  int n = 0;

  ~DeviceBuf() {
    if (ptr != nullptr) (void)hipFree(ptr);
  }

  bool alloc(int count) {
    n = count;
    return hipMalloc(&ptr, sizeof(T) * size_t(count)) == hipSuccess;
  }

  bool upload(const std::vector<T>& host) {
    return hipMemcpy(ptr, host.data(), sizeof(T) * host.size(),
                     hipMemcpyHostToDevice) == hipSuccess;
  }

  bool download(std::vector<T>& host) const {
    host.resize(size_t(n));
    return hipMemcpy(host.data(), ptr, sizeof(T) * size_t(n),
                     hipMemcpyDeviceToHost) == hipSuccess;
  }
};

int blocks_for(int n, int threads) { return (n + threads - 1) / threads; }

// --- regions ---------------------------------------------------------------

void test_slice_scale() {
  const int rows = 3, len = 53, stride = 61, offset = 7;
  const float scale = 2.5f;
  const int xs = offset + (rows - 1) * stride + len;
  const std::size_t xs_n = std::size_t(xs);
  std::vector<float> x(xs_n), want(std::size_t(rows * len));
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  for (auto& v : x) v = dist(rng);
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < len; ++k)
      want[size_t(r) * len + k] = x[size_t(offset + r * stride + k)] * scale;

  DeviceBuf<float> dx, dy;
  check(dx.alloc(xs) && dx.upload(x) && dy.alloc(rows * len), "slice_scale alloc");
  void* xarg = dx.ptr;
  void* yarg = dy.ptr;
  int rarg = rows, larg = len, oarg = offset, sarg = stride;
  float farg = scale;
  void* args[] = {&xarg, &yarg, &rarg, &larg, &oarg, &sarg, &farg};
  const int grid = blocks_for(rows * len, compiler::kSliceScaleBlockThreads);
  check(glue::launch("slice_scale", args, grid,
                              compiler::kSliceScaleBlockThreads, 0),
        "slice_scale launch");
  check(hipDeviceSynchronize() == hipSuccess, "slice_scale sync");
  std::vector<float> got;
  check(dy.download(got), "slice_scale download");
  check(got.size() == want.size() && std::memcmp(got.data(), want.data(),
                                                want.size() * sizeof(float)) == 0,
        "slice_scale output is x[offset + r*stride + k] * s");
}

void test_gelu_mul() {
  const int n = 1000;
  const std::size_t n_n = std::size_t(n);
  std::vector<float> gate(n_n), up(n_n), want(n_n);
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> dist(-9.0f, 9.0f);
  for (int i = 0; i < n; ++i) {
    gate[size_t(i)] = dist(rng);
    up[size_t(i)] = dist(rng);
    want[size_t(i)] = gelu_ref(gate[size_t(i)]) * up[size_t(i)];
  }
  DeviceBuf<float> dg, du, dy;
  check(dg.alloc(n) && dg.upload(gate) && du.alloc(n) && du.upload(up) &&
            dy.alloc(n),
        "gelu_mul alloc");
  void* garg = dg.ptr;
  void* uarg = du.ptr;
  void* yarg = dy.ptr;
  int narg = n;
  void* args[] = {&garg, &uarg, &yarg, &narg};
  const int grid = blocks_for(n, compiler::kGeluMulBlockThreads);
  check(glue::launch("gelu_mul", args, grid,
                              compiler::kGeluMulBlockThreads, 0),
        "gelu_mul launch");
  check(hipDeviceSynchronize() == hipSuccess, "gelu_mul sync");
  std::vector<float> got;
  check(dy.download(got), "gelu_mul download");
  double worst = 0.0;
  for (int i = 0; i < n; ++i)
    worst = std::max(worst, rel_err(got[size_t(i)], want[size_t(i)]));
  check(worst <= 1e-3, "gelu_mul matches gelu_fp16(gate) * up");
  if (worst > 1e-3) std::cerr << "    worst rel err = " << worst << "\n";
}

void test_rms_norm_add() {
  const int nb = 2, dim = 64;
  const float eps = 1e-5f, scale = 1.0f;
  const std::size_t row_n = std::size_t(nb * dim), dim_n = std::size_t(dim);
  std::vector<float> x(row_n), r(row_n), w(dim_n), want(row_n);
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : x) v = dist(rng);
  for (auto& v : r) v = dist(rng);
  for (auto& v : w) v = 0.5f + 0.5f * std::fabs(dist(rng));
  for (int row = 0; row < nb; ++row) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i)
      ss += double(x[size_t(row) * dim + i]) * double(x[size_t(row) * dim + i]);
    const float inv = 1.0f / std::sqrt(float(ss / dim) + eps);
    for (int i = 0; i < dim; ++i)
      want[size_t(row) * dim + i] =
          r[size_t(row) * dim + i] + x[size_t(row) * dim + i] * inv * w[size_t(i)];
  }
  DeviceBuf<float> dx, dr, dw, dy;
  check(dx.alloc(nb * dim) && dx.upload(x) && dr.alloc(nb * dim) && dr.upload(r) &&
            dw.alloc(dim) && dw.upload(w) && dy.alloc(nb * dim),
        "rms_norm_add alloc");
  void* xarg = dx.ptr;
  void* rarg = dr.ptr;
  void* warg = dw.ptr;
  void* yarg = dy.ptr;
  int dimarg = dim;
  float earg = eps, sarg = scale;
  void* args[] = {&xarg, &rarg, &warg, &yarg, &dimarg, &earg, &sarg};
  check(glue::launch("rms_norm_add", args, nb,
                              compiler::kRmsNormAddBlockThreads,
                              compiler::kRmsNormAddSharedBytes),
        "rms_norm_add launch");
  check(hipDeviceSynchronize() == hipSuccess, "rms_norm_add sync");
  std::vector<float> got;
  check(dy.download(got), "rms_norm_add download");
  double worst = 0.0;
  for (size_t i = 0; i < want.size(); ++i)
    worst = std::max(worst, rel_err(got[i], want[i]));
  check(worst <= 1e-3, "rms_norm_add matches r + x * inv * w");
  if (worst > 1e-3) std::cerr << "    worst rel err = " << worst << "\n";
}

void test_heads_norm_rope_cast() {
  const int heads = 2, dim = 64;
  const float eps = 1e-5f;
  const std::size_t all_n = std::size_t(heads * dim), dim_n = std::size_t(dim);
  std::vector<float> x(all_n), w(dim_n), want(all_n);
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : x) v = dist(rng);
  for (auto& v : w) v = 0.5f + 0.5f * std::fabs(dist(rng));
  for (int h = 0; h < heads; ++h) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i)
      ss += double(x[size_t(h) * dim + i]) * double(x[size_t(h) * dim + i]);
    const float inv = 1.0f / std::sqrt(float(ss / dim) + eps);
    for (int i = 0; i < dim; ++i)
      want[size_t(h) * dim + i] = x[size_t(h) * dim + i] * inv * w[size_t(i)];
  }
  DeviceBuf<float> dx, dw, dy;
  DeviceBuf<uint64_t> dpos;
  check(dx.alloc(heads * dim) && dx.upload(x) && dw.alloc(dim) && dw.upload(w) &&
            dy.alloc(heads * dim) && dpos.alloc(1),
        "heads_norm_rope_cast alloc");
  const uint64_t pos = 0;
  check(hipMemcpy(dpos.ptr, &pos, sizeof(pos), hipMemcpyHostToDevice) == hipSuccess,
        "heads_norm_rope_cast pos upload");
  void* xarg = dx.ptr;
  void* warg = dw.ptr;
  int haswarg = 1;
  void* yarg = dy.ptr;
  void* posarg = dpos.ptr;
  int dimarg = dim;
  float earg = eps, fsarg = 1.0f;
  float lbarg = std::log(10000.0f);
  void* ffarg = nullptr;
  int hpparg = 1, ropearg = 0, castarg = 0;
  void* carg = nullptr;
  int slotsarg = 0, rowarg = 0, storearg = 0;
  void* args[] = {&xarg,    &warg,    &haswarg, &yarg,   &posarg,   &dimarg,
                  &earg,    &fsarg,   &lbarg,   &ffarg,  &hpparg,   &ropearg,
                  &castarg, &carg,    &slotsarg, &rowarg, &storearg};
  check(glue::launch("heads_norm_rope_cast", args, heads,
                              compiler::kHeadsNormRopeCastBlockThreads,
                              compiler::kHeadsNormRopeCastSharedBytes),
        "heads_norm_rope_cast launch");
  check(hipDeviceSynchronize() == hipSuccess, "heads_norm_rope_cast sync");
  std::vector<float> got;
  check(dy.download(got), "heads_norm_rope_cast download");
  double worst = 0.0;
  for (size_t i = 0; i < want.size(); ++i)
    worst = std::max(worst, rel_err(got[i], want[i]));
  check(worst <= 1e-3, "heads_norm_rope_cast (no rope, no cast) matches x * inv * w");
  if (worst > 1e-3) std::cerr << "    worst rel err = " << worst << "\n";
}

void test_rms_norm_quantize_q8_1() {
  const int dim = 2048;  // 64 q8_1 blocks, a decode-width row
  const float eps = 1e-5f;
  const std::size_t dim_n = std::size_t(dim);
  std::vector<float> x(dim_n), w(dim_n);
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : x) v = dist(rng);
  for (auto& v : w) v = 0.5f + 0.5f * std::fabs(dist(rng));

  struct Block {
    float d;
    int8_t qs[32];
  };
  std::vector<Block> want(size_t(dim / 32));
  double ss = 0.0;
  for (int i = 0; i < dim; ++i) ss += double(x[size_t(i)]) * double(x[size_t(i)]);
  const float inv = 1.0f / std::sqrt(float(ss / dim) + eps);
  for (int b = 0; b < dim / 32; ++b) {
    float amax = 0.0f;
    float v[32];
    for (int i = 0; i < 32; ++i) {
      v[i] = x[size_t(b * 32 + i)] * inv * w[size_t(b * 32 + i)];
      amax = std::max(amax, std::fabs(v[i]));
    }
    want[size_t(b)].d = amax / 127.0f;
    for (int i = 0; i < 32; ++i)
      want[size_t(b)].qs[i] = int8_t(std::lround(v[i] / want[size_t(b)].d));
  }

  DeviceBuf<float> dx, dw;
  DeviceBuf<Block> dy;
  check(dx.alloc(dim) && dx.upload(x) && dw.alloc(dim) && dw.upload(w) &&
            dy.alloc(dim / 32),
        "rms_norm_quantize_q8_1 alloc");
  void* xarg = dx.ptr;
  void* warg = dw.ptr;
  void* yarg = dy.ptr;
  int narg = dim;
  float earg = eps;
  void* args[] = {&xarg, &warg, &yarg, &narg, &earg};
  check(glue::launch("rms_norm_quantize_q8_1", args, 1,
                              compiler::kGlueBlockThreads,
                              compiler::kGlueDynamicSharedBytes),
        "rms_norm_quantize_q8_1 launch");
  check(hipDeviceSynchronize() == hipSuccess, "rms_norm_quantize_q8_1 sync");
  std::vector<Block> got;
  check(dy.download(got), "rms_norm_quantize_q8_1 download");
  double worst = 0.0;
  for (int b = 0; b < dim / 32; ++b) {
    check(got[size_t(b)].d > 0.0f, "rms_norm_quantize_q8_1 block scale is positive");
    int amax = 0;
    for (int i = 0; i < 32; ++i) {
      const int q = got[size_t(b)].qs[i];
      amax = std::max(amax, std::abs(q));
      worst = std::max(worst,
                       rel_err(double(q) * double(got[size_t(b)].d),
                               double(want[size_t(b)].qs[i]) *
                                   double(want[size_t(b)].d)));
    }
    check(amax == 127, "rms_norm_quantize_q8_1 saturates a ±127 quant");
  }
  check(worst <= 1e-2, "rms_norm_quantize_q8_1 dequantizes to the normed row");
  if (worst > 1e-2) std::cerr << "    worst dequant rel err = " << worst << "\n";
}

void test_add_rms_norm_quantize_mmq() {
  const int nb = 2, dim = 128;  // one mmq block per token
  const int ds = 0;             // D4 layout (float d4)
  const float eps = 1e-5f;
  const std::size_t row_n = std::size_t(nb * dim), dim_n = std::size_t(dim);
  std::vector<float> a(row_n), b(row_n), w(dim_n);
  std::mt19937 rng(6);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : a) v = dist(rng);
  for (auto& v : b) v = dist(rng);
  for (auto& v : w) v = 0.5f + 0.5f * std::fabs(dist(rng));

  struct BlockMmq {
    float d4[4];
    int8_t qs[128];
  };
  DeviceBuf<float> da, db, dw, dres;
  DeviceBuf<BlockMmq> dy;
  check(da.alloc(nb * dim) && da.upload(a) && db.alloc(nb * dim) && db.upload(b) &&
            dw.alloc(dim) && dw.upload(w) && dres.alloc(nb * dim) &&
            dy.alloc(nb),
        "add_rms_norm_quantize_mmq alloc");
  void* aarg = da.ptr;
  void* barg = db.ptr;
  void* warg = dw.ptr;
  void* yarg = dy.ptr;
  void* rarg = dres.ptr;
  int dimarg = dim, narg = nb, dsarg = ds;
  float earg = eps;
  void* args[] = {&aarg,  &barg, &warg,   &yarg,     &rarg,
                  &dimarg, &narg, &dsarg, &earg};
  check(glue::launch("add_rms_norm_quantize_mmq_q8_1", args, nb,
                              compiler::kAddRmsNormBlockThreads,
                              compiler::kAddRmsNormSharedBytes),
        "add_rms_norm_quantize_mmq launch");
  check(hipDeviceSynchronize() == hipSuccess, "add_rms_norm_quantize_mmq sync");

  std::vector<float> res;
  check(dres.download(res), "add_rms_norm_quantize_mmq residual download");
  // Bit-exact: the residual is the kernel's own f32 add, so the reference must
  // be the fp32 sum, not the double sum of the two fp32 operands.
  bool exact = true;
  for (size_t i = 0; i < res.size(); ++i)
    exact = exact && res[i] == a[i] + b[i];
  check(exact, "add_rms_norm_quantize_mmq residual is a + b");

  std::vector<BlockMmq> got;
  check(dy.download(got), "add_rms_norm_quantize_mmq download");
  for (int t = 0; t < nb; ++t) {
    check(got[size_t(t)].d4[0] > 0.0f,
          "add_rms_norm_quantize_mmq block scale is positive");
    int amax = 0;
    for (int i = 0; i < 128; ++i)
      amax = std::max(amax, std::abs(int(got[size_t(t)].qs[i])));
    check(amax > 0, "add_rms_norm_quantize_mmq wrote quants");
  }
}

// Every region must compile to an ELF HSA code object and load with its entry
// symbol present.
void test_codegen_and_load(const int arch) {
  using compiler::GlueRegionKind;
  const GlueRegionKind kinds[] = {
      GlueRegionKind::RmsNormQuantizeQ8_1,
      GlueRegionKind::AddRmsNormQuantizeMmqQ8_1,
      GlueRegionKind::RmsNormAdd,
      GlueRegionKind::HeadsNormRopeCast,
      GlueRegionKind::SliceScale,
      GlueRegionKind::GeluMul,
  };
  for (GlueRegionKind kind : kinds) {
    const char* region = compiler::glue_region_name(kind);
    auto hsaco = compiler::generate_glue_hsaco(kind, arch);
    check(hsaco.has_value(), region);
    if (!hsaco) continue;
    check(hsaco->size() > 64 &&
              std::memcmp(hsaco->data(), "\x7f" "ELF", 4) == 0,
          "HSACO is an ELF object");
    hipModule_t module = nullptr;
    check(hipModuleLoadData(&module, hsaco->data()) == hipSuccess,
          "hipModuleLoadData");
    if (module == nullptr) continue;
    hipFunction_t function = nullptr;
    check(hipModuleGetFunction(&function, module,
                               compiler::glue_kernel_entry(kind)) == hipSuccess,
          compiler::glue_kernel_entry(kind));
    (void)hipModuleUnload(module);
  }
}

}  // namespace

int main() {
  if (!have_rocm()) {
    std::cerr << "skipped: no AMD device\n";
    return 0;
  }
  const int arch = gfx_arch();
  check(arch > 0, "gfx architecture");
  if (arch <= 0) return 1;
  std::cout << "gfx:" << arch
            << " codegen:" << compiler::glue_codegen_version() << "\n";

  test_codegen_and_load(arch);
  test_rms_norm_quantize_q8_1();
  test_slice_scale();
  test_gelu_mul();
  test_rms_norm_add();
  test_heads_norm_rope_cast();
  test_add_rms_norm_quantize_mmq();

  if (g_failures != 0) {
    std::cerr << "test_glue_hsaco: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "test_glue_hsaco: OK\n";
  return 0;
}
