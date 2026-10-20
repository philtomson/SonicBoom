// Device-gate test (Phase 4).
//
// The runtime used to ask `quant::cuda_available()` — a CUDA-only query —
// before taking any device path, so a ROCm build never entered one even with an
// AMD device present. The vendor-neutral query is now `quant::gpu_available()`
// (aliased in Layer 2 as `nt::gpu_available()`), and the dispatcher
// (MatmulBackend::Cuda) routes to whichever backend the build has.
//
// What this pins down:
//   * the gate agrees with the backend-specific queries on every build
//     (gpu_available() == cuda_available() || hip_available()), so a CPU-only
//     build gates all device paths off and a GPU build gates them on;
//   * with a device present, quant::matvec / quant::matmul through
//     MatmulBackend::Cuda succeed and match the scalar CPU f32 reference, i.e.
//     the dispatcher reaches a working device path instead of refusing.
//
// main() sets SONICBOOM_GEMV_F32=1 so the device path uses f32 activations, the
// only mode whose numbers are comparable to the CPU f32 reference (the default
// q8_1 activations are int8 and only bound the result; test_cuda_quant covers
// those against a q8_1 CPU oracle).
//
// Caveat by construction: this does not prove device-vs-fallback, because both
// implement f32-activation semantics over the same dequant arithmetic and so
// agree to fp32 tolerance. Correctness of the device kernels themselves is
// test_cuda_quant's job (it calls the cuda:: entry points directly).
//
// Skips cleanly (exit 0) when no device is present, mirroring test_cuda.

#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/quant/quantized_tensor.h>
#include <sonicboom/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace sbquant = sonicboom::quant;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

std::vector<std::byte> build_weight(sbquant::QuantType type, uint64_t rows,
                                    uint64_t cols, std::mt19937& rng) {
  const sbquant::QuantTypeInfo* info = sbquant::quant_type_info(type);
  const uint64_t n_blocks =
      rows * ((cols + info->block_size - 1) / info->block_size);
  std::vector<std::byte> bytes(n_blocks * info->bytes_per_block);
  for (auto& b : bytes) b = std::byte(uint8_t(rng()));
  for (uint64_t i = 0; i < n_blocks; ++i) {
    const size_t off = i * info->bytes_per_block;
    if (type == sbquant::QuantType::Q8_0) {
      // Legacy 32-element blocks: d = 1.0 keeps dequantized magnitudes sane.
      bytes[off + 0] = std::byte(0x00);
      bytes[off + 1] = std::byte(0x3C);
    } else {
      // Q4_K/Q6_K-style: super-block d (and dmin) at the head of the block.
      uint16_t d, dmin = 0;
      do { d = uint16_t(rng()); } while (((d >> 10) & 0x1F) == 0x1F);
      if (type != sbquant::QuantType::Q6_K) {
        do { dmin = uint16_t(rng()); } while (((dmin >> 10) & 0x1F) == 0x1F);
      } else {
        d = 0x3C00;  // 1.0
      }
      bytes[off + 0] = std::byte(uint8_t(d & 0xFF));
      bytes[off + 1] = std::byte(uint8_t(d >> 8));
      if (type != sbquant::QuantType::Q6_K) {
        bytes[off + 2] = std::byte(uint8_t(dmin & 0xFF));
        bytes[off + 3] = std::byte(uint8_t(dmin >> 8));
      }
    }
  }
  return bytes;
}

double compare(std::span<const float> got, std::span<const float> ref,
               double tol, bool* ok) {
  *ok = true;
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double denom = std::max(1.0, double(std::fabs(ref[i])));
    const double err = std::fabs(double(got[i]) - double(ref[i])) / denom;
    worst = std::max(worst, err);
    if (err > tol) *ok = false;
  }
  return worst;
}

void test_gate_semantics() {
  const bool nt_want =
      nt::cuda_available() || nt::hip_available();
  check(nt::gpu_available() == nt_want,
        "nt::gpu_available() == cuda_available() || hip_available()");

  const bool q_want = sbquant::cuda_available() || nt::hip_available();
  check(sbquant::gpu_available() == q_want,
        "quant::gpu_available() == cuda_available() || hip_available()");

  // Exactly one of the two backend-specific queries can be true in one tree.
  check(!(nt::cuda_available() && nt::hip_available()),
        "at most one GPU backend is compiled in");
}

void test_dispatcher(sbquant::QuantType type, const char* name, uint64_t rows,
                     uint64_t cols, double tol, std::mt19937& rng) {
  // The device weight cache keys by host pointer + byte size, so keep every
  // weight's backing buffer alive for the whole run.
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(type, rows, cols, rng));
  const std::vector<std::byte>& bytes = keepalive.back();
  sbquant::QuantizedTensor W(type, {cols, rows}, bytes);
  if (!W.valid()) {
    check(false, "build_weight produced an invalid tensor");
    return;
  }

  std::vector<float> x(cols), y(rows), y_ref(rows);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  const bool ok = sbquant::matvec(W, x, y, sbquant::MatmulBackend::Cuda);
  check(ok, "dispatched Cuda matvec returns true");
  if (!ok) return;

  check(sbquant::matvec(W, x, y_ref, sbquant::MatmulBackend::CpuF32),
        "dispatched CpuF32 matvec returns true");

  bool cmp = false;
  const double worst = compare(y, y_ref, tol, &cmp);
  check(cmp, "dispatched Cuda matvec matches the CPU f32 reference");
  if (!cmp)
    std::cerr << "    worst rel err (" << name << ") = " << worst << "\n";

  // Batched form: Y = W @ X for n = 3 positions.
  constexpr uint64_t n = 3;
  std::vector<float> X(cols * n), Y(rows * n), Y_ref(rows * n);
  for (auto& v : X) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  check(sbquant::matmul(W, X, Y, n, sbquant::MatmulBackend::Cuda),
        "dispatched Cuda matmul returns true");
  check(sbquant::matmul(W, X, Y_ref, n, sbquant::MatmulBackend::CpuF32),
        "dispatched CpuF32 matmul returns true");
  bool cmp2 = false;
  const double worst2 = compare(Y, Y_ref, tol, &cmp2);
  check(cmp2, "dispatched Cuda matmul matches the CPU f32 reference");
  if (!cmp2) std::cerr << "    worst rel err (" << name << ", batched) = "
                       << worst2 << "\n";
}

} // namespace

int main() {
  // The dispatched device path must use f32 activations to be comparable with
  // the f32 CPU reference; the default q8_1 activations are int8 (see
  // test_cuda_quant, which also sets this).
  setenv("SONICBOOM_GEMV_F32", "1", 1);

  test_gate_semantics();

  if (!sbquant::gpu_available()) {
    std::cout << "no GPU device present; dispatcher checks skipped\n";
    return g_failures == 0 ? 0 : 1;
  }

  std::cout << "GPU backend present: "
            << (nt::cuda_available() ? "CUDA" : "ROCm/HIP") << "\n";

  std::mt19937 rng(1234);
  // 4 blocks/row: fp32 dot over 4*256 terms reproduces the reference closely.
  test_dispatcher(sbquant::QuantType::Q8_0, "Q8_0", 64, 32 * 4, 1e-3, rng);
  test_dispatcher(sbquant::QuantType::Q4_K, "Q4_K", 64, 256 * 4, 1e-3, rng);
  test_dispatcher(sbquant::QuantType::Q6_K, "Q6_K", 64, 256 * 4, 1e-3, rng);

  if (g_failures != 0) std::cerr << g_failures << " failure(s)\n";
  return g_failures == 0 ? 0 : 1;
}
