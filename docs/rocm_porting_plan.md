# ROCm / Strix Halo Porting Plan

Status: **functional path complete** (branch `amd_dev`) — Phases 0-4 done: the
hand-written kernels, the device-resident forward, the planner spine (graph
capture/replay through the HIP twin) and the generated decode/prefill attention
and dense bf16 GEMM all run on AMD. What is not closed is verification beyond
each unit (no GGUF fixture here, so the token-level spine tests skip) and the
tensor-core optimization (MFMA).
`origin/master` merged in
(bb8e43f): upstream's training/backward work is CUDA-gated in its host files,
so the ROCm build takes it as host-only code; two conflicts (NVTX include,
test registration) and one device-path fix in `training/param.cpp`.
Target GPU: **AMD Radeon 8060S** (Strix Halo iGPU), **gfx1151** (RDNA 3.5,
40 CUs). ROCm **7.x** (TheRock 10.1.0) at `/opt/rocm` (`hipcc`, `hipconfig`,
`rocminfo`). `/dev/kfd` and `/dev/dri/renderD128` present; user is in the
`render` group, so the HSA/KFD device is usable.

## Decisions (recorded)

- **Approach:** direct parallel HIP port (`SONICBOOM_USE_ROCM` alongside
  `SONICBOOM_USE_CUDA`), not the full HAL plugin split first (per
  `design/backend-hal.md`). Keep the CUDA path untouched and gated.
- **Kernel scope:** functional AMD path first (portable glue regions +
  non-tensor-core attention/GEMM), optimize to MFMA/WMMA later.
- **LLVM/MLIR:** rebuild the project LLVM/MLIR with the `AMDGPU` target added
  to `LLVM_TARGETS_TO_BUILD`.

## Current CUDA hard-wiring (what must be ported)

1. **Hand-written kernels (nvcc):**
   - `core/src/nn/cuda_elementwise.cu`
   - `core/src/quant/cuda_quant_matmul.cu`
   - `core/src/quant/ggml_derived/mmq.cuh`, `mmvq_q8_1.cuh`
2. **MLIR JIT kernels** (`core/src/planner/glue_jit.cpp`): generated as PTX
   via `#nvvm.target<chip="sm_86">>` + `nvgpu.ldmatrix`,
   `nvgpu.mma.sync`, `nvvm.cp.async.*`; loaded with `cuModuleLoadData` +
   `cuLaunchKernel`.
3. **Host orchestration:** `core/src/planner/sonic_cuda_backend.cpp`,
   `core/src/model/cuda_block.cpp`, `core/src/planner/device_pool.cpp`,
   `core/src/nn/cuda_stream.h`, `core/src/nn/cuda_graph.h` — raw
   `cuda_runtime.h` alloc/memcpy/streams/graph-capture.
4. **Layer-2 public API** hard-codes CUDA:
   - `core/include/sonicboom/device.h` — `DeviceType::GPU`, `Device::gpu()`
     (renamed from `CUDA` / `cuda()`; the nt enum is now vendor-neutral)
   - `core/include/sonicboom/backend.h` — `BackendId::gpu()` (same rename)
   - `core/include/sonicboom/tensor.h` — `cuda_available()` stays (vendor
     query, paired with the neutral `gpu_available()`)
   - `core/include/sonicboom/quant/quantized_matmul_cuda.h` — `quant::cuda`
     (vendor-specific kernel namespace, paired with `quant::gpu_available()`)
   - `core/src/planner/sonic_cuda_backend.h` — `SonicCudaBackend` (moved out
     of the public include tree; it was never a Layer-2 surface)
   - `core/layer1/adapter/{device,backend,tensor}.cpp` — `DispatchKey::CUDA`
     (still the native mapping for the GPU backend on this build)
   - `core/CMakeLists.txt` CUDA block; `CMakeLists.txt`
     `SONICBOOM_USE_CUDA`; `CMAKE_CUDA_ARCHITECTURES=86`
5. **Build tooling:** `tools/mlir/build-mlir.sh` builds LLVM targets
   `$HOST_TARGET;NVPTX` only (LLVM/MLIR submodule not checked out).

## Phases

### Phase 0 — Toolchain & build skeleton
- [x] Check out `third_party/llvm-project` (shallow, pinned commit).
- [x] Add `AMDGPU` to `LLVM_TARGETS_TO_BUILD` in `tools/mlir/build-mlir.sh`;
      build LLVM/MLIR with AMDGPU (done, installed). Verified: `libLLVMAMDGPU*`
      codegen libs + `libMLIRAMDGPU*`/`libMLIRGPUToROCDL*`/
      `libMLIRMathToROCDL` dialect/conversion libs + headers present.
- [x] Add `option(SONICBOOM_USE_ROCM ...)` and `find_package(hip)`;
      `CMAKE_HIP_ARCHITECTURES=gfx1151`. Configure with `USE_ROCM=ON`
      successfully detects HIP (clang 24, hipcc) and MLIR.
- [x] Smoke-test a HIP kernel on gfx1151 (hipcc, `--offload-arch=gfx1151`):
      `add` kernel PASS on AMD Radeon 8060S.
- [!] **Blocker:** a full SonicBoom configure (any backend) additionally requires
      SonicCross's `guild` codegen tool
      (`tools/generation/GenerateNativeTorch.cmake:16`); it is not installed on
      this machine. Install it (or pass `-DSONICCROSS_GUILD=...`) before any
      end-to-end build can configure. This is a pre-existing prerequisite,
      independent of ROCm.
- [x] **Resolved:** built SonicCross v0.0.2 (`e3acee5`) locally to
      `~/.local` (needs `guile30`+`guile30-devel`; libyaml via a locals
      `~/.local/lib/libyaml.so` symlink; Guile site-dir overridden via a local
      `pkg-config` file). `guild` wrapper installed at `~/.local/bin/guild`.
      Cheked out the native-torch submodules (cpuinfo/sleef/FP16/fmt).
- [x] Configure with `-DSONICBOOM_USE_ROCM=ON` succeeds (SonicCross generates
      `7063` per-op headers + `54` Register*.cpp).
- [x] `cmake --build build-rocm --target sonicboom` succeeds -> `libsonicboom.so`
      with ROCm enabled. Two pre-existing CUDA-off build fixes applied:
      `core/src/planner/autotune.cpp` (guard `median_launch_ms` stream overload
      behind `SONICBOOM_USE_CUDA` to match its header) and
      `core/src/planner/device_pool.cpp` (guard `cuda_stream.h` include). These
      are general CUDA-off fixes, not ROCm-specific.

### Phase 1 — Hand-written kernels + host runtime (mechanical)
- [x] `.cu -> .hip` via ROCm `hipify-perl`, then hand-fixed:
      `hip_elementwise.hip` (twin of `cuda_elementwise.cu`), `hip_stream.h`
      (twin of `cuda_stream.h`), `hip_quant_matmul.hip` (twin of
      `cuda_quant_matmul.cu`, pre-existing) + `hip_intrinsics.h` (emulated
      `__vsub4`/`__dp4a` for AMD).
- [x] Host API map: `cuda*` -> `hip*`; cuBLAS/cuBLASLt -> hipBLAS/rocBLAS
      (HIP path uses `hipblasGemmEx` fallback; hipBLASLt block guarded out for
      ROCm). `hip_stream.h` provides `sb_current_stream()`/`StreamScope`/
      `CaptureScope`.
- [x] Build-system fixes for the HIP TU: add ROCm include root to HIP compile
      flags; `hipblas/hipblas.h` subpath includes; `#if !defined(USE_ROCM)`
      guards around tensor-core flash attention (`fa_mma` helpers + tiled
      kernel) and hipBLASLt helpers; `launch_tiled_flash_attention` returns
      false on ROCm (caller falls back to scalar chunked kernel).
- [x] HIP 64-bit shfl mask fix (`0xFFFFFFFFFFFFFFFFull`) in `mmq.cuh`/
      `mmvq_q8_1.cuh`; fixed a void* pointer-arithmetic in `zero_pad_planes`.
- [x] `planner/device_pool.cpp` made backend-agnostic (CUDA under
      `SONICBOOM_USE_CUDA`, HIP under `SONICBOOM_USE_ROCM`); `nn/hip_graph.h`
      (SIP twin of cuda_graph.h); `planner/autotune.{h,cpp}` HIP stream overload.
- [x] `libsonicboom.so` builds+links against libamdhip64/libhsa with both HIP
      kernel TUs (elementwise + quant), ~248 benign `nodiscard` warnings.
- [x] Host orchestration made backend-agnostic (CUDA/HIP `#if`):
      `planner/device_pool.cpp`, `planner/sonic_cuda_backend.cpp` (GraphStep/
      capture-scope/`stage_upload` memcpy behind an `sbstream` alias; NVTX
      stubbed on HIP), `model/cuda_block.cpp` (alloc/memcpy via `SB_DEV_*`
      macros). `nn/hip_graph.h` adds the HIP `GraphStep`.
- [x] `nt::hip_available()` added (HIP device count) alongside `cuda_available()`.
- [x] Link ROCm BLAS: `find_package(hipblas/rocblas CONFIG)` +
      `roc::hipblas`/`roc::rocblas`; the shared lib's hipblas symbols now resolve
      for consumers. **Later replaced** by weak hipBLAS refs (see the dual-LLVM
      fix) so no rocBLAS/hipBLAS linkage is required.
- [x] Runtime device smoke: a consumer binary linking the built
      `libsonicboom.so` reports `hip_available=1` on the Radeon 8060S iGPU
      (`cuda_available=0`).
- [!] Note: the shared library has a pre-existing teardown double-free on
      process exit (reproducible with an empty `main` that just links the lib,
      in the `thread_pool` static state) — unrelated to the ROCm port; not
      observed on the normal runtime path.
- [x] **Dual-LLVM conflict (resolved):** linking rocBLAS/hipBLAS (transitively
      via libamd_comgr) pulled ROCm's libLLVM/libclang-cpp (v24) into the same
      process as the statically-linked project LLVM (v23) that the MLIR path
      uses. Two LLVM global ctor/dtor sets + interposed symbols caused clang
      option-parse failures (`--amdhsa-code-object-version` etc.) and heap
      corruption (double-free). Fixed two ways:
      - dropped the `roc::hipblas`/`roc::rocblas` linkage; the HIP kernels now
        use **weak hipBLAS references** (`core/src/nn/hip_blas.h`); when no
        libhipblas is present they fall back to the own gemv kernels;
      - `-Wl,--exclude-libs,ALL` on `libsonicboom.so` keeps the project LLVM /
        native-torch symbols internal so they can't interpose onto ROCm's LLVM
        (public core/C-API symbols remain exported). Verified: no more clang
        option errors, no teardown double-free.
- [x] **Phase 1 numeric verification on the Radeon 8060S iGPU** (ctest):
      `test_cuda_elementwise` OK, `test_cuda_quant` OK (MMQ/cuBLAS-only
      assertions gated out on ROCm via `#if !defined(SONICBOOM_USE_ROCM)`),
      `test_cuda_gather` OK, `test_cuda` skips (CUDA-specific Layer-2 path).
      CPU-core regression tests (`test_nn`, `test_dequant`, `test_tensor`,
      `test_backend`, `test_quantized_matmul`) still pass.
- [x] **Full C++ suite on the ROCm build:** `ctest -E guile` = **74/74 pass**
      (5 expected SKIPs for absent model fixtures; 1 failure until the Guile
      prerequisite below is met). The Guile tests need the `(json)` module —
      `guile-json` is **not** packaged in Fedora, so it was built from source
      (`https://download.savannah.gnu.org/releases/guile-json/guile-json-4.7.3.tar.gz`,
      `./configure --prefix=$HOME/.local && make && make install`) and the
      Guile tests run with `GUILE_LOAD_PATH=$HOME/.local/share/guile/site/3.0`.
      With it present, the full suite is 74/74 and the Guile subset 14/14 on
      the ROCm build. Not related to ROCm: the same module is missing on any
      checkout here.
- [x] Kernel-level numeric verification on the iGPU (elementwise + quant
      matvec/mmq vs CPU reference) via `test_cuda_elementwise` /
      `test_cuda_quant` / `test_cuda_gather`, which run on the ROCm build.
      End-to-end (model load + decode) is still not exercised on AMD: every
      device-gated test gates on `cuda_available()`, see Phase 4.
- [x] `#undef __noinline__` after HIP includes in `autotune.h`/`plan_hook.h`
      (ROCm defines it as a macro, breaking libstdc++ 16 `<format>`).
- [x] **Build: `libsonicboom.so` links cleanly with ROCm enabled** (HIP
      kernels compiled for gfx1151; links libamdhip64 + libhsa-runtime64;
      exports hipGetDeviceCount/hipMalloc/hipblasCreate/hipLaunchKernel).
- [x] Runtime verification: elementwise + quantized matvec/mmq run on the iGPU
      (`ctest -E guile` = 67/67 pass on the ROCm build), CPU fallback intact.
      The model-level runtime path (planner + glue regions over a real GGUF) is
      still unverified on AMD — the remaining `cuda_available()` gates are the
      blocker, see Phase 4.

### Phase 2 — JIT glue regions (first codegen win)
All six decode-step regions (`rms_norm_quantize_q8_1`,
`add_rms_norm_quantize_mmq_q8_1`, `rms_norm_add`, `heads_norm_rope_cast`,
`slice_scale`, `gelu_mul`) now generate, load and run on gfx1151. Checked by
`tests/core/test_glue_hsaco.cpp` (compiles each region to an HSACO, loads it
with `hipModuleLoadData`, launches it and compares against CPU references;
skips when no AMD device is present).
- [x] gfx-arch query (`gcnArchName` -> 1151) and cache key `gfx1151`; the disk
      cache is shared with the PTX path (`<region>-gfx<arch>.hsaco` from
      `<region>-sm<cc>.ptx`), stale bodies rejected by the version/hash header.
- [x] `#nvvm.target<chip="sm_86">` -> `#rocdl.target<chip="gfx1151">`; the
      region text is shared and only the target attribute + the `nvvm.*`
      intrinsics are rewritten (`nvvm.sqrt` -> `math.sqrt`,
      `nvvm.addf {rnd=rz}` -> `llvm.intr.round`), then lowered through ROCDL and
      serialized by the AMDGPU backend to an HSA code object.
      Codegen version is now `llvm-<ver>-glue4` (the AMD text changed).
- [x] `cuModuleLoadData(PTX)` -> `hipModuleLoadData(HSACO)`, and
      `hipModuleLaunchKernel` rather than `hipLaunchKernel`: HIP's
      `hipLaunchKernel` takes a host kernel symbol, and these come from a loaded
      module. Launched on the planner's current stream so the launch stays
      capturable into a HIP graph, like the PTX path.
- [x] Wave32 pinned via the target attribute (`flags = {no_wave64}`), matching
      the regions' 32-wide shuffle/reduction trees.
- [ ] Byte-identity vs CPU where possible: `slice_scale` and the MMQ residual
      are exact today; the norm/quantize/RoPE and gelu regions are checked at
      1e-3 (the fp16 round-trips and the ocml trig/exp leave the CUDA result's
      last bit behind). Tightening this is a follow-up, not a blocker.

### Phase 3 — Attention + dense GEMM (functional first)
- [x] **Generated decode attention (the `fa_decode`/`fa_combine` pair) runs on
      AMD.** The decode kernels are target-neutral (32-lane `gpu.shuffle`,
      `gpu.barrier`, `math.exp`, f32 fma), so the same text retargets cleanly:
      `amdgpu_adapt_module_text` (compiler/GlueCodegen) swaps the NVVM target
      for `#rocdl.target<chip="gfx1151", flags={no_wave64}>`, declares each
      `gpu.func`'s block size (`fa_decode` 32, `fa_combine` head_dim) and
      applies the `nvvm.*` rewrites; `generate_attention_hsaco` lowers it
      through the shared ROCDL pipeline, and `glue_jit` caches/loads/launches
      the HSACO (`hipModuleLaunchKernel`, same geometry). The glue regions now
      share that one retargeting helper.
      `tests/core/test_hip_attention_decode_codegen.cpp` runs the real
      `nn::cuda::decode_attention_dev` entry with `SONICBOOM_ATTN_DECODE=codegen`
      over d64/d256/d512 (short context, full window, slot wrap-around,
      GQA 7): within fp64 tolerance, not worse than the hand-written split-KV,
      deterministic, and **bit-identical to it at the default chunk** — so the
      generated pair is the default on AMD too, as on CUDA. Full suite 75/75.
      It is also faster here (ad-hoc hipEvent probe, 300 iters, gfx1151):
      d256/512 keys 15.3 us/call vs 47.3 for the hand-written split-KV;
      d512/1024 keys 26.1 vs 107.9. The hand-written pair's blocks share a KV
      group across query heads but expose far fewer blocks; on the 40-CU iGPU
      the generated one-warp-per-(head, chunk) grid wins.
- [x] **Generated prefill attention (the `fa` region) runs on AMD.** The
      NVIDIA body is tensor-core (`nvgpu.ldmatrix` / `nvgpu.mma.sync` /
      `nvvm.cp.async`, f16 staging) and has no ROCDL lowering, so
      `AttentionCodegen.cpp` now has two emitters behind one ABI:
      `attention_region_text` (unchanged) on CUDA and
      `attention_prefill_region_text_amd` on ROCm. The AMD body is scalar f32:
      four warps per block, one query row per warp, `RPW = ceil(br/4)` rows per
      block, a per-key loop with a lane-strided fma dot and a **butterfly
      (`gpu.shuffle xor`) reduction** — unlike decode, every lane needs the
      score for its own softmax state and P·V columns, so a shuffle-down tree
      is wrong here (it was caught by the test immediately) — then online
      softmax (`math.exp`, fma) and the same masked key range, batch/cache
      `llvm.select` and slot wrap as the CUDA body. No LDS and no barriers:
      `attention_shared_bytes` returns 0, `attention_block_threads` 128 for
      prefill (decode stays 32), and `attention_shape_supported` accepts any
      `head_dim` that is a multiple of the wave (32) up to 16 elements per
      lane. A warp-head pin is rejected (falls back to the hand-written
      kernel). The same text retargets through `amdgpu_adapt_module_text` and
      the ROCDL pipeline as the decode pair. The HIP caller
      (`nn/hip_elementwise.hip`) previously never reached any generated prefill
      kernel: `launch_tiled_flash_attention` returns false on ROCm and the
      attempts were not guarded, so d256/d512 prefill silently fell through to
      the hand-written scalar kernel while d64 returned nothing. Those
      attempts are now `#if !defined(SONICBOOM_USE_ROCM)`, so all shapes take
      the scalar path.
      `tests/core/test_hip_attention_prefill_codegen.cpp` runs the real
      `nn::cuda::flash_attention_chunk_dev_pos` entry with
      `SONICBOOM_ATTN_IMPL=codegen` (d256 window short/full, d256 causal and
      slot wrap, d512, d64 GQA 7), checks the fp64 reference, requires the
      generated error to be no worse than 2× the hand-written scalar kernel's,
      asserts determinism and probes the `.hsaco` cache for the `fa` entry.
      Numerics land at 1.1–2.8e-5 relative, the same class as the hand-written
      kernel. Full suite 89/89, 5 expected skips.
      Perf is not benchmarked and no default is forced: the only ROCm prefill
      alternative is the hand-written scalar warp kernel, which does the same
      work with the same grid granularity, and the CUDA tiled kernel (the one
      the old note compared against) does not exist on ROCm. d256/d512 keep
      their static `Codegen` default, d64 keeps `Own`; because
      `attn_launch_own` now works on ROCm, the runtime tuner can compare both
      when `SONICBOOM_ATTN_CODEGEN` is on.
- [x] **Generated dense bf16 GEMM runs on AMD.** As with attention, the
      NVIDIA body is tensor-core-only (`nvgpu.ldmatrix`/`mma.sync`, `cp.async`)
      and `generate_dense_gemm_ptx` was the only path, so
      `launch_dense_gemm` was `#if SONICBOOM_USE_CUDA` and returned false on
      ROCm. `GemmCodegen.cpp` now has a second emitter for the backend switch:
      one BK slab of W and X per K step is staged cooperatively in shared
      memory (scalar loads, zero fill outside `m`/`k`/`n`), then each of the
      256 threads accumulates a 4 x (bm*bn/1024) register tile out of that slab
      with 8-wide bf16 vector loads, `arith.extf` to f32 and
      `vector.reduction <add>`; single-buffered with two `gpu.barrier`s per
      step. Same kernel ABI (memref descriptors + `m`,`k`,`n`,`alpha`), same
      `Y[token][row]` contract and the same exact `alpha * acc` epilogue.
      `dense_gemm_block_threads` returns 256 and `dense_gemm_shared_bytes` one
      slab for the AMD body; `dense_gemm_shape_supported` checks the derived
      4 x btt register tile and the 64 KiB LDS budget instead of the mma tile
      rules. `planner/glue_jit` gained the HSACO cache/load/launch twin, and
      `hip_elementwise.hip` needed no change: its dense path already calls
      `launch_dense_gemm` and falls through to the rocBLAS/gemv arm when the
      generated tile is not chosen.
      `tests/core/test_hip_dense_gemm_codegen.cpp` checks all ten autotune
      candidate tiles plus a tail shape (m=1000, k=1000, n=7), the exact scale
      epilogue, and bitwise batch invariance (a token's row must not depend on
      N or on its block coordinate, which is what lets the planner drop the
      fixed 128-column blocking). Relative error vs fp64 is 8.9e-7..1.2e-6,
      i.e. the f32 accumulation order is not the accuracy limit.
      On this build hipBLAS is reached only through weak symbols
      (`nn/hip_blas.h`, to keep ROCm's LLVM out of the process), so the test's
      rocBLAS baseline arm is skipped when the runtime has not loaded it — and
      in a plain SonicBoom process it is *not* loaded, which means the dense
      tuner's baseline arm fails and the generated tile is selected for every
      shape class. That is the intended AMD dense path; it is a scalar f32
      kernel, so `MFMA` remains the optimization item below.
      Perf is not benchmarked (functional-first); the tuner still times every
      candidate, so a faster body can replace it without touching the planner.
      Full suite 90/90, 5 expected skips.

      Pipeline note, found the hard way: `vector.load` from a **workgroup**
      memref is not converted by `ConvertGpuOpsToROCDLOps` (the standalone
      vector-to-LLVM pass has no workgroup memory-space mapping), so it
      survives to the translation as an unrealized cast and fails with
      "conversion of memref memory space #gpu.address_space<workgroup> failed".
      The body therefore stages with scalar `memref.store` and reads through a
      second `memref.view` of the same slab typed
      `memref<N x (BK+8)/8 x vector<8xbf16>>`, so `memref.load` yields the
      vector directly. `SONICBOOM_DUMP_HSACO_IR=1` prints the fully converted
      module before serialization; the ROCDL failure itself only shows up as a
      serializer diagnostic.
- [ ] Optimize later: `MFMA` (fp16 `MFMA_F32_16x16x16_F16`),
      `ds_read_b64`, `s_waitcnt`/bulk prefetch (follow-up).

### Phase 4 — Layer-2 hygiene + tests
- [x] **Vendor-neutral device gate.** `quant::gpu_available()` and
      `nt::gpu_available()` (`cuda_available() || hip_available()`) answer "is
      there a device" without naming a vendor; `cuda_available()` stays the
      CUDA-specific query (false on a ROCm build). The dispatcher
      (`MatmulBackend::Cuda` in `core/src/quant/quantized_matmul.cpp`) now sits
      behind the two-backend `#if`, so on an AMD build the Cuda backend runs the
      HIP gemv instead of silently falling back to the f32 CPU reference.
      Covered by `tests/core/test_gpu_gate.cpp` (gate semantics across builds +
      dispatched matvec/matmul vs the CPU f32 reference; it runs on the ROCm
      build here).
- [x] **Per-op routing and the device-resident forward on AMD.**
      `core/src/model/exec.cpp` now builds its `*_d` dispatch layer under both
      GPU backends (the HIP twin of `cuda_elementwise.cu` defines the same
      `nn::cuda::*` symbols) and gates it on `quant::gpu_available()`, so a
      `MatmulBackend::Cuda` forward runs rms_norm / rope / gelu / mul / add /
      scale / dense matvec on the iGPU. The device-resident fast path
      (`cuda_forward_resident`, `:438`) is the same file on both backends
      (`model/cuda_block.cpp` allocates via `SB_DEV_MALLOC` and calls the same
      twins), so it is enabled too. Not exercised end-to-end here (no GGUF
      fixture); the kernels themselves are the ones `test_cuda_elementwise` /
      `test_cuda_quant` already cover on this build.
- [x] **Spine + planner GPU enumeration (one unit).** The spine turned out to
      be almost entirely backend-neutral already: its graph capture and stream
      scopes go through `SonicCudaBackend` (`sbstream::GraphStep` /
      `CaptureScope`, `nn/hip_graph.h` / `hip_stream.h` on ROCm), and every
      kernel call is a `nn::cuda` / `quant::cuda` twin the HIP build defines.
      Only three things were CUDA-specific, and none of them needed logic
      changes:
      * the two `#ifdef SONICBOOM_USE_CUDA` guards around the decode and
        prefill spine blocks (`gemma4_plan.cpp:1107`, `:2607`) widened to
        `#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)`; the
        `#else` stubs now apply only to CPU-only builds;
      * NVTX — no-op `nvtxRangePushA`/`nvtxRangePop` shims on ROCm (the same
        choice `planner/sonic_cuda_backend.cpp` already made: the ranges are
        opt-in nsys markers, and nvtx3 is a CUDA-toolkit header);
      * `device_identity()` — `hipGetDeviceProperties` twin reporting
        `"<name> cu <n> <gcnArchName>"` (the CU count is what the split-KV grid
        sizing reads; there is no cc major/minor on ROCm).
      The gates then move to the vendor-neutral query: `generate()` /
      `generate_prompt()` (`core/src/model/exec.cpp`) call
      `quant::gpu_available()`, and `CpuResourceProvider::snapshot` enumerates
      the device behind `cfg.enumerate_gpu` (renamed from `enumerate_cuda`;
      the field is the "this snapshot may place work on a GPU" switch, not a
      vendor one). The enumerated `Device::name` is `"cuda"` / `"hip"` per
      build, because it is hashed into the snapshot fingerprint and a ROCm plan
      must not collide with a CUDA plan.
      `tests/core/test_planner_resource.cpp` follows the same gate;
      `tests/core/test_generate.cpp:171` / `:269` now run their device arms on
      either backend.
      **Verification limit:** the unit tests validate the pieces (kernels,
      generated regions, the backend's own capture path), but no GGUF fixture
      is present here, so the token-level spine tests still skip and the HIP
      spine has not been run end-to-end. Compile + suite green is the evidence
      available in this environment.
- [x] **Vendor-neutral IDs.** `DeviceType::GPU` + `Device::gpu()` (was
      `DeviceType::CUDA` / `Device::cuda()`), `BackendId::gpu()` (was
      `cuda()`), and `SonicCudaBackend`'s header moved from
      `core/include/sonicboom/planner/` to `core/src/planner/` — it is a
      backend implementation detail and never was Layer-2 surface. The adapter
      keeps naming the native GPU device type / dispatch keys: Layer 1 decides
      what "GPU" means, and on this build that is still CUDA. `quant::cuda`
      stays as the vendor-specific kernel namespace, paired with the neutral
      `quant::gpu_available()`.
- [x] **Relax byte-identity asserts where the reference is CUDA-specific.**
      The generated-region tests never compared against CUDA bytes: the ROCm
      twins (`test_hip_attention_{decode,prefill}_codegen`,
      `test_hip_dense_gemm_codegen`) check an fp64 reference with a relative
      tolerance, plus within-run bitwise properties (determinism, batch
      invariance, own-vs-generated consistency), and those all pass. The one
      cross-backend identity assertion is the token stream in `test_generate`:
      it compares against a llama.cpp **CUDA** golden / the CPU q8_K oracle.
      Greedy argmax reproduces that stream on CUDA but is not a numeric
      invariant — ROCm's ocml `expf`, its reduction orders (split-KV atomics)
      and its generated attention differ in the last bits, any of which can
      flip a near-tie. So the strict stream identity stays asserted on a CUDA
      device only; on ROCm the stream is printed ("matches"/"differs") and the
      structural checks (step count, vocab range, routing) stay hard. Same rule
      as the prefill-vs-per-token near-tie check that already existed there.

## Findings worth keeping (Phase 2)

These cost the most time to rediscover; they are properties of the toolchain,
not of this port:
- **Launch bounds.** The ROCDL LLVM IR translation stamps every kernel
  `amdgpu-flat-work-group-size = "1,256"` unless the `llvm.func` already carries
  one, and this LLVM's GPU-to-ROCDL conversion derives *no* work-group size from
  `gpu.known_block_size`. A 1024-thread launch (the q8_1 decode region) is then
  rejected by the loader ("Launch params (1024, 1, 1) are larger than launch
  bounds (256)"). Fix: declare `gpu.known_block_size` on each `gpu.func` in the
  AMD region text and turn it into `rocdl.max_flat_work_group_size` after
  lowering (`core/compiler/lib/GlueCodegen.cpp`).
- **Conversion registration.** The arith and cf `ConvertToLLVMPatternInterface`s
  must be in the dialect registry or `gpu.shuffle`'s lowering is rolled back
  silently (the ops its patterns create are unlegalizable).
- **sqrt.** `math.sqrt` (IEEE-correctly-rounded) is used, not `rocdl.sqrt` (the
  approximate `v_sqrt_f32`); `ConvertMathToROCDL` leaves it to the LLVM math
  conversion, which is marked illegal for the ROCDL pass, so the LLVM math
  conversion must run *after* the ROCDL one. Trig/exp go through ocml.
- **Run-time toolkit dependency.** HSACO linking invokes
  `<rocm>/llvm/bin/ld.lld`, located from `ROCM_PATH` (baked into
  `libMLIRROCDLTarget.a` at LLVM-build time, `/opt/rocm` by default, override at
  run time with `ROCM_PATH`/`ROCM_ROOT`/`ROCM_HOME`). Unlike the PTX path, the
  AMD glue path therefore touches the toolkit at run time.
- **Dual LLVM.** Linking the HIP runtime pulls ROCm's `libLLVM`/`libclang-cpp`
  (v24) in through `libamd_comgr`, next to the statically linked project LLVM
  (v23). Both layers of the fix are needed: no rocBLAS/hipBLAS link (weak refs)
  *and* `-Wl,--exclude-libs,ALL` — anything that links the project LLVM and the
  HIP runtime in one binary needs the same flag (including tests).
- **Shared headers.** `ggml_derived/mmq.cuh` and `mmvq_q8_1.cuh` are compiled by
  both nvcc and hipcc and are kept in the hipify spelling; `quant/hip_compat.h`
  maps the HIP builtin/runtime names onto CUDA for the nvcc build.

## Risks
- LLVM/MLIR rebuild with AMDGPU is the long pole (hours); submodule not expanded.
- Attention/GEMM codegen is NVIDIA-mma-specific and stays that way for CUDA;
  the functional AMD bodies are new code, not a flag flip (Phase 3).
- The AMD bodies are scalar (f32 accumulation, `vector.reduction`); MFMA is the
  performance follow-up, so generated dense GEMM throughput is well below
  rocBLAS. gfx1151's 64 KiB LDS is checked at codegen (the dense body rejects
  tiles that do not fit); attention uses no LDS on AMD.
- Wave width / shared-memory / async-copy semantics differ; wave32 is pinned
  (see Phase 2).
- The HIP spine compiles and links but has not been run end-to-end (no model
  fixture): capture/replay + pooled buffers + the generated decode/prefill
  regions have only been exercised piecewise.
- Native-torch HIP sources exist upstream but are `EXCLUDE` in the migration
  manifest; not needed for v0's own kernels.
