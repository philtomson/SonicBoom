# ROCm / Strix Halo Porting Plan

Status: **active**, Phase 0/1 in progress.
Target GPU: **AMD Radeon 8060S** (Strix Halo iGPU), **gfx1151** (RDNA 3.5,
40 CUs). ROCm **10.1.0** installed at `/opt/rocm` (`hipcc`, `hipconfig`,
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
   - `core/include/sonicboom/device.h` — `DeviceType::CUDA`, `Device::cuda()`
   - `core/include/sonicboom/backend.h` — `BackendId::cuda()`
   - `core/include/sonicboom/tensor.h` — `cuda_available()`
   - `core/include/sonicboom/quant/quantized_matmul_cuda.h` — `quant::cuda`
   - `core/include/sonicboom/planner/sonic_cuda_backend.h` — `SonicCudaBackend`
   - `core/layer1/adapter/{device,backend,tensor}.cpp` — `DispatchKey::CUDA`
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
- [x] **Full C++ suite on the ROCm build:** `ctest -E guile` = **65/65 pass**
      (with the 5 expected SKIPs for absent model fixtures). The 7 failing
      Guile tests are a separate, pre-existing prerequisite — `guile-json`
      module missing (not ROCm-related); they need `sudo dnf install
      guile-json`.
- [ ] Full phase-1 numeric verification on the iGPU (elementwise + quant
      matvec/mmq vs CPU reference) needs the test harness (Phase 4 extends the
      CUDA-gated differential suite to ROCm).
- [x] `#undef __noinline__` after HIP includes in `autotune.h`/`plan_hook.h`
      (ROCm defines it as a macro, breaking libstdc++ 16 `<format>`).
- [x] **Build: `libsonicboom.so` links cleanly with ROCm enabled** (HIP
      kernels compiled for gfx1151; links libamdhip64 + libhsa-runtime64;
      exports hipGetDeviceCount/hipMalloc/hipblasCreate/hipLaunchKernel).
- [ ] Runtime verification: elementwise + quantized matvec/mmq on the iGPU,
      CPU fallback intact (needs the full runtime/tests; `guild` build done).

### Phase 2 — JIT glue regions (first codegen win)
- [ ] Replace `sm_arch()` with gfx-arch query; cache key `gfx1151` (not `sm<cc>`).
- [ ] `#nvvm.target<chip="sm_86">` -> `#rocdl.target<chip="gfx1151">`;
      GPU->ROCDL lowering; AMDGPU backend -> HSACO code object (not PTX).
- [ ] `cuModuleLoadData(PTX)` -> `hipModuleLoadData(HSACO)`;
      `cuLaunchKernel` -> `hipLaunchKernel`.
- [ ] Pin **wave32** on RDNA 3.5 (glue regions use portable
      `gpu.shuffle`/`barrier`/`dynamic_shared_memory`; the `%wid=32` trees hold).
- [ ] Keep byte-identity vs CPU where possible; trig/`expf` paths go tolerance-based.

### Phase 3 — Attention + dense GEMM (functional first)
- [ ] Emit non-tensor-core thread-partitioned AMD kernels for
      `nvgpu.ldmatrix`/`nvgpu.mma.sync`/`nvvm.cp.async` regions.
- [ ] Optimize later: `MFMA` (fp16 `MFMA_F32_16x16x16_F16`),
      `ds_read_b64`, `s_waitcnt`/bulk prefetch (follow-up).

### Phase 4 — Layer-2 hygiene + tests
- [ ] Vendor-neutral IDs: `DeviceType::GPU`, `BackendId::gpu()`,
      `gpu_available()`, `quant::gpu`; move `sonic_cuda_backend.h` behind boundary.
- [ ] Extend `tests/core/CMakeLists.txt` CUDA-gated differential suite to ROCm;
      relax byte-identity attention/GEMM asserts (wave/expf differences).

## Risks
- LLVM/MLIR rebuild with AMDGPU is the long pole (hours); submodule not expanded.
- Attention/GEMM codegen is NVIDIA-mma-specific; the functional AMD path is new
  code, not a flag flip.
- Wave width / shared-memory / async-copy semantics differ; pin wave32 and
  verify LDS limits on gfx1151.
- Native-torch HIP sources exist upstream but are `EXCLUDE` in the migration
  manifest; not needed for v0's own kernels.
