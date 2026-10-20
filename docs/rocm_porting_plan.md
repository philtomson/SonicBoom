# ROCm / Strix Halo Porting Plan

Status: **active** — Phase 0-2 complete (branch `amd_dev`), Phase 4 started
(vendor-neutral device gate landed), Phase 3 open. `origin/master` merged in
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
- [x] **Full C++ suite on the ROCm build:** `ctest -E guile` = **74/74 pass**
      (5 expected SKIPs for absent model fixtures). The Guile tests need the
      `(json)` module — `guile-json` is **not** packaged in Fedora, so it was
      built from source
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
- [ ] Emit non-tensor-core thread-partitioned AMD kernels for
      `nvgpu.ldmatrix`/`nvgpu.mma.sync`/`nvvm.cp.async` regions.
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
- [ ] **The rest of the runtime is still `cuda_available()`-gated and therefore
      stays off on AMD:** the `exec.cpp` elementwise / device-resident routing
      (`core/src/model/exec.cpp:264`, `:434`, `:507`, `:561`, `:580` — the
      kernels it calls exist as HIP twins, but each routing decision needs its
      kernel's AMD coverage confirmed before flipping), the CUDA spine
      (`core/src/model/gemma4_plan.cpp:1107`, whose `#else` stubs at `:2497` /
      `:3260` make the whole `generate_spine_cuda*` path a no-op), planner
      device enumeration (`core/src/planner/resource.cpp:270`) and the
      spine-gated tests (`test_generate.cpp:269`, `test_planner_resource.cpp`,
      `test_plan_cache.cpp`, `test_plan_record.cpp`, `test_prefix_cache.cpp`,
      `test_boom_conversion.cpp`). The gate itself is no longer the blocker for
      `test_generate.cpp:171` (Cuda matvec vs the CPU q8_K oracle), which is
      backend-neutral and can move to `gpu_available()`; it still needs a GGUF
      fixture to run.
- [ ] Vendor-neutral IDs: `DeviceType::GPU`, `BackendId::gpu()`, `quant::gpu`;
      move `sonic_cuda_backend.h` behind boundary.
- [ ] Relax byte-identity attention/GEMM asserts (wave/expf differences) —
      applies once those regions exist for AMD (Phase 3).

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
- Attention/GEMM codegen is NVIDIA-mma-specific; the functional AMD path is new
  code, not a flag flip.
- Wave width / shared-memory / async-copy semantics differ; wave32 is pinned
  (see Phase 2). LDS limits on gfx1151 still need checking once the generated
  attention/GEMM regions exist.
- Native-torch HIP sources exist upstream but are `EXCLUDE` in the migration
  manifest; not needed for v0's own kernels.
