// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// CUDA-side compatibility names for the shared ggml-derived MMQ headers.
//
// `ggml_derived/mmq.cuh` and `ggml_derived/mmvq_q8_1.cuh` are compiled by both
// backends: `cuda_quant_matmul.cu` (nvcc) and `hip_quant_matmul.hip` (hipcc)
// include the same two headers. They are kept in the hipify spelling, so their
// bodies name the HIP builtins (hipThreadIdx_x, hipBlockIdx_x, …) and the HIP
// runtime API (hipStream_t, hipMemsetAsync, hipFuncSetAttribute). HIP defines
// all of those itself; this header maps them back to the CUDA runtime so the
// same bodies compile under nvcc as well.
//
// It is included only from the non-__HIPCC__ branch of those two headers, so a
// ROCm build never sees it.
//
// Not part of the public Layer 2 surface.

#pragma once

#ifndef __HIPCC__

#include <cuda_runtime.h>

// Block/thread indices: HIP's own spellings of the CUDA builtins.
#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z
#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z
#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z
#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y

// Streams and the two runtime calls the MMQ launchers use.
using hipStream_t = cudaStream_t;
#define hipMemsetAsync cudaMemsetAsync
#define hipFuncSetAttribute cudaFuncSetAttribute
#define hipFuncAttributeMaxDynamicSharedMemorySize \
  cudaFuncAttributeMaxDynamicSharedMemorySize

#endif // !__HIPCC__
