// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

// Weak-reference glue for the rocBLAS-backed hipBLAS API. SonicBoom does not
// link librocblas/libhipblas directly: doing so transitively pulls ROCm's own
// libLLVM/libclang-cpp (via libamd_comgr) into the process, which collides with
// the statically-linked project LLVM that the MLIR path uses (two LLVM global
// constructor/destructor sets and interposed symbols -> option-parse failures
// and heap corruption). Instead the entry points are declared weak; when the
// runtime does not provide libhipblas they resolve to nullptr and the HIP
// kernels fall back to their own (scalar/vector) implementations.
//
// NOTE: weak undefined references do not trigger a dlopen of libhipblas. A
// build that wants BLAS acceleration must either link it explicitly or dlopen
// it before first use. This header keeps the ROCm build safe and correct
// without it.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#if defined(__GNUC__)
#define SB_WEAK __attribute__((weak))
#else
#define SB_WEAK
#endif

extern "C" {
SB_WEAK hipblasStatus_t hipblasCreate(hipblasHandle_t* handle);
SB_WEAK hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
SB_WEAK hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t stream);
SB_WEAK hipblasStatus_t hipblasSetMathMode(hipblasHandle_t handle, hipblasMath_t mode);
SB_WEAK hipblasStatus_t hipblasSetWorkspace(hipblasHandle_t handle, void* workspace,
                                            size_t size);
SB_WEAK hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, hipblasOperation_t transa,
                                     hipblasOperation_t transb, int m, int n, int k,
                                     const float* alpha, const float* A, int lda,
                                     const float* B, int ldb, const float* beta,
                                     float* C, int ldc);
SB_WEAK hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle, hipblasOperation_t transa,
                                      hipblasOperation_t transb, int m, int n, int k,
                                      const void* alpha, const void* A, hipDataType atype,
                                      int lda, const void* B, hipDataType btype, int ldb,
                                      const void* beta, void* C, hipDataType ctype, int ldc,
                                      hipblasComputeType_t compute_type,
                                      hipblasGemmAlgo_t algo);
}

// True when the hipBLAS symbols resolved (i.e. a libhipblas was linked in or
// dlopen'd before first use). The hipBLAS entry points are only called when
// this is true; otherwise the batched matvec falls back to the own kernels.
inline bool sb_hipblas_available() noexcept { return hipblasCreate != nullptr; }
