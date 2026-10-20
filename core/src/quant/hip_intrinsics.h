// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// HIP replacements for the NVIDIA byte-dot intrinsics used by the ggml-derived
// MMQ kernels (__vsub4 / __dp4a). AMD GPUs have no single instruction for these;
// the emulations are functionally exact (same per-byte semantics) but slower.
// This header is included only by the ROCm build of the K-quant matmul path.
//
// Not part of the public Layer 2 surface.

#pragma once

// unsigned per-byte subtraction with 8-bit wraparound (a_byte - b_byte mod 256)
__device__ __forceinline__ int __vsub4(int a, int b) {
  const unsigned int x = (unsigned int)a, y = (unsigned int)b;
  unsigned int r = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned int ai = (x >> (8 * i)) & 0xffu;
    const unsigned int bi = (y >> (8 * i)) & 0xffu;
    r |= ((ai - bi) & 0xffu) << (8 * i);
  }
  return (int)r;
}

// signed per-byte dot product: c + sum_i int8(a_i) * int8(b_i)
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
  int sum = c;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int ai = (int)(int8_t)((unsigned int)a >> (8 * i));
    const int bi = (int)(int8_t)((unsigned int)b >> (8 * i));
    sum += ai * bi;
  }
  return sum;
}

// signed per-byte subtraction with 8-bit wraparound (a_byte - b_byte mod 256)
__device__ __forceinline__ int __vsubss4(int a, int b) {
  const unsigned int x = (unsigned int)a, y = (unsigned int)b;
  unsigned int r = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned int ai = (x >> (8 * i)) & 0xffu;
    const unsigned int bi = (y >> (8 * i)) & 0xffu;
    r |= ((ai - bi) & 0xffu) << (8 * i);
  }
  return (int)r;
}
