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

#include "device_pool.h"

#ifdef SONICBOOM_USE_CUDA
#include <cuda_runtime.h>
#include "../nn/cuda_stream.h"
#elif defined(SONICBOOM_USE_ROCM)
#include <hip/hip_runtime.h>
#include "../nn/hip_stream.h"
#endif

#include <mutex>
#include <unordered_map>
#include <vector>

namespace sonicboom::planner::device {

namespace {

constexpr uint64_t kAlignment = 256;

uint64_t align_up(uint64_t n) {
  const uint64_t rem = n % kAlignment;
  return rem == 0 ? n : n + (kAlignment - rem);
}

#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)

std::mutex g_mutex;
bool g_checked = false;
bool g_available = false;
// Free buffers keyed by aligned size.
std::unordered_map<uint64_t, std::vector<uint64_t>> g_free;

#endif

} // namespace

bool available() noexcept {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  std::lock_guard<std::mutex> lk(g_mutex);
  if (!g_checked) {
    int n = 0;
#if defined(SONICBOOM_USE_CUDA)
    g_available = (cudaGetDeviceCount(&n) == cudaSuccess && n > 0);
#else
    g_available = (hipGetDeviceCount(&n) == hipSuccess && n > 0);
#endif
    g_checked = true;
  }
  return g_available;
#else
  return false;
#endif
}

uint64_t acquire(uint64_t bytes) {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  if (bytes == 0 || !available()) return 0;
  const uint64_t size = align_up(bytes);
  std::lock_guard<std::mutex> lk(g_mutex);
  auto it = g_free.find(size);
  if (it != g_free.end() && !it->second.empty()) {
    const uint64_t h = it->second.back();
    it->second.pop_back();
    return h;
  }
  void* p = nullptr;
#if defined(SONICBOOM_USE_CUDA)
  if (cudaMalloc(&p, size) != cudaSuccess) return 0;
#else
  if (hipMalloc(&p, size) != hipSuccess) return 0;
#endif
  return reinterpret_cast<uint64_t>(p);
#else
  (void)bytes;
  return 0;
#endif
}

void release(uint64_t handle, uint64_t bytes) {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  if (handle == 0) return;
  const uint64_t size = align_up(bytes);
  std::lock_guard<std::mutex> lk(g_mutex);
  g_free[size].push_back(handle);
#else
  (void)handle;
  (void)bytes;
#endif
}

bool upload(const void* host, uint64_t handle, uint64_t bytes) {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  if (!host || handle == 0 || bytes == 0) return false;
#if defined(SONICBOOM_USE_CUDA)
  return cudaMemcpy(reinterpret_cast<void*>(handle), host, bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess;
#else
  return hipMemcpy(reinterpret_cast<void*>(handle), host, bytes,
                   hipMemcpyHostToDevice) == hipSuccess;
#endif
#else
  (void)host; (void)handle; (void)bytes;
  return false;
#endif
}

bool download(uint64_t handle, void* host, uint64_t bytes) {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  if (!host || handle == 0 || bytes == 0) return false;
  // A synchronous memcpy cannot be recorded into a graph. Under capture (see
  // the backend CaptureScope) skip the copy: the caller recovers the result
  // from a persistent device slot after replay, so the graph records only the
  // capturable device work.
#if defined(SONICBOOM_USE_CUDA)
  if (sonicboom::cuda::sb_capturing()) return true;
  return cudaMemcpy(host, reinterpret_cast<const void*>(handle), bytes,
                    cudaMemcpyDeviceToHost) == cudaSuccess;
#else
  if (sonicboom::hip::sb_capturing()) return true;
  return hipMemcpy(host, reinterpret_cast<const void*>(handle), bytes,
                   hipMemcpyDeviceToHost) == hipSuccess;
#endif
#else
  (void)handle; (void)host; (void)bytes;
  return false;
#endif
}

bool device_copy(uint64_t src, uint64_t dst, uint64_t bytes) {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  if (src == 0 || dst == 0 || bytes == 0) return false;
#if defined(SONICBOOM_USE_CUDA)
  return cudaMemcpyAsync(reinterpret_cast<void*>(dst),
                         reinterpret_cast<const void*>(src), bytes,
                         cudaMemcpyDeviceToDevice,
                         sonicboom::cuda::sb_current_stream()) == cudaSuccess;
#else
  return hipMemcpyAsync(reinterpret_cast<void*>(dst),
                        reinterpret_cast<const void*>(src), bytes,
                        hipMemcpyDeviceToDevice,
                        sonicboom::hip::sb_current_stream()) == hipSuccess;
#endif
#else
  (void)src; (void)dst; (void)bytes;
  return false;
#endif
}

bool memset_device(uint64_t handle, int value, uint64_t bytes) {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  if (handle == 0 || bytes == 0) return false;
#if defined(SONICBOOM_USE_CUDA)
  return cudaMemsetAsync(reinterpret_cast<void*>(handle), value, bytes,
                         sonicboom::cuda::sb_current_stream()) == cudaSuccess;
#else
  return hipMemsetAsync(reinterpret_cast<void*>(handle), value, bytes,
                        sonicboom::hip::sb_current_stream()) == hipSuccess;
#endif
#else
  (void)handle; (void)value; (void)bytes;
  return false;
#endif
}

bool synchronize() {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
#if defined(SONICBOOM_USE_CUDA)
  return cudaStreamSynchronize(0) == cudaSuccess;
#else
  return hipStreamSynchronize(0) == hipSuccess;
#endif
#else
  return false;
#endif
}

void clear_all() {
#if defined(SONICBOOM_USE_CUDA) || defined(SONICBOOM_USE_ROCM)
  std::lock_guard<std::mutex> lk(g_mutex);
  for (auto& kv : g_free)
    for (uint64_t h : kv.second) {
#if defined(SONICBOOM_USE_CUDA)
      cudaFree(reinterpret_cast<void*>(h));
#else
      hipFree(reinterpret_cast<void*>(h));
#endif
    }
  g_free.clear();
#endif
}

} // namespace sonicboom::planner::device
