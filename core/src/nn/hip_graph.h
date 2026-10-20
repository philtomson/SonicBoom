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

#include <hip/hip_runtime.h>

#include <cstdio>

// HIP twin of cuda_graph.h. Minimal HIP Graph capture/replay wrapper for one
// decode step, mirroring the CUDA GraphStep API so the planner backend can use
// the same capture/replay model on ROCm.
//
// NOT part of the public Layer 2 surface: lives under core/src/.
namespace sonicboom::hip {

class GraphStep {
 public:
  GraphStep() = default;
  ~GraphStep() { destroy(); }
  GraphStep(const GraphStep&) = delete;
  GraphStep& operator=(const GraphStep&) = delete;

  bool begin() {
    if (hipStreamCreate(&stream_) != hipSuccess) return false;
    if (hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal) != hipSuccess) {
      destroy();
      return false;
    }
    return true;
  }

  bool end() {
    hipError_t e = hipStreamEndCapture(stream_, &graph_);
    if (e != hipSuccess) {
      std::fprintf(stderr, "[hip-graph] end capture failed: %s\n", hipGetErrorString(e));
      destroy();
      return false;
    }
    e = hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    if (e != hipSuccess) {
      std::fprintf(stderr, "[hip-graph] instantiate failed: %s\n", hipGetErrorString(e));
      destroy();
      return false;
    }
    return true;
  }

  bool replay() { return hipGraphLaunch(graph_exec_, stream_) == hipSuccess; }

  bool sync() { return hipStreamSynchronize(stream_) == hipSuccess; }

  hipStream_t stream() const { return stream_; }
  bool valid() const { return graph_exec_ != nullptr; }

 private:
  void destroy() {
    if (graph_exec_) {
      hipGraphExecDestroy(graph_exec_);
      graph_exec_ = nullptr;
    }
    if (graph_) {
      hipGraphDestroy(graph_);
      graph_ = nullptr;
    }
    if (stream_) {
      hipStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  hipStream_t stream_ = nullptr;
  hipGraph_t graph_ = nullptr;
  hipGraphExec_t graph_exec_ = nullptr;
};

}  // namespace sonicboom::hip
