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

// The generated dense bf16 GEMM (compiler/GemmCodegen.h).
//
// The kernel is emitted directly in the GPU dialect: one gpu.func, a grid of
// (token tile, row tile) blocks, each warp owning a 16-row strip of the output
// tile. Every K step stages BM x BK of W and BN x BK of X into double-buffered
// shared memory with nvgpu.device_async_copy (one cp.async per lane), waits
// with cp.async.wait_group (one group outstanding while the next tile is in
// flight), and then runs one m16n8k16 mma per (16-row strip, 8-token tile,
// 16-K slice).
//
// The A fragment is the non-transposed ldmatrix x4 form (row = warp*16 + l&15,
// col = kb*16 + (l>>4)*8); the B fragment is the non-transposed ldmatrix x2
// form read from a [BN][BK] token-major stage (row = nt*8 + l&7,
// col = kb*16 + ((l>>3)&1)*8). Both match the layouts documented in
// AttentionCodegen.h and validated there.

#include <sonicboom/compiler/GemmCodegen.h>

#include <llvm/Config/llvm-config.h>

#include <map>
#include <span>
#include <string>
#include <vector>

#include <sonicboom/compiler/GlueCodegen.h>

namespace sonicboom::compiler {
namespace {

// The AMDGPU body stages a shared slab whose rows are padded to this many
// elements: 32 bytes, the width of one WMMA fragment read.
constexpr int kAmdWmmaPad = 16;

// gfx11 gives a workgroup 64 KiB of LDS; the AMD body stages one slab per K
// step and rejects shapes whose slab does not fit.
constexpr int kAmdDenseGemmLdsBytes = 64 * 1024;

struct Emitter {
  std::string body;
  int next = 0;

  void op(const std::string &s) { body += "      " + s + "\n"; }

  std::string val(const std::string &expr) {
    std::string v = "%v" + std::to_string(++next);
    body += "      " + v + " = " + expr + "\n";
    return v;
  }
};

std::string gemm_region_text(const DenseGemmShape &sh, int sm_arch) {
  const int BM = sh.bm;
  const int BN = sh.bn;
  const int BK = sh.bk;
  const int NW = BM / 16;  // warps (one 16-row strip each)
  const int TH = NW * 32;
  const int NT = BN / 8;   // 8-token accumulator tiles per lane
  const int KB = BK / 16;  // 16-K mma steps per stage
  // Shared-memory row padding, in elements (16 bytes: rows stay 16-byte
  // aligned for ldmatrix). A dense [rows][BK] stage makes every row start at
  // the same bank ((BK*2/4) % 32 == 0 for BK = 64), which ldmatrix reads as an
  // 8-way conflict; +8 elements shifts each row by 16 banks for BK = 32 and by
  // 4 banks for BK = 64, so the 8 rows of a matrix hit disjoint bank ranges.
  const int PAD = 8;
  const int BKW = BK + PAD;

  Emitter e;
  std::map<std::string, std::string> C;
  const auto s = [](int v) { return std::to_string(v); };
  const auto c32 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i32");
  };
  const auto c64 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i64");
  };
  const auto ci = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : index");
  };

  c32("c0", 0); c32("c1", 1); c32("c2", 2); c32("c3", 3); c32("c4", 4);
  c32("c5", 5); c32("c7", 7); c32("c8", 8); c32("c15", 15); c32("c16", 16);
  c32("c31", 31);
  c64("l0", 0); c64("l1", 1); c64("l8", 8);
  ci("i0", 0); ci("i1", 1); ci("i8", 8);
  ci("ith", TH);                 // staging stride (one chunk per lane)
  ci("ibk", BK);
  ci("ibsoff", 2 * BM * BKW * 2); // B stage starts after the A double buffer
  ci("icha", BM * BK / 8);       // 8-bf16 chunks per A tile
  ci("ichb", BN * BK / 8);       // 8-bf16 chunks per B tile (token-major)
  c32("ibmi", BM); c32("ibni", BN); c32("ibki", BK);

  const std::string FUNC =
      "gpu.func @gemm(%x: memref<?x?xbf16>, %w: memref<?x?xbf16>, "
      "%y: !llvm.ptr, %m: i32, %k: i32, %n: i32, %alpha: f32) kernel {";

  const std::string tid = e.val("gpu.thread_id x");
  const std::string bx = e.val("gpu.block_id x");
  const std::string by = e.val("gpu.block_id y");
  const std::string smem = e.val("gpu.dynamic_shared_memory : memref<?xi8, "
                                 "#gpu.address_space<workgroup>>");
  const std::string asm_ =
      e.val("memref.view " + smem + "[" + C["i0"] +
            "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
            s(BM) + "x" + s(BKW) +
            "xbf16, #gpu.address_space<workgroup>>");
  const std::string bsm =
      e.val("memref.view " + smem + "[" + C["ibsoff"] +
            "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
            s(BN) + "x" + s(BKW) +
            "xbf16, #gpu.address_space<workgroup>>");
  const std::string tidi = e.val("arith.index_cast " + tid + " : index to i32");
  const std::string lane =
      e.val("arith.andi " + tidi + ", " + C["c31"] + " : i32");
  const std::string warp =
      e.val("arith.shrui " + tidi + ", " + C["c5"] + " : i32");
  const std::string bxi = e.val("arith.index_cast " + bx + " : index to i32");
  const std::string byi = e.val("arith.index_cast " + by + " : index to i32");
  const std::string n0 =
      e.val("arith.muli " + bxi + ", " + C["ibni"] + " : i32");
  const std::string m0 =
      e.val("arith.muli " + byi + ", " + C["ibmi"] + " : i32");

  // One staged tile of `nrows` rows x BK columns, 8 bf16 (16 B) per lane per
  // iteration. `src` is the global operand, `dst` the shared double buffer,
  // `base`/`bound` the row origin and its runtime bound, `kt` the K step, and
  // `buf` the double-buffer slot. Out-of-range chunks request a zero fill.
  const auto stage_body = [&](const char *src, const std::string &dst,
                              const std::string &buf, const std::string &kt,
                              const std::string &base, const char *bound,
                              int nrows, const char *chunks, const char *tag) {
    const std::string it = "%ci_" + std::string(tag);
    e.op("scf.for " + it + " = " + tid + " to " + C[chunks] + " step " +
         C["ith"] + " : index {");
    const std::string el =
        e.val("arith.muli " + it + ", " + C["i8"] + " : index");
    const std::string row = e.val("arith.divui " + el + ", " + C["ibk"] +
                                 " : index");
    const std::string col = e.val("arith.remui " + el + ", " + C["ibk"] +
                                 " : index");
    const std::string row32 =
        e.val("arith.index_cast " + row + " : index to i32");
    const std::string col32 =
        e.val("arith.index_cast " + col + " : index to i32");
    const std::string ktcol =
        e.val("arith.muli " + kt + ", " + C["ibki"] + " : i32");
    const std::string grow =
        e.val("arith.addi " + base + ", " + row32 + " : i32");
    const std::string gcol =
        e.val("arith.addi " + ktcol + ", " + col32 + " : i32");
    const std::string rok =
        e.val("arith.cmpi slt, " + grow + ", " + bound + " : i32");
    const std::string cok = e.val("arith.cmpi slt, " + gcol + ", %k : i32");
    const std::string ok =
        e.val("arith.andi " + rok + ", " + cok + " : i1");
    const std::string sr =
        e.val("arith.select " + ok + ", " + grow + ", " + C["c0"] + " : i32");
    const std::string sc =
        e.val("arith.select " + ok + ", " + gcol + ", " + C["c0"] + " : i32");
    const std::string sri =
        e.val("arith.index_cast " + sr + " : i32 to index");
    const std::string sci =
        e.val("arith.index_cast " + sc + " : i32 to index");
    const std::string nel =
        e.val("arith.select " + ok + ", " + C["i8"] + ", " + C["i0"] +
              " : index");
    e.val(std::string("nvgpu.device_async_copy %") + src + "[" + sri + ", " +
          sci + "], " + dst + "[" + buf + ", " + row + ", " + col + "], 8, " +
          nel + " : memref<?x?xbf16> to memref<2x" + s(nrows) + "x" + s(BKW) +
          "xbf16, #gpu.address_space<workgroup>>");
    e.op("}");
  };

  const std::string kidx = e.val("arith.index_cast %k : i32 to index");
  const std::string lane15 =
      e.val("arith.andi " + lane + ", " + C["c15"] + " : i32");
  const std::string lsh16 =
      e.val("arith.shrui " + lane + ", " + C["c4"] + " : i32");
  const std::string l7 =
      e.val("arith.andi " + lane + ", " + C["c7"] + " : i32");
  const std::string lsh3 =
      e.val("arith.shrui " + lane + ", " + C["c3"] + " : i32");
  const std::string lsh3m =
      e.val("arith.andi " + lsh3 + ", " + C["c1"] + " : i32");
  const std::string w16 =
      e.val("arith.muli " + warp + ", " + C["c16"] + " : i32");
  const std::string arow32 =
      e.val("arith.addi " + w16 + ", " + lane15 + " : i32");
  const std::string arowi =
      e.val("arith.index_cast " + arow32 + " : i32 to index");
  const std::string acol32 =
      e.val("arith.muli " + lsh16 + ", " + C["c8"] + " : i32");
  const std::string bcoloff =
      e.val("arith.muli " + lsh3m + ", " + C["c8"] + " : i32");
  const std::string ktmax =
      e.val("arith.ceildivui " + kidx + ", " + C["ibk"] + " : index");
  const std::string ktmaxi =
      e.val("arith.index_cast " + ktmax + " : index to i32");

  // Prologue: stage K step 0 into slot 0 and commit it as its own group.
  stage_body("w", asm_, C["i0"], C["c0"], m0, "%m", BM, "icha", "a0");
  stage_body("x", bsm, C["i0"], C["c0"], n0, "%n", BN, "ichb", "b0");
  const std::string gtok =
      e.val("\"nvgpu.device_async_create_group\"() : () -> "
            "!nvgpu.device.async.token");

  std::vector<std::string> accs, names;
  for (int i = 0; i < NT; ++i) {
    accs.push_back(
        e.val("arith.constant dense<0.0> : vector<2x2xf32>"));
    names.push_back("%o" + std::to_string(i));
  }
  std::string ia = "iter_args(";
  for (int i = 0; i < NT; ++i)
    ia += (i ? ", " : "") + names[i] + " = " + accs[i];
  ia += ")";
  std::string ty = " -> (";
  std::string res;
  for (int i = 0; i < NT; ++i) {
    ty += (i ? ", " : "") + std::string("vector<2x2xf32>");
    res += (i ? ", " : "") + std::string("%r") + std::to_string(i);
  }
  ty += ")";

  e.op(res + " = scf.for %kt = " + C["i0"] + " to " + ktmax + " step " +
       C["i1"] + " " + ia + ty + " {");
  const std::string kti = e.val("arith.index_cast %kt : index to i32");
  std::vector<std::string> cur = names;
  const std::string nxt =
      e.val("arith.addi " + kti + ", " + C["c1"] + " : i32");
  const std::string has =
      e.val("arith.cmpi slt, " + nxt + ", " + ktmaxi + " : i32");
  // Issue the next stage while this one computes; the wait below keeps at most
  // one group in flight, so the last iteration waits for completion.
  e.op("scf.if " + has + " {");
  const std::string bufn =
      e.val("arith.andi " + nxt + ", " + C["c1"] + " : i32");
  const std::string bufi =
      e.val("arith.index_cast " + bufn + " : i32 to index");
  stage_body("w", asm_, bufi, nxt, m0, "%m", BM, "icha", "a1");
  stage_body("x", bsm, bufi, nxt, n0, "%n", BN, "ichb", "b1");
  e.val("\"nvgpu.device_async_create_group\"() : () -> "
        "!nvgpu.device.async.token");
  e.op("}");
  e.op("scf.if " + has + " {");
  e.op("nvgpu.device_async_wait " + gtok + " {numGroups = 1 : i32}");
  e.op("} else {");
  e.op("nvgpu.device_async_wait " + gtok + " {numGroups = 0 : i32}");
  e.op("}");
  e.op("gpu.barrier");
  const std::string buf =
      e.val("arith.andi " + kti + ", " + C["c1"] + " : i32");
  const std::string buf_i =
      e.val("arith.index_cast " + buf + " : i32 to index");

  for (int kb = 0; kb < KB; ++kb) {
    std::string acol = acol32, bcol = bcoloff;
    if (kb != 0) {
      const std::string kc = e.val("arith.constant " + s(kb * 16) + " : i32");
      acol = e.val("arith.addi " + acol32 + ", " + kc + " : i32");
      bcol = e.val("arith.addi " + bcoloff + ", " + kc + " : i32");
    }
    const std::string aci =
        e.val("arith.index_cast " + acol + " : i32 to index");
    const std::string a =
        e.val("nvgpu.ldmatrix " + asm_ + "[" + buf_i + ", " + arowi + ", " +
              aci + "] {numTiles = 4 : i32, transpose = false} : memref<2x" +
              s(BM) + "x" + s(BKW) +
              "xbf16, #gpu.address_space<workgroup>> -> vector<4x2xbf16>");
    const std::string bci =
        e.val("arith.index_cast " + bcol + " : i32 to index");
    for (int nt = 0; nt < NT; ++nt) {
      std::string brow32 = l7;
      if (nt != 0) {
        const std::string bc =
            e.val("arith.constant " + s(nt * 8) + " : i32");
        brow32 = e.val("arith.addi " + l7 + ", " + bc + " : i32");
      }
      const std::string browi =
          e.val("arith.index_cast " + brow32 + " : i32 to index");
      const std::string b =
          e.val("nvgpu.ldmatrix " + bsm + "[" + buf_i + ", " + browi + ", " +
                bci + "] {numTiles = 2 : i32, transpose = false} : memref<2x" +
                s(BN) + "x" + s(BKW) +
                "xbf16, #gpu.address_space<workgroup>> -> vector<2x2xbf16>");
      cur[nt] = e.val("nvgpu.mma.sync (" + a + ", " + b + ", " + cur[nt] +
                      ") {mmaShape = [16, 8, 16]} : (vector<4x2xbf16>, "
                      "vector<2x2xbf16>, vector<2x2xf32>) -> "
                      "vector<2x2xf32>");
    }
  }
  e.op("gpu.barrier");
  std::string yields, ytypes;
  for (int i = 0; i < NT; ++i) {
    yields += (i ? ", " : "") + cur[i];
    ytypes += (i ? ", " : "") + std::string("vector<2x2xf32>");
  }
  e.op("scf.yield " + yields + " : " + ytypes);
  e.op("}");

  // ---- store: Y[token][row] (token-major), f32 ----
  const std::string l2c =
      e.val("arith.andi " + lane + ", " + C["c3"] + " : i32");
  const std::string l2x =
      e.val("arith.muli " + l2c + ", " + C["c2"] + " : i32");
  const std::string r4 =
      e.val("arith.shrui " + lane + ", " + C["c2"] + " : i32");
  const std::string mrow_base =
      e.val("arith.addi " + w16 + ", " + r4 + " : i32");
  const std::string m64 = e.val("arith.extsi %m : i32 to i64");
  for (int nt = 0; nt < NT; ++nt) {
    const std::string o = "%r" + std::to_string(nt);
    std::string tok = n0;
    if (nt != 0) {
      const std::string tc = e.val("arith.constant " + s(nt * 8) + " : i32");
      tok = e.val("arith.addi " + n0 + ", " + tc + " : i32");
    }
    tok = e.val("arith.addi " + tok + ", " + l2x + " : i32");
    for (int rr = 0; rr < 2; ++rr) {
      std::string mrow = mrow_base;
      if (rr != 0)
        mrow = e.val("arith.addi " + mrow_base + ", " + C["c8"] + " : i32");
      const std::string gmrow =
          e.val("arith.addi " + m0 + ", " + mrow + " : i32");
      const std::string mok =
          e.val("arith.cmpi slt, " + gmrow + ", %m : i32");
      for (int cc = 0; cc < 2; ++cc) {
        const std::string tokc =
            cc != 0 ? e.val("arith.addi " + tok + ", " + C["c1"] + " : i32")
                    : tok;
        const std::string tok_ok =
            e.val("arith.cmpi slt, " + tokc + ", %n : i32");
        const std::string ok =
            e.val("arith.andi " + mok + ", " + tok_ok + " : i1");
        e.op("scf.if " + ok + " {");
        const std::string tok64 = e.val("arith.extsi " + tokc + " : i32 to i64");
        const std::string off0 =
            e.val("arith.muli " + tok64 + ", " + m64 + " : i64");
        const std::string mrow64 =
            e.val("arith.extsi " + gmrow + " : i32 to i64");
        const std::string off =
            e.val("arith.addi " + off0 + ", " + mrow64 + " : i64");
        const std::string p =
            e.val("llvm.getelementptr %y[" + off +
                  "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
        const std::string v =
            e.val("vector.extract " + o + "[" + s(rr) + ", " + s(cc) +
                  "] : f32 from vector<2x2xf32>");
        // Epilogue: the planner folds a single-reader scalar scale into the
        // projection (the per-layer model scaling in both models), so the
        // output is written scaled and the consumer reads it as before.
        const std::string ys = e.val("arith.mulf " + v + ", %alpha : f32");
        e.op("llvm.store " + ys + ", " + p + " : f32, !llvm.ptr");
        e.op("}");
      }
    }
  }
  e.op("gpu.return");
  e.op("}");

  std::string text = "module attributes {gpu.container_module} {\n";
  text += "  gpu.module @gemm [#nvvm.target<chip = \"sm_" + s(sm_arch) +
          "\">] {\n";
  text += "      " + FUNC + "\n" + e.body + "\n";
  text += "  }\n}\n";
  return text;
}

// AMDGPU body. The tensor-core text above is NVVM-only (nvgpu ldmatrix/mma,
// cp.async), so the AMD body targets what gfx11/gfx12 actually have: the
// ROCDL WMMA intrinsic. RDNA has no MFMA (CDNA-only; llvm-mc rejects
// `v_mfma_*` for gfx1151), while `v_wmma_f32_16x16x16_bf16` computes a
// 16x16x16 bf16 -> f32 tile per wave32 wavefront:
//   A (16x16, M x K): lane L holds A[L % 16][*] -- one row, K in the slots;
//   B (16x16, K x N): lane L holds B[*][L % 16] -- one column, K in the slots;
//   D (16x16, M x N): lane L, slot i -> D[2*i + L/16][L % 16].
// Lanes 16-31 hold the same A and B fragments as lanes 0-15 (matrix
// replication); their D slots are the odd rows, so a warp produces a whole tile
// with no cross-lane traffic. Operands are therefore "one K-contiguous row"
// fragments (X's token row, W's row): no ldmatrix, no transposes.
//
// Staging moves 8 bf16 (16 B) per lane per chunk, so a BK = 64 slab is four
// vector loads per lane instead of 32 scalar ones, and the chunk index is fully
// unrolled so all of a thread's global loads are in flight before its first LDS
// store. A staged row is 16-byte aligned only when K is a multiple of 8, so
// that is a runtime branch: the fast path clamps the address and selects a zero
// vector out of range, the slow path is the scalar per-element loop with zero
// fill.
//
// The slab is double-buffered whenever it fits the 64 KiB LDS budget
// (2 * (BM + BN) * (BK + 16) * 2 bytes: every candidate tile but 128x128x64).
// K step kt+1 is then staged into the second buffer while kt computes, and a
// single `gpu.barrier` per step publishes it; shapes that do not fit keep the
// single buffer and two barriers per step.
//
// A warp owns a 16-row strip (mi = warp % MT) and TPT whole 16x16 tiles across
// the block's columns (ni = warp / MT + slot * WPG), so its A fragment is one
// LDS read per 16-K step shared by all of its tiles; 8 f32 accumulators per
// tile stay in registers across the K loop. LDS rows are XOR-swizzled in
// 8-element chunks by the row index, so a fragment is two 16-byte reads joined
// by a register-only shuffle and the 16 lane rows of a fragment land on 8 bank
// groups instead of 4. The epilogue writes Y[token * M + row] with the same
// exact `alpha * acc` as the other bodies.
std::string gemm_region_text_amd(const DenseGemmShape &sh, int gfx_arch) {
  const int BM = sh.bm;
  const int BN = sh.bn;
  const int BK = sh.bk;
  constexpr int kPad = kAmdWmmaPad;  // elements; 32-byte LDS row alignment
  const int BKW = BK + kPad;
  const int NV8 = BKW / 8;           // vector<8xbf16> slots per LDS row
  const int NS8 = BK / 8;            // ... of which this many hold staged data
  const int MT = BM / 16;            // 16-row tiles down the block
  const int NT = BN / 16;            // 16-column tiles across the block
  const int TILES = MT * NT;
  const int WARPS = TILES >= 8 ? 8 : 4;
  const int THREADS = WARPS * 32;
  const int WPG = WARPS / MT;        // warps sharing one 16-row strip
  const int TPT = NT / WPG;          // whole tiles per warp
  const int KS = BK / 16;            // 16-K WMMA steps per staged slab
  constexpr int BRT = 16;            // rows of a WMMA tile
  constexpr int ACC = 8;             // f32 accumulators per tile per lane
  const int CH8 = BK / 8;            // 8-bf16 chunks per staged row
  const int CHA = BM * CH8;          // staged chunks in the W slab
  const int CHB = BN * CH8;          // staged chunks in the X slab
  const int SlabA = BM * BKW * 2;    // bytes
  const int SlabB = BN * BKW * 2;
  const int Pair = SlabA + SlabB;    // bytes; A plus B of one buffer
  const bool DB = 2 * Pair <= kAmdDenseGemmLdsBytes;
  const std::string SmemTy = "#gpu.address_space<workgroup>";

  Emitter e;
  std::map<std::string, std::string> C;
  const auto s = [](int v) { return std::to_string(v); };
  const auto c32 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i32");
  };
  const auto c64 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i64");
  };
  const auto cf = [&](const char *n, const char *v) {
    C[n] = e.val(std::string("arith.constant ") + v + " : f32");
  };
  const auto ci = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : index");
  };
  const auto cconst = [&](int v) {
    return e.val("arith.constant " + s(v) + " : index");
  };

  const std::string FUNC =
      "gpu.func @gemm(%x: memref<?x?xbf16>, %w: memref<?x?xbf16>, "
      "%y: !llvm.ptr, %m: i32, %k: i32, %n: i32, %alpha: f32) kernel {";

  c32("c0", 0); c32("c1", 1); c32("c2", 2); c32("c8", 8);
  for (int i = 4; i < 2 * ACC; i += 2) c32(("c" + s(i)).c_str(), i);
  c32("cbm", BM); c32("cbn", BN);
  c64("l0", 0); c64("l1", 1);
  ci("i0", 0); ci("i1", 1); ci("i2", 2); ci("i8", 8); ci("i16", BRT);
  ci("i32", 32);
  ci("ith", THREADS);
  ci("ibk", BK);
  ci("ich8", CH8);
  ci("imask", NS8 - 1);
  ci("icha", CHA);
  ci("ichb", CHB);
  ci("islaba", SlabA);
  ci("ipair", Pair);
  ci("imt", MT);
  cf("cf0", "0.0");
  C["czero"] = e.val("arith.constant dense<0.0> : vector<8xf32>");
  C["czerobf"] = e.val("arith.constant dense<0.0> : vector<8xbf16>");

  const std::string tid = e.val("gpu.thread_id x");
  const std::string bx = e.val("gpu.block_id x");
  const std::string by = e.val("gpu.block_id y");
  const std::string smem =
      e.val("gpu.dynamic_shared_memory : memref<?xi8, " + SmemTy + ">");
  const std::string bxi = e.val("arith.index_cast " + bx + " : index to i32");
  const std::string byi = e.val("arith.index_cast " + by + " : index to i32");
  const std::string n0 = e.val("arith.muli " + bxi + ", " + C["cbn"] +
                               " : i32");
  const std::string m0 = e.val("arith.muli " + byi + ", " + C["cbm"] +
                               " : i32");
  const std::string kidx = e.val("arith.index_cast %k : i32 to index");
  const std::string ktmax =
      e.val("arith.ceildivui " + kidx + ", " + C["ibk"] + " : index");
  const std::string krem =
      e.val("arith.remui " + kidx + ", " + C["i8"] + " : index");
  const std::string fast =
      e.val("arith.cmpi eq, " + krem + ", " + C["i0"] + " : index");

  // One buffer holds the W slab at its base and the X slab after it, viewed
  // twice: bf16 (scalar staging fallback) and vector<8xbf16> (staging and
  // fragment reads). `vector.load` from a workgroup memref is not converted by
  // the ROCDL pipeline, while `memref.load` of a vector element type is.
  //
  // Each row is XOR-swizzled in 8-element chunks by its own row index: chunk c
  // of row r is stored at c ^ (r & (NS8 - 1)). A fragment is then two 16-byte
  // reads at (2*ks ^ m) and (2*ks+1 ^ m), which puts the 16 lane rows of a
  // fragment (row stride BKW/2 dwords) on 8 distinct bank groups instead of 4.
  struct SlabViews {
    std::string a_s, b_s, a_8, b_8;
  };
  const auto mkviews = [&](const std::string &off) {
    SlabViews v;
    const std::string boff =
        e.val("arith.addi " + off + ", " + C["islaba"] + " : index");
    const auto mv = [&](const std::string &base, const std::string &ty) {
      return e.val("memref.view " + smem + "[" + base +
                   "][] : memref<?xi8, " + SmemTy + "> to " + ty);
    };
    v.a_s = mv(off, "memref<" + s(BM) + "x" + s(BKW) + "xbf16, " + SmemTy +
                        ">");
    v.b_s = mv(boff, "memref<" + s(BN) + "x" + s(BKW) + "xbf16, " + SmemTy +
                         ">");
    v.a_8 = mv(off, "memref<" + s(BM) + "x" + s(NV8) +
                        "xvector<8xbf16>, " + SmemTy + ">");
    v.b_8 = mv(boff, "memref<" + s(BN) + "x" + s(NV8) +
                         "xvector<8xbf16>, " + SmemTy + ">");
    return v;
  };

  // Stage one slab of `nrows` x BK from K step `kt` into `dst8v`/`dstsv` (the
  // vector<8xbf16> and bf16 views of the same buffer), zero-filling outside the
  // runtime row and K bounds. Each thread owns a strided set of 8-element
  // chunks; the chunk index is fully unrolled because every candidate shape
  // divides the chunk count by the block size, so all of a thread's global
  // loads are issued before the first LDS store instead of one latency at a
  // time (the staged slab dominates the kernel otherwise).
  const auto stage = [&](const char *src, int nrows, const std::string &dst8v,
                         const std::string &dstsv, const std::string &base,
                         const char *bound, int chunks, const char *chs,
                         const std::string &kt, const char *tag) {
    const std::string ty8 = "memref<" + s(nrows) + "x" + s(NV8) +
                            "xvector<8xbf16>, " + SmemTy + ">";
    const std::string tyb =
        "memref<" + s(nrows) + "x" + s(BKW) + "xbf16, " + SmemTy + ">";
    const std::string ktcol =
        e.val("arith.muli " + kt + ", " + C["ibk"] + " : index");

    // Fast chunk (K a multiple of 8): one clamped 16-byte global load, a zero
    // fill when the chunk is out of range, and the swizzled LDS store.
    const auto fast_chunk = [&](const std::string &c) {
      const std::string row =
          e.val("arith.divui " + c + ", " + C["ich8"] + " : index");
      const std::string cg =
          e.val("arith.remui " + c + ", " + C["ich8"] + " : index");
      const std::string col =
          e.val("arith.muli " + cg + ", " + C["i8"] + " : index");
      const std::string row32 =
          e.val("arith.index_cast " + row + " : index to i32");
      const std::string grow =
          e.val("arith.addi " + base + ", " + row32 + " : i32");
      const std::string gcol =
          e.val("arith.addi " + ktcol + ", " + col + " : index");
      const std::string rok =
          e.val("arith.cmpi slt, " + grow + ", " + bound + " : i32");
      const std::string cok =
          e.val("arith.cmpi slt, " + gcol + ", " + kidx + " : index");
      const std::string ok =
          e.val("arith.andi " + rok + ", " + cok + " : i1");
      const std::string sr =
          e.val("arith.select " + ok + ", " + grow + ", " + C["c0"] +
                " : i32");
      const std::string sc =
          e.val("arith.select " + ok + ", " + gcol + ", " + C["i0"] +
                " : index");
      const std::string sri =
          e.val("arith.index_cast " + sr + " : i32 to index");
      const std::string v =
          e.val(std::string("vector.load %") + src + "[" + sri + ", " + sc +
                "] : memref<?x?xbf16>, vector<8xbf16>");
      const std::string vv =
          e.val("arith.select " + ok + ", " + v + ", " + C["czerobf"] +
                " : vector<8xbf16>");
      const std::string rm =
          e.val("arith.andi " + row + ", " + C["imask"] + " : index");
      const std::string cgp =
          e.val("arith.xori " + cg + ", " + rm + " : index");
      e.op("memref.store " + vv + ", " + dst8v + "[" + row + ", " + cgp +
           "] : " + ty8);
    };

    // Slow chunk (K not a multiple of 8): per-element bounds and zero fill.
    const auto slow_chunk = [&](const std::string &c) {
      const std::string row =
          e.val("arith.divui " + c + ", " + C["ich8"] + " : index");
      const std::string cg =
          e.val("arith.remui " + c + ", " + C["ich8"] + " : index");
      const std::string col =
          e.val("arith.muli " + cg + ", " + C["i8"] + " : index");
      const std::string rm =
          e.val("arith.andi " + row + ", " + C["imask"] + " : index");
      const std::string cgp =
          e.val("arith.xori " + cg + ", " + rm + " : index");
      const std::string colp =
          e.val("arith.muli " + cgp + ", " + C["i8"] + " : index");
      const std::string row32 =
          e.val("arith.index_cast " + row + " : index to i32");
      const std::string grow =
          e.val("arith.addi " + base + ", " + row32 + " : i32");
      const std::string gcol =
          e.val("arith.addi " + ktcol + ", " + col + " : index");
      const std::string rok =
          e.val("arith.cmpi slt, " + grow + ", " + bound + " : i32");
      const std::string jt = "%cj_" + std::string(tag);
      e.op("scf.for " + jt + " = " + C["i0"] + " to " + C["i8"] + " step " +
           C["i1"] + " : index {");
      const std::string gcj =
          e.val("arith.addi " + gcol + ", " + jt + " : index");
      const std::string cokj =
          e.val("arith.cmpi slt, " + gcj + ", " + kidx + " : index");
      const std::string okj =
          e.val("arith.andi " + rok + ", " + cokj + " : i1");
      const std::string got = e.val("scf.if " + okj + " -> bf16 {");
      {
        const std::string gr =
            e.val("arith.index_cast " + grow + " : i32 to index");
        const std::string g =
            e.val(std::string("memref.load %") + src + "[" + gr + ", " + gcj +
                  "] : memref<?x?xbf16>");
        e.op("scf.yield " + g + " : bf16");
      }
      e.op("} else {");
      {
        const std::string z = e.val("arith.constant 0.0 : bf16");
        e.op("scf.yield " + z + " : bf16");
      }
      e.op("}");
      const std::string cjp =
          e.val("arith.addi " + colp + ", " + jt + " : index");
      e.op("memref.store " + got + ", " + dstsv + "[" + row + ", " + cjp +
           "] : " + tyb);
      e.op("}");
    };

    e.op("scf.if " + fast + " {");
    if (chunks % THREADS == 0) {
      for (int u = 0; u * THREADS < chunks; ++u) {
        const std::string cu =
            e.val("arith.addi " + tid + ", " + cconst(u * THREADS) +
                  " : index");
        fast_chunk(cu);
      }
    } else {
      const std::string it = "%cf_" + std::string(tag);
      e.op("scf.for " + it + " = " + tid + " to " + C[chs] + " step " +
           C["ith"] + " : index {");
      fast_chunk(it);
      e.op("}");
    }
    e.op("} else {");
    const std::string st = "%cs_" + std::string(tag);
    e.op("scf.for " + st + " = " + tid + " to " + C[chs] + " step " +
         C["ith"] + " : index {");
    slow_chunk(st);
    e.op("}");
    e.op("}");
  };

  // Wave coordinates: a warp owns one 16-row strip (mi) and TPT tiles across
  // the columns (ni). lane % 16 picks the tile row (A) / tile column (B) and
  // lane / 16 picks the odd accumulator rows.
  const std::string warp =
      e.val("arith.divui " + tid + ", " + C["i32"] + " : index");
  const std::string lane =
      e.val("arith.remui " + tid + ", " + C["i32"] + " : index");
  const std::string l16 =
      e.val("arith.remui " + lane + ", " + C["i16"] + " : index");
  const std::string hi =
      e.val("arith.divui " + lane + ", " + C["i16"] + " : index");
  const std::string hi32 = e.val("arith.index_cast " + hi + " : index to i32");
  const std::string l32 = e.val("arith.index_cast " + l16 + " : index to i32");
  const std::string mi =
      e.val("arith.remui " + warp + ", " + C["imt"] + " : index");
  const std::string ng =
      e.val("arith.divui " + warp + ", " + C["imt"] + " : index");
  const std::string mi16 =
      e.val("arith.muli " + mi + ", " + C["i16"] + " : index");
  const std::string arow =
      e.val("arith.addi " + mi16 + ", " + l16 + " : index");
  const std::string mi32 =
      e.val("arith.index_cast " + mi16 + " : index to i32");
  const std::string mbase = e.val("arith.addi " + m0 + ", " + mi32 +
                                  " : i32");

  std::vector<std::string> brow(TPT), tcol(TPT), bmask(TPT);
  for (int ti = 0; ti < TPT; ++ti) {
    const std::string off = cconst(ti * WPG);
    const std::string ni =
        e.val("arith.addi " + ng + ", " + off + " : index");
    const std::string ni16 =
        e.val("arith.muli " + ni + ", " + C["i16"] + " : index");
    brow[ti] = e.val("arith.addi " + ni16 + ", " + l16 + " : index");
    const std::string ni32 =
        e.val("arith.index_cast " + ni16 + " : index to i32");
    const std::string nbase = e.val("arith.addi " + n0 + ", " + ni32 +
                                    " : i32");
    tcol[ti] = e.val("arith.addi " + nbase + ", " + l32 + " : i32");
    bmask[ti] = e.val("arith.andi " + brow[ti] + ", " + C["imask"] +
                      " : index");
  }
  const std::string amask =
      e.val("arith.andi " + arow + ", " + C["imask"] + " : index");

  // One swizzled 8-element chunk of a stage, at logical column `chunk`.
  const auto chunk = [&](const std::string &view, int nrows,
                         const std::string &row, const std::string &mask,
                         int chunk) {
    return e.val("memref.load " + view + "[" + row + ", " +
                 e.val("arith.xori " + mask + ", " + cconst(chunk) +
                       " : index") +
                 "] : memref<" + s(nrows) + "x" + s(NV8) +
                 "xvector<8xbf16>, " + SmemTy + ">");
  };
  // A WMMA fragment: the two swizzled halves of the 16-K slice, joined in
  // registers (only the lane's own copies move, so the shuffle is free).
  const auto fragment = [&](const std::string &view, int nrows,
                            const std::string &row, const std::string &mask,
                            int ks) {
    const std::string lo = chunk(view, nrows, row, mask, 2 * ks);
    const std::string hi = chunk(view, nrows, row, mask, 2 * ks + 1);
    const std::string v = e.val(
        "vector.shuffle " + lo + ", " + hi +
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : "
        "vector<8xbf16>, vector<8xbf16>");
    return e.val("arith.bitcast " + v +
                 " : vector<16xbf16> to vector<16xi16>");
  };

  // One A fragment per 16-K step, shared by all of the warp's tiles.
  const auto compute = [&](const SlabViews &v, std::vector<std::string> &acc) {
    for (int ks = 0; ks < KS; ++ks) {
      const std::string ai = fragment(v.a_8, BM, arow, amask, ks);
      for (int ti = 0; ti < TPT; ++ti) {
        const std::string bi = fragment(v.b_8, BN, brow[ti], bmask[ti], ks);
        acc[ti] = e.val(
            "rocdl.wmma.f32.16x16x16.bf16 " + ai + ", " + bi + ", " + acc[ti] +
            " : (vector<16xi16>, vector<16xi16>, vector<8xf32>) -> "
            "vector<8xf32>");
      }
    }
  };

  std::vector<std::string> names;
  for (int i = 0; i < TPT; ++i) names.push_back("%w" + s(i));
  std::string ia = "iter_args(", ty = " -> (", res;
  for (int i = 0; i < TPT; ++i) {
    ia += (i ? ", " : "") + names[i] + " = " + C["czero"];
    ty += (i ? ", " : "") + std::string("vector<8xf32>");
    res += (i ? ", " : "") + std::string("%g") + s(i);
  }
  ia += ")";
  ty += ")";

  if (DB) {
    // Prologue: stage K step 0 into buffer 0 and publish it.
    const SlabViews v0 = mkviews(C["i0"]);
    stage("w", BM, v0.a_8, v0.a_s, m0, "%m", CHA, "icha", C["i0"], "a0");
    stage("x", BN, v0.b_8, v0.b_s, n0, "%n", CHB, "ichb", C["i0"], "b0");
    e.op("gpu.barrier");
  }

  e.op(res + " = scf.for %kt = " + C["i0"] + " to " + ktmax + " step " +
       C["i1"] + " " + ia + ty + " {");
  {
    std::vector<std::string> cur = names;
    if (DB) {
      const std::string buf =
          e.val("arith.remui %kt, " + C["i2"] + " : index");
      const std::string nbuf =
          e.val("arith.subi " + C["i1"] + ", " + buf + " : index");
      const std::string kt1 =
          e.val("arith.addi %kt, " + C["i1"] + " : index");
      const std::string has =
          e.val("arith.cmpi slt, " + kt1 + ", " + ktmax + " : index");
      const std::string boff =
          e.val("arith.muli " + buf + ", " + C["ipair"] + " : index");
      const std::string noff =
          e.val("arith.muli " + nbuf + ", " + C["ipair"] + " : index");
      // Stage the next step into the other buffer while this one computes; the
      // single trailing barrier both publishes that stage and keeps every warp
      // from overwriting the buffer it just read.
      e.op("scf.if " + has + " {");
      const SlabViews vn = mkviews(noff);
      stage("w", BM, vn.a_8, vn.a_s, m0, "%m", CHA, "icha", kt1, "a1");
      stage("x", BN, vn.b_8, vn.b_s, n0, "%n", CHB, "ichb", kt1, "b1");
      e.op("}");
      const SlabViews vc = mkviews(boff);
      compute(vc, cur);
      e.op("gpu.barrier");
    } else {
      // Single buffer: the leading barrier waits for the previous step's
      // readers, the trailing one publishes the new stage.
      e.op("gpu.barrier");
      const SlabViews vc = mkviews(C["i0"]);
      stage("w", BM, vc.a_8, vc.a_s, m0, "%m", CHA, "icha", "%kt", "a0");
      stage("x", BN, vc.b_8, vc.b_s, n0, "%n", CHB, "ichb", "%kt", "b0");
      e.op("gpu.barrier");
      compute(vc, cur);
    }
    std::string yields, ytypes;
    for (int i = 0; i < TPT; ++i) {
      yields += (i ? ", " : "") + cur[i];
      ytypes += (i ? ", " : "") + std::string("vector<8xf32>");
    }
    e.op("scf.yield " + yields + " : " + ytypes);
  }
  e.op("}");

  // ---- store: Y[token][row] (token-major), f32 ----
  const std::string m64 = e.val("arith.extsi %m : i32 to i64");
  for (int ti = 0; ti < TPT; ++ti) {
    const std::string tok_ok =
        e.val("arith.cmpi slt, " + tcol[ti] + ", %n : i32");
    const std::string tok64 =
        e.val("arith.extsi " + tcol[ti] + " : i32 to i64");
    const std::string off0 =
        e.val("arith.muli " + tok64 + ", " + m64 + " : i64");
    for (int i = 0; i < ACC; ++i) {
      const int rowoff = 2 * i;
      std::string mrow_i = mbase;
      if (rowoff != 0)
        mrow_i = e.val("arith.addi " + mrow_i + ", " + C["c" + s(rowoff)] +
                       " : i32");
      const std::string mrow = e.val("arith.addi " + mrow_i + ", " + hi32 +
                                     " : i32");
      const std::string mrow_ok =
          e.val("arith.cmpi slt, " + mrow + ", %m : i32");
      const std::string ok =
          e.val("arith.andi " + mrow_ok + ", " + tok_ok + " : i1");
      e.op("scf.if " + ok + " {");
      const std::string mrow64 =
          e.val("arith.extsi " + mrow + " : i32 to i64");
      const std::string off =
          e.val("arith.addi " + off0 + ", " + mrow64 + " : i64");
      const std::string p = e.val("llvm.getelementptr %y[" + off +
                                  "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
      const std::string av =
          e.val("vector.extract %g" + s(ti) + "[" + s(i) +
                "] : f32 from vector<8xf32>");
      const std::string ys =
          e.val("arith.mulf " + av + ", %alpha : f32");
      e.op("llvm.store " + ys + ", " + p + " : f32, !llvm.ptr");
      e.op("}");
    }
  }
  e.op("gpu.return");
  e.op("}");

  std::string text = "module attributes {gpu.container_module} {\n";
  text += "  gpu.module @gemm [#nvvm.target<chip = \"sm_" + s(gfx_arch) +
          "\">] {\n";
  text += "      " + FUNC + "\n" + e.body + "\n";
  text += "  }\n}\n";
  return text;
}

} // namespace

bool dense_gemm_shape_supported(const DenseGemmShape &shape) {
  if (shape.bm <= 0 || shape.bn <= 0 || shape.bk <= 0) return false;
  if (shape.bm % 16 != 0 || shape.bn % 8 != 0 || shape.bk % 16 != 0)
    return false;
#if defined(SONICBOOM_USE_ROCM)
  // The AMD body is a grid of 16x16 WMMA tiles, 4 or 8 warps per block. A warp
  // owns one 16-row strip and TPT tiles across the columns, so the strip count
  // must divide the warp count, the tiles must divide evenly, and a lane holds
  // at most 8 accumulator tiles (64 f32 registers).
  if (shape.bn % 16 != 0) return false;
  const int mt = shape.bm / 16;
  const int tiles = mt * (shape.bn / 16);
  const int warps = tiles >= 8 ? 8 : 4;
  if (mt > warps || warps % mt != 0) return false;
  if (tiles % warps != 0 || tiles / warps > 8) return false;
  return dense_gemm_shared_bytes(shape) <= kAmdDenseGemmLdsBytes;
#else
  if (shape.bm / 16 > 8) return false;  // at most 256 threads
  if (shape.bn / 8 > 16) return false;  // at most 16 f32 accumulators/lane
  // Up to 96 KiB of dynamic shared memory; the loader raises the per-function
  // limit above 48 KiB.
  return dense_gemm_shared_bytes(shape) <= 96 * 1024;
#endif
}

#if defined(SONICBOOM_USE_ROCM)
// AMD: 4 warps for a small tile grid, 8 when there is a tile per warp to fill.
// One whole 16x16 WMMA tile per warp slot; the emitter uses the same rule.
static int amd_dense_gemm_threads(const DenseGemmShape &shape) {
  const int tiles = (shape.bm / 16) * (shape.bn / 16);
  return (tiles >= 8 ? 8 : 4) * 32;
}
#endif

int dense_gemm_block_threads(const DenseGemmShape &shape) {
#if defined(SONICBOOM_USE_ROCM)
  return amd_dense_gemm_threads(shape);
#else
  return (shape.bm / 16) * 32;
#endif
}

int dense_gemm_shared_bytes(const DenseGemmShape &shape) {
  const int row = shape.bk + 8;
#if defined(SONICBOOM_USE_ROCM)
  // One staged slab each for W and X (see gemm_region_text_amd), rows padded to
  // a 32-byte boundary so each WMMA fragment is one aligned LDS read. The slabs
  // are double-buffered when both pairs fit, which the emitter decides with the
  // same rule.
  const int arow = shape.bk + kAmdWmmaPad;
  const int pair = (shape.bm * arow + shape.bn * arow) * 2;
  return 2 * pair <= kAmdDenseGemmLdsBytes ? 2 * pair : pair;
#else
  // Double-buffered A and B stages, each row padded by 8 elements (see
  // gemm_region_text) so ldmatrix does not read every row from the same bank.
  return 2 * (shape.bm * row + shape.bn * row) * 2;
#endif
}

std::optional<std::string> generate_dense_gemm_ptx(const DenseGemmShape &shape,
                                                   int sm_arch) {
  if (!dense_gemm_shape_supported(shape)) return std::nullopt;
  const std::string text = gemm_region_text(shape, sm_arch);
  return lower_gpu_module_to_ptx(text, /*link_libdevice=*/false, "dense-gemm");
}

std::optional<std::string> generate_dense_gemm_hsaco(const DenseGemmShape &shape,
                                                     int gfx_arch) {
#if defined(SONICBOOM_USE_ROCM)
  if (gfx_arch <= 0 || !dense_gemm_shape_supported(shape)) return std::nullopt;
  const std::string text = gemm_region_text_amd(shape, gfx_arch);
  const AmdKernelBlock block{kDenseGemmKernelEntry,
                             amd_dense_gemm_threads(shape)};
  const std::string adapted = amdgpu_adapt_module_text(
      text, gfx_arch, std::span<const AmdKernelBlock>(&block, 1));
  return lower_gpu_module_to_hsaco(adapted, /*link_libdevice=*/false,
                                   "dense-gemm");
#else
  (void)shape;
  (void)gfx_arch;
  return std::nullopt;
#endif
}

std::string dense_gemm_codegen_version() {
  // gemm2: padded shared staging (conflict-free ldmatrix); gemm3: scalar
  // scale epilogue (the folded `scale` consumer); gemm4: the AMD scalar body;
  // gemm5: the AMD WMMA body; gemm6: 16-byte vectorised and fully unrolled
  // AMD staging, double buffering where the LDS budget allows, one shared A
  // fragment per warp and an XOR-swizzled LDS row layout (the version keys the
  // disk cache for both backends).
  return std::string("llvm-") + LLVM_VERSION_STRING + "-gemm6";
}

} // namespace sonicboom::compiler
