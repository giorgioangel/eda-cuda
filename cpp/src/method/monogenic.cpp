// SPDX-License-Identifier: Apache-2.0
// eda.monogenic (method/method.hpp): the monogenic signal of every field of a call, in chunks of
// elements sized by the free device memory like the decomposition's. Per chunk: the input copy
// (host input staged, CUDA input read in place or converted, as the driver does), the row partials
// with the non-finite count, the exact prescale r 2^-k with max|r| in [1, 2) (the driver's), and
// enqueue_monogenic (method/element.hpp) into the call's output. An element's result depends on its
// own samples only: every pass is per element and the transforms run in the orientation's groups.
#include "runtime/context.hpp"
#include "runtime/device.hpp"
#include "runtime/error.hpp"
#include "runtime/memory.hpp"
#include "decompose.hpp"
#include "field/convert.hpp"
#include "field/field_ops.hpp"
#include "field/reduce.hpp"
#include "method/element.hpp"
#include "method/method.hpp"
#include "method/mode.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace eda {
namespace {

// The workspace of a chunk: the field f, R1, D_a f, n (one slot per component), the work fields,
// c, the orientation's group workspace, the row partials, the element states, the prescale
// exponents and the staged input.
template <class T>
struct MonogenicLayout {
  ArenaLayout lay;
  ArenaSlot<T> f, r1[kMaxDim], df[kMaxDim], n[kMaxDim], work, c, staged;
  ArenaSlot<unsigned char> orient;
  ArenaSlot<RowPartial> rows;
  ArenaSlot<ElemState> elems;
  ArenaSlot<int32_t> exp;

  MonogenicLayout(const Geom& g, bool stage) {
    const size_t field = static_cast<size_t>(g.batch) * static_cast<size_t>(g.field);
    const int D = g.dim;
    f = lay.add<T>(field);
    for (int a = 0; a < kMaxDim; ++a) {
      r1[a] = lay.add<T>(a < D ? field : 0);
      df[a] = lay.add<T>(a < D ? field : 0);
      n[a] = lay.add<T>(a < D && D >= 2 ? field : 0);
    }
    work = lay.add<T>(static_cast<size_t>(D * (D + 1) / 2) * field);
    c = lay.add<T>(D >= 2 ? field : 0);
    staged = lay.add<T>(stage ? static_cast<size_t>(g.batch) * static_cast<size_t>(g.count) : 0);
    orient = lay.add<unsigned char>(orientation_workspace_bytes(g, sizeof(T) == 4 ? DType::F32 : DType::F64));
    rows = lay.add<RowPartial>(static_cast<size_t>(g.batch) * static_cast<size_t>(g.rows));
    elems = lay.add<ElemState>(static_cast<size_t>(g.batch));
    exp = lay.add<int32_t>(static_cast<size_t>(g.batch));
  }
};

template <class T>
int64_t monogenic_t(Context& ctx, const Input& in, bool check_finite, int64_t max_chunk, T* out,
                    std::vector<int32_t>& nonfinite) {
  DeviceGuard guard(ctx.device());
  const Geom g = make_geom(in.dim, in.batch, in.n);
  const int64_t B = g.batch;
  cudaStream_t stream = ctx.stream();
  const bool stage = !in.on_device || in.source != nullptr;
  const DType dtype = sizeof(T) == 4 ? DType::F32 : DType::F64;
  const double tiny = method_ctl(g, dtype).tiny;  // FLT_MIN (float32), 1e-300 (float64)
  const DctPlans& plans = orientation_plans(ctx, g, dtype);
  // the chunk: the largest batch whose workspace fits the free device memory (halving)
  int64_t Bc = max_chunk > 0 ? std::min(B, max_chunk) : B;
  {
    const int64_t avail = device_available_bytes(ctx.device());
    const int64_t margin = std::max<int64_t>(int64_t{256} << 20, avail / 20);
    for (;;) {
      const int64_t need = static_cast<int64_t>(MonogenicLayout<T>(chunk_geom(g, Bc), stage).lay.bytes());
      if (need + margin <= avail) break;
      if (Bc == 1) {
        throw OutOfDeviceMemory("eda: one element needs a workspace of " + format_gib(need) + " and " +
                                format_gib(avail) + " of device memory are free");
      }
      Bc = (Bc + 1) / 2;
    }
  }
  nonfinite.assign(static_cast<size_t>(B), 0);
  std::vector<ElemState> host(static_cast<size_t>(Bc));
  for (int64_t b0 = 0; b0 < B; b0 += Bc) {
    const int64_t nb = std::min(Bc, B - b0);
    const Geom gc = chunk_geom(g, nb);
    const MonogenicLayout<T> L(gc, stage);
    const ArenaBinding ws = bind_arena(ctx, L.lay);
    T* f = ws.get(L.f);
    T* staged = ws.get(L.staged);
    const T* x;
    InputStrides xs;
    if (in.source) {
      enqueue_convert_gather<T>(stream, *in.source, in.x, b0, gc, staged);
      x = staged;
      xs = c_strides(gc);
    } else if (stage) {
      EDA_CUDA(cudaMemcpyAsync(staged, static_cast<const T*>(in.x) + b0 * g.count,
                               static_cast<size_t>(nb) * g.count * sizeof(T), cudaMemcpyHostToDevice, stream));
      x = staged;
      xs = c_strides(gc);
    } else {
      x = static_cast<const T*>(in.x) + b0 * in.strides.s[0];
      xs = in.strides;
    }
    ElemState* elems = ws.get(L.elems);
    RowPartial* rows = ws.get(L.rows);
    int32_t* exp = ws.get(L.exp);
    EDA_CUDA(cudaMemsetAsync(elems, 0, sizeof(ElemState) * static_cast<size_t>(nb), stream));
    enqueue_canonical_in<T>(stream, gc, x, xs, f);
    enqueue_row_partials<T>(stream, gc, f, rows, check_finite ? elems : nullptr);
    enqueue_reduce_partials(stream, gc, rows, elems);
    enqueue_mode_prescale<T>(stream, gc, elems, exp, f);
    MonogenicFields<T> mf{};
    for (int a = 0; a < g.dim; ++a) {
      mf.r1.p[a] = ws.get(L.r1[a]);
      mf.df.p[a] = ws.get(L.df[a]);
      mf.n.p[a] = ws.get(L.n[a]);
    }
    mf.work = ws.get(L.work);
    mf.c = ws.get(L.c);
    enqueue_monogenic<T>(stream, gc, plans, f, exp, mf, ws.get(L.orient), tiny, out, B, b0);
    if (check_finite) {
      EDA_CUDA(cudaMemcpyAsync(host.data(), elems, sizeof(ElemState) * static_cast<size_t>(nb), cudaMemcpyDeviceToHost,
                               stream));
      ctx.synchronize();
      for (int64_t b = 0; b < nb; ++b) nonfinite[static_cast<size_t>(b0 + b)] = host[static_cast<size_t>(b)].nonfinite ? 1 : 0;
    }
  }
  return Bc;
}

}  // namespace

void validate_monogenic(const Input& in) { validate_method(in, Params{}); }

size_t monogenic_bytes(const Input& in) {
  int64_t count = 1;
  for (int a = 0; a < in.dim; ++a) count *= in.n[a];
  return static_cast<size_t>(in.batch) * static_cast<size_t>(count) * static_cast<size_t>(monogenic_rows(in.dim)) *
         (in.dtype == DType::F32 ? sizeof(float) : sizeof(double));
}

int64_t run_monogenic(Context& ctx, const Input& in, bool check_finite, int64_t max_chunk, void* out,
                      std::vector<int32_t>& nonfinite) {
  validate_monogenic(in);
  if (out == nullptr) throw std::invalid_argument("eda: monogenic: null output");
  if (max_chunk < 0) throw std::invalid_argument("eda: max_chunk must be >= 0");
  if (in.dtype == DType::F32) {
    return monogenic_t<float>(ctx, in, check_finite, max_chunk, static_cast<float*>(out), nonfinite);
  }
  return monogenic_t<double>(ctx, in, check_finite, max_chunk, static_cast<double*>(out), nonfinite);
}

}  // namespace eda
