// SPDX-License-Identifier: Apache-2.0
// test_e2e: the end-to-end equivalences of the driver (method/driver.cpp): a batch element's rows
// do not depend on its position, on the chunking (a forced small chunk) or on repeated runs (also
// in a new thread with its own context); a NaN element fails alone (error 1); a constant element
// has no IMF;
// max_imf caps the IMFs and keeps the first rows; the rows sum to the input. float32 and float64,
// D = 1, 2, 3.
#include "runtime/context.hpp"
#include "twin.hpp"
#include "decompose.hpp"
#include "test_support.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using host_twin::Shape;

namespace {

const bool kSmall = std::getenv("EDA_SANITIZE") && std::string(std::getenv("EDA_SANITIZE")) == "1";

Shape shape(int dim) {
  Shape s;
  s.dim = dim;
  // the EDA_SANITIZE sizes keep a racecheck pass over the statistics kernels within 5 minutes
  if (dim == 1) {
    s.n[0] = kSmall ? 128 : 512;
  } else if (dim == 2) {
    s.n[0] = kSmall ? 16 : 48;
    s.n[1] = kSmall ? 14 : 40;
  } else {
    s.n[0] = kSmall ? 9 : 20;
    s.n[1] = kSmall ? 8 : 18;
    s.n[2] = kSmall ? 7 : 16;
  }
  return s;
}

template <class T>
struct Out {
  std::vector<T> rows;
  std::vector<int32_t> n_imfs, error, scales, sifts;
  int32_t P = 0;
  int64_t chunk = 0;
};

template <class T>
Out<T> run(eda::Context& ctx, const Shape& sh, int64_t B, const std::vector<T>& x, const eda::Params& p) {
  host_twin::GpuRun<T> g = host_twin::run_host<T>(ctx, sh, B, x, p);
  Out<T> o;
  o.rows = std::move(g.rows);
  o.n_imfs = g.run->n_imfs;
  o.error = g.run->error;
  o.scales = g.run->hist_scale;
  o.sifts = g.run->hist_sifts;
  o.P = g.run->rows;
  o.chunk = g.run->chunk;
  return o;
}

// Element b's rows [0, n_imfs] of a run.
template <class T>
std::vector<T> rows_of(const Out<T>& o, int64_t b, int64_t count) {
  const int32_t k = o.error[b] ? 0 : o.n_imfs[b] + 1;
  const T* p = o.rows.data() + static_cast<size_t>(b) * o.P * count;
  return std::vector<T>(p, p + static_cast<size_t>(k) * count);
}

template <class T>
void check_case(int dim) {
  eda::Context& ctx = eda::Context::current(0);
  const Shape sh = shape(dim);
  const int64_t N = sh.count();
  const int64_t B = 5;
  // elements: smooth waves + noise (two seeds), noise, a constant, one with a NaN
  std::vector<T> x;
  for (int64_t b = 0; b < B; ++b) {
    const int kind = b == 2 ? 0 : (b == 3 ? 3 : 2);
    std::vector<T> f = host_twin::make_field<T>(sh, kind, 700 + 13 * b + dim);
    if (b == 4) f[static_cast<size_t>(N / 3)] = std::numeric_limits<T>::quiet_NaN();
    x.insert(x.end(), f.begin(), f.end());
  }
  eda::Params p;
  const Out<T> all = run<T>(ctx, sh, B, x, p);
  // the NaN element fails alone; the constant has no IMF and its residue is the input
  for (int64_t b = 0; b < B; ++b) CHECK(all.error[b] == (b == 4 ? 1 : 0));
  CHECK(all.n_imfs[3] == 0);
  {
    const std::vector<T> r3 = rows_of(all, 3, N);
    CHECK(r3.size() == static_cast<size_t>(N) && host_twin::same_bits(r3.data(), x.data() + 3 * N, N));
  }
  CHECK(all.n_imfs[0] >= 1 && all.n_imfs[1] >= 1);
  // reconstruction (4 eps P max|x|)
  for (int64_t b = 0; b < 3; ++b) {
    const std::vector<T> r = rows_of(all, b, N);
    const int32_t P = all.n_imfs[b] + 1;
    double amax = 0, err = 0;
    for (int64_t i = 0; i < N; ++i) {
      double s = 0;
      for (int32_t k = 0; k < P; ++k) s += static_cast<double>(r[static_cast<size_t>(k * N + i)]);
      amax = std::max(amax, std::fabs(static_cast<double>(x[b * N + i])));
      err = std::max(err, std::fabs(s - static_cast<double>(x[b * N + i])));
    }
    CHECK(err <= 4.0 * std::numeric_limits<T>::epsilon() * P * amax);
    for (int32_t k = 0; k < all.n_imfs[b]; ++k) {
      const int32_t sf = all.sifts[static_cast<size_t>(b) * (all.sifts.size() / B) + k];
      CHECK(sf >= 1 && sf <= 3);
    }
  }
  // batch position: every element alone (not under EDA_SANITIZE, where the forced chunk of 1
  // below runs every element alone through the same driver: a racecheck pass over the statistics
  // kernels pays per launch, and the separate runs took it past 5 minutes)
  for (int64_t b = 0; b < B; ++b) {
    if (kSmall) break;
    const std::vector<T> xb(x.begin() + b * N, x.begin() + (b + 1) * N);
    const Out<T> one = run<T>(ctx, sh, 1, xb, p);
    CHECK(one.error[0] == all.error[b] && one.n_imfs[0] == all.n_imfs[b]);
    if (!all.error[b]) {
      const std::vector<T> a = rows_of(all, b, N), o = rows_of(one, 0, N);
      CHECK(a.size() == o.size() && host_twin::same_bits(a.data(), o.data(), a.size()));
    }
  }
  // the batch reversed
  if (!kSmall) {
    std::vector<T> xr;
    for (int64_t b = B - 1; b >= 0; --b) xr.insert(xr.end(), x.begin() + b * N, x.begin() + (b + 1) * N);
    const Out<T> rev = run<T>(ctx, sh, B, xr, p);
    for (int64_t b = 0; b < B; ++b) {
      if (all.error[b]) continue;
      const std::vector<T> a = rows_of(all, b, N), o = rows_of(rev, B - 1 - b, N);
      CHECK(a.size() == o.size() && host_twin::same_bits(a.data(), o.data(), a.size()));
    }
  }
  // a forced small chunk
  for (int64_t lim : {int64_t{1}, int64_t{2}}) {
    if (kSmall && lim == 2) continue;
    eda::Params pc = p;
    pc.max_chunk = lim;
    const Out<T> ch = run<T>(ctx, sh, B, x, pc);
    CHECK(ch.chunk == lim);
    CHECK(ch.P == all.P && ch.n_imfs == all.n_imfs && ch.error == all.error && ch.scales == all.scales);
    for (int64_t b = 0; b < B; ++b) {
      if (all.error[b]) continue;
      const std::vector<T> a = rows_of(all, b, N), o = rows_of(ch, b, N);
      CHECK(host_twin::same_bits(a.data(), o.data(), a.size()));
    }
  }
  // repeated runs: on the same context, and on a new thread's own context (under EDA_SANITIZE the
  // new thread's run only)
  {
    if (!kSmall) {
      const Out<T> again = run<T>(ctx, sh, B, x, p);
      CHECK(again.rows.size() == all.rows.size() &&
            host_twin::same_bits(again.rows.data(), all.rows.data(), all.rows.size()));
    }
    Out<T> off;
    host_twin::in_new_thread(0, [&](eda::Context& c2) { off = run<T>(c2, sh, B, x, p); });
    CHECK(off.rows.size() == all.rows.size() && host_twin::same_bits(off.rows.data(), all.rows.data(), all.rows.size()));
  }
  // max_imf: at most that many IMFs, the first IMF unchanged, the rows still sum to x
  {
    eda::Params q = p;
    q.max_imf = 1;
    const Out<T> m1 = run<T>(ctx, sh, B, x, q);
    for (int64_t b = 0; b < 3; ++b) {
      CHECK(m1.n_imfs[b] == std::min(1, all.n_imfs[b]));
      const std::vector<T> a = rows_of(all, b, N), o = rows_of(m1, b, N);
      CHECK(host_twin::same_bits(a.data(), o.data(), static_cast<size_t>(N)));
    }
  }
}

}  // namespace

int main() {
  // Under EDA_SANITIZE=1 (compute-sanitizer): float64 in every D and float32 in 2D, on the small
  // shapes, so that a pass stays within its time budget.
  try {
    for (int dim = 1; dim <= 3; ++dim) {
      check_case<double>(dim);
      if (!kSmall || dim == 2) check_case<float>(dim);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test_e2e: %s\n", error.what());
    eda::Context::shutdown();
    return 1;
  }
  // release the pinned host blocks and the pools (memcheck's leak check at exit)
  eda::Context::shutdown();
  return eda_test::finish("test_e2e");
}
