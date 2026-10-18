// SPDX-License-Identifier: Apache-2.0
// Host twins of the element (method/element.hpp), written from its definitions with the arithmetic
// the kernels use (plain operators in a translation unit compiled with -ffp-contract=off, so that
// every operation rounds separately): central-difference gradients, the boxed products, the
// element's d and e, the hat weights fed with the device's logs, the rung cube by the vHGW twin
// (twin.hpp), the gather of the 2 m + 1 points with the reflected D-linear interpolation, and a
// long-double Jacobi reference of the eigenvectors.
#pragma once

#include "twin.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace element_twin {

using host_twin::Shape;

inline void coords(const Shape& sh, int64_t i, int32_t* x) {
  for (int a = sh.dim - 1; a >= 0; --a) {
    x[a] = static_cast<int32_t>(i % sh.n[a]);
    i /= sh.n[a];
  }
}

// The gradient along axis a (interior (r[i+1] - r[i-1]) / 2, one-sided ends, 0 for an extent 1):
// the input of the structure-tensor products that feed the eigen pass in the tests.
template <class T>
std::vector<T> gradient(const Shape& sh, const std::vector<T>& r, int a) {
  std::vector<T> g(r.size(), T(0));
  const int64_t n = sh.n[a], st = sh.stride(a);
  if (n < 2) return g;
  for (int64_t i = 0; i < sh.count(); ++i) {
    int32_t x[3];
    coords(sh, i, x);
    if (x[a] == 0) g[i] = r[i + st] - r[i];
    else if (x[a] == n - 1) g[i] = r[i] - r[i - st];
    else g[i] = (r[i + st] - r[i - st]) * T(0.5);
  }
  return g;
}

// cube(h, op, w): max / min along the axes (any order), box along 0..D-1 then fl_T(1 / prod).
template <class T>
std::vector<T> cube(const Shape& sh, int op, std::vector<T> f, int32_t w) {
  if (op == 3) {
    for (int a = 0; a < sh.dim; ++a) host_twin::vhgw_axis<T>(sh, a, w, host_twin::kSum, f);
    const T fac = host_twin::box_factor<T>(sh, w, 1.0);
    for (T& v : f) v = v * fac;
    return f;
  }
  for (int a = sh.dim - 1; a >= 0; --a) host_twin::vhgw_axis<T>(sh, a, w, op == 0 ? host_twin::kMax : host_twin::kMin, f);
  return f;
}

// The boxed products J_k, k over (a <= b) in the order (0,0), (0,1), (0,2), (1,1), (1,2), (2,2).
template <class T>
std::vector<std::vector<T>> structure_tensor(const Shape& sh, const std::vector<T>& r, int32_t W) {
  std::vector<std::vector<T>> g, J;
  for (int a = 0; a < sh.dim; ++a) g.push_back(gradient(sh, r, a));
  for (int a = 0; a < sh.dim; ++a) {
    for (int b = a; b < sh.dim; ++b) {
      std::vector<T> p(r.size());
      for (size_t i = 0; i < r.size(); ++i) p[i] = g[a][i] * g[b][i];
      J.push_back(cube(sh, 3, p, W));
    }
  }
  return J;
}

// Long-double reference: cyclic Jacobi to convergence; c = (mu1 - mu2) / (mu1 + mu2) clipped, n the
// top eigenvector.
inline void eigen_ref(int D, const long double (&J0)[3][3], long double* c, long double* n) {
  long double A[3][3], V[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A[i][j] = i < D && j < D ? J0[i][j] : 0.0L;
      V[i][j] = i == j ? 1.0L : 0.0L;
    }
  }
  for (int sweep = 0; sweep < 30; ++sweep) {
    long double off = 0, nrm = 0;
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < D; ++j) (i == j ? nrm : off) += A[i][j] * A[i][j];
    }
    if (off <= 1e-50L * nrm || off == 0) break;
    for (int p = 0; p < D; ++p) {
      for (int q = p + 1; q < D; ++q) {
        if (A[p][q] == 0) continue;
        const long double th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const long double t = (th >= 0 ? 1.0L : -1.0L) / (std::fabs(th) + std::sqrt(th * th + 1));
        const long double cs = 1 / std::sqrt(t * t + 1), sn = t * cs;
        for (int k = 0; k < D; ++k) {
          const long double akp = A[k][p], akq = A[k][q];
          A[k][p] = cs * akp - sn * akq;
          A[k][q] = sn * akp + cs * akq;
        }
        for (int k = 0; k < D; ++k) {
          const long double apk = A[p][k], aqk = A[q][k];
          A[p][k] = cs * apk - sn * aqk;
          A[q][k] = sn * apk + cs * aqk;
        }
        for (int k = 0; k < D; ++k) {
          const long double vkp = V[k][p], vkq = V[k][q];
          V[k][p] = cs * vkp - sn * vkq;
          V[k][q] = sn * vkp + cs * vkq;
        }
      }
    }
  }
  int top = 0;
  for (int a = 1; a < D; ++a) top = A[a][a] > A[top][top] ? a : top;
  long double mu2 = -INFINITY;
  for (int a = 0; a < D; ++a) {
    if (a != top && A[a][a] > mu2) mu2 = A[a][a];
  }
  const long double mu1 = A[top][top], s = mu1 + mu2;
  long double cv = s > 0 ? (mu1 - mu2) / s : 0;
  *c = cv < 0 ? 0 : (cv > 1 ? 1 : cv);
  for (int a = 0; a < D; ++a) n[a] = V[a][top];
}

// d = L ((1 - c) + c / q), e = 0.5 max(L (1 + (span - 1) c) - d, 0) in T.
template <class T>
void element_de(T L, T c, double q, double span, T* d, T* e) {
  const T f = (T(1) - c) + c / static_cast<T>(q);
  *d = L * f;
  const T len = L * (T(1) + static_cast<T>(span - 1.0) * c);
  const T diff = len - *d;
  *e = T(0.5) * (diff > T(0) ? diff : T(0));
}

inline int64_t reflect(int64_t i, int64_t n) {
  int64_t m = i % (2 * n);
  if (m < 0) m += 2 * n;
  return m < n ? m : 2 * n - 1 - m;
}

// D-linear interpolation of the half-sample reflected periodic extension of f at y (double).
template <class T>
T interp(const Shape& sh, const std::vector<T>& f, const double* y) {
  const int D = sh.dim;
  int64_t o0[3], o1[3];
  T w0[3], w1[3];
  for (int a = 0; a < D; ++a) {
    const double fl = std::floor(y[a]);
    const double tt = y[a] - fl;
    const int64_t i0 = static_cast<int64_t>(fl);
    w1[a] = static_cast<T>(tt);
    w0[a] = T(1) - w1[a];
    o0[a] = reflect(i0, sh.n[a]) * sh.stride(a);
    o1[a] = reflect(i0 + 1, sh.n[a]) * sh.stride(a);
  }
  T acc = T(0);
  for (int corner = 0; corner < (1 << D); ++corner) {
    int64_t off = 0;
    for (int a = 0; a < D; ++a) off += ((corner >> (D - 1 - a)) & 1) ? o1[a] : o0[a];
    T term = f[off];
    for (int a = 0; a < D; ++a) term = term * (((corner >> (D - 1 - a)) & 1) ? w1[a] : w0[a]);
    acc = acc + term;
  }
  return acc;
}

// The element of one batch element, as the operators read it.
template <class T>
struct ElemHost {
  int32_t K = 0;
  std::vector<int32_t> rung_w;
  uint32_t used = 0, pts = 0;
  bool active = true;
  std::vector<uint8_t> lo;
  std::vector<T> t;
  std::vector<std::vector<T>> o;  // D components (D >= 2)
};

// _apply(h, op): op 0 max, 1 min, 3 box; the output starts as `out` (a voxel's first positive
// weight stores, the second adds; voxels are always stored once before any add).
template <class T>
std::vector<T> apply(const Shape& sh, const ElemHost<T>& el, const std::vector<T>& h, int op, std::vector<T> out,
                     int m = 2) {
  if (!el.active) return out;
  const int D = sh.dim;
  for (int j = 0; j < el.K; ++j) {
    if (((el.used >> j) & 1u) == 0u) continue;
    const std::vector<T> f = cube(sh, op, h, el.rung_w[j]);
    const bool pts = D >= 2 && ((el.pts >> j) & 1u) != 0u;
    for (int64_t i = 0; i < sh.count(); ++i) {
      const int l = el.lo[i];
      const T tt = el.t[i];
      const T wl = T(1) - tt;
      T wgt;
      bool add;
      if (l == j && wl > T(0)) {
        wgt = wl;
        add = false;
      } else if (l + 1 == j && tt > T(0)) {
        wgt = tt;
        add = wl > T(0);
      } else {
        continue;
      }
      int32_t x[3];
      coords(sh, i, x);
      T acc = f[i];
      if (pts) {
        for (int k = 1; k <= m; ++k) {
          const double s = static_cast<double>(k) / static_cast<double>(m);
          for (int sg = 0; sg < 2; ++sg) {
            double y[3];
            for (int a = 0; a < D; ++a) {
              const double off = s * static_cast<double>(el.o[a][i]);
              y[a] = sg == 0 ? static_cast<double>(x[a]) + off : static_cast<double>(x[a]) - off;
            }
            const T v = interp(sh, f, y);
            if (op == 0) acc = host_twin::max_c(acc, v);
            else if (op == 1) acc = host_twin::min_c(acc, v);
            else acc = acc + v;
          }
        }
        if (op == 3) acc = acc / static_cast<T>(2 * m + 1);
      }
      const T v = wgt * acc;
      out[i] = add ? out[i] + v : v;
    }
  }
  return out;
}

}  // namespace element_twin
