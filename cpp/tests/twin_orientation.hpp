// SPDX-License-Identifier: Apache-2.0
// A long-double twin of the orientation (method/orientation.cu), written from the definitions
// and sharing no code with the kernels:
//   - f = r - mean(r); X = the orthonormal N-D DCT-II of f by direct sums along each axis,
//     X_k = c_k sum_x f_x cos(pi k (2x+1) / (2n)), c_k = sqrt((k ? 2 : 1) / n);
//   - the band X_b = G(rad) X, u_a = k_a / (2 n_a) (the frequency of DCT index k in the 2 n_a-sample
//     half-sample symmetric extension), rad = |u|, G the one-octave log-Gabor at f0 = 1 / w,
//     G(rad) = exp(-(log(rad w))^2 / (2 sigma^2)), sigma = log 2 / (2 sqrt(2 log 2)), G(0) = 0;
//   - the Riesz components of the band, from the continuous multipliers -i u_a / |u| (first order)
//     and their products (second order). The extension is a sum of plane waves of frequency
//     (+-u_0, .., +-u_{D-1}) per DCT index; on such a sum, -i sign(u_a) turns cos(theta_a) into
//     sin(theta_a), theta_a = pi k_a (2 x_a + 1) / (2 n_a), so:
//       R1_a  = sum_k c X_b (u_a / rad) sin(theta_a) prod_{b != a} cos(theta_b)
//       R2_aa = sum_k c X_b (-(u_a / rad)^2) prod_b cos(theta_b)
//       R2_ab = sum_k c X_b (u_a u_b / rad^2) sin(theta_a) sin(theta_b) prod_{others} cos(theta)
//     (c = prod_a c_{k_a}): inverse transforms with the sin basis on the odd axes;
//   - the energy tensor T = R1 R1^T + R2 R2^T and its eigen-decomposition by cyclic Jacobi to
//     convergence: n the top eigenvector, c = (mu1 - mu2) / (mu1 + mu2) clipped to [0, 1]
//     (0 where mu1 + mu2 <= 0);
//   - the monogenic signal (monogenic): no band (G = 1 but G(0) = 0), the same R1, R2 and the
//     derivatives in cycles per sample, D_a = d/dx_a / (2 pi), which turn cos(theta_a) into
//     -u_a sin(theta_a):
//       D_a f   = sum_k c X (-u_a) sin(theta_a) prod_{b != a} cos(theta_b)
//       D_a R_a = sum_k c X (u_a^2 / rad) prod_b cos(theta_b)
//       D_b R_a = sum_k c X (-u_a u_b / rad) sin(theta_a) sin(theta_b) prod_{others} cos(theta)
//     and per voxel, for a given unit n: q = n . R1, A = sqrt(f^2 + |R1|^2), the phase
//     atan2(q, f) and the frequency (f D_n q - q D_n f) / A^2.
// The angle tables hold cos(pi j / (2n)), sin(pi j / (2n)) for j < 4n, read at k (2x+1) mod 4n, so no
// large arguments are reduced. Cost O(N sum_a n_a) per transform.
#pragma once

#include "twin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace orient_twin {

using LD = long double;
using host_twin::Shape;

constexpr LD kPi = 3.141592653589793238462643383279502884L;

// The component order of the device: R2 (and T) in the pairs (a <= b) of tensor_pair.
inline int pair_index(int D, int a, int b) {
  if (a > b) std::swap(a, b);
  int p = 0;
  for (int q = 0; q < D; ++q) {
    for (int s = q; s < D; ++s, ++p) {
      if (q == a && s == b) return p;
    }
  }
  return -1;
}

inline void coords(const Shape& sh, int64_t i, int32_t* x) {
  for (int a = sh.dim - 1; a >= 0; --a) {
    x[a] = static_cast<int32_t>(i % sh.n[a]);
    i /= sh.n[a];
  }
}

// The band G(rad) of window w.
inline LD band(LD rad, int32_t w) {
  if (!(rad > 0)) return 0;
  const LD l2 = std::log(2.0L), sg = l2 / (2 * std::sqrt(2 * l2));
  const LD l = std::log(rad * static_cast<LD>(w));
  return std::exp(-(l * l) / (2 * sg * sg));
}

// The matrix of one axis transform of length n: kind 0 forward DCT-II (row k, column x), 1 the
// inverse cos basis (row x, column k), 2 the inverse sin basis (row x, column k).
inline std::vector<LD> axis_matrix(int64_t n, int kind) {
  std::vector<LD> cs(4 * n), sn(4 * n), m(n * n);
  for (int64_t j = 0; j < 4 * n; ++j) {
    cs[j] = std::cos(kPi * static_cast<LD>(j) / static_cast<LD>(2 * n));
    sn[j] = std::sin(kPi * static_cast<LD>(j) / static_cast<LD>(2 * n));
  }
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const int64_t k = kind == 0 ? i : j, x = kind == 0 ? j : i;
      const LD c = std::sqrt((k ? 2.0L : 1.0L) / static_cast<LD>(n));
      const int64_t q = (k * (2 * x + 1)) % (4 * n);
      m[i * n + j] = c * (kind == 2 ? sn[q] : cs[q]);
    }
  }
  return m;
}

// f <- M f along axis a.
inline void axis_apply(const Shape& sh, int a, const std::vector<LD>& M, std::vector<LD>& f) {
  const int64_t n = sh.n[a], st = sh.stride(a), outer = sh.count() / (n * st);
  std::vector<LD> line(n);
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t s = 0; s < st; ++s) {
      const int64_t base = o * n * st + s;
      for (int64_t i = 0; i < n; ++i) line[i] = f[base + i * st];
      for (int64_t i = 0; i < n; ++i) {
        LD acc = 0;
        const LD* row = &M[i * n];
        for (int64_t j = 0; j < n; ++j) acc += row[j] * line[j];
        f[base + i * st] = acc;
      }
    }
  }
}

// The axis matrices of one shape (forward, inverse cos, inverse sin per axis).
struct Plans {
  Shape sh;
  std::vector<LD> m[3][3];
  explicit Plans(const Shape& s) : sh(s) {
    for (int a = 0; a < s.dim; ++a) {
      for (int k = 0; k < 3; ++k) m[a][k] = axis_matrix(s.n[a], k);
    }
  }
  // kind 0: forward DCT-II on every axis; otherwise the inverse with the sin basis on the axes of
  // the bit mask odd.
  void dct(std::vector<LD>& f) const {
    for (int a = 0; a < sh.dim; ++a) axis_apply(sh, a, m[a][0], f);
  }
  void inverse(std::vector<LD>& f, int odd) const {
    for (int a = 0; a < sh.dim; ++a) axis_apply(sh, a, m[a][((odd >> a) & 1) ? 2 : 1], f);
  }
};

struct Riesz {
  LD nu = 0;                   // ||r - mean||_2
  std::vector<LD> Xb;          // the band
  std::vector<LD> R1[3];       // first order, per axis
  std::vector<LD> R2[6];       // second order, pair_index order
};

// The spectrum of one element r (N values, C order): nu = ||r - mean||_2, X the DCT of r - mean
// times the band of window w (w = 0: no band, X(0) = 0), the frequencies uk (D blocks of N) and
// rad = |u|.
struct Spectrum {
  LD nu = 0;
  std::vector<LD> X, uk, rad;
};
inline Spectrum spectrum(const Plans& P, const std::vector<LD>& r, int32_t w) {
  const Shape& sh = P.sh;
  const int D = sh.dim;
  const int64_t N = sh.count();
  Spectrum out;
  LD mean = 0;
  for (int64_t i = 0; i < N; ++i) mean += r[i];
  mean /= static_cast<LD>(N);
  std::vector<LD>& f = out.X;
  f.resize(N);
  for (int64_t i = 0; i < N; ++i) {
    f[i] = r[i] - mean;
    out.nu += f[i] * f[i];
  }
  out.nu = std::sqrt(out.nu);
  P.dct(f);
  out.uk.resize(D * N);
  out.rad.resize(N);
  for (int64_t i = 0; i < N; ++i) {
    int32_t k[3];
    coords(sh, i, k);
    LD r2 = 0;
    for (int a = 0; a < D; ++a) {
      out.uk[a * N + i] = static_cast<LD>(k[a]) / static_cast<LD>(2 * sh.n[a]);
      r2 += out.uk[a * N + i] * out.uk[a * N + i];
    }
    out.rad[i] = std::sqrt(r2);
    f[i] *= w > 0 ? band(out.rad[i], w) : (out.rad[i] > 0 ? 1 : 0);
  }
  return out;
}

// The inverse transform of X times mult(i) (0 where rad = 0) with the sin basis on the axes of odd.
template <class M>
inline std::vector<LD> component(const Plans& P, const Spectrum& S, int odd, M mult) {
  const int64_t N = P.sh.count();
  std::vector<LD> y(N, 0);
  for (int64_t i = 0; i < N; ++i) y[i] = S.rad[i] > 0 ? S.X[i] * mult(i) : 0;
  P.inverse(y, odd);
  return y;
}

// The Riesz components of one element r (N values, C order) in the band of window w.
inline Riesz riesz(const Plans& P, const std::vector<LD>& r, int32_t w) {
  const Shape& sh = P.sh;
  const int D = sh.dim;
  const int64_t N = sh.count();
  Riesz out;
  Spectrum S = spectrum(P, r, w);
  out.nu = S.nu;
  const std::vector<LD>& f = S.X;
  const std::vector<LD>& uk = S.uk;
  const std::vector<LD>& rad = S.rad;
  out.Xb = f;
  for (int a = 0; a < D; ++a) {
    std::vector<LD> y(N);
    for (int64_t i = 0; i < N; ++i) y[i] = rad[i] > 0 ? f[i] * (uk[a * N + i] / rad[i]) : 0;
    P.inverse(y, 1 << a);
    out.R1[a] = std::move(y);
  }
  for (int a = 0; a < D; ++a) {
    for (int b = a; b < D; ++b) {
      std::vector<LD> y(N);
      for (int64_t i = 0; i < N; ++i) {
        if (!(rad[i] > 0)) continue;
        const LD ua = uk[a * N + i] / rad[i], ub = uk[b * N + i] / rad[i];
        y[i] = f[i] * (a == b ? -ua * ua : ua * ub);
      }
      P.inverse(y, a == b ? 0 : ((1 << a) | (1 << b)));
      out.R2[pair_index(D, a, b)] = std::move(y);
    }
  }
  return out;
}

// h(a, b) = R2_ab at voxel i (symmetric).
inline LD h_at(const Riesz& R, int D, int a, int b, int64_t i) { return R.R2[pair_index(D, a, b)][i]; }

// The components of the monogenic signal of one element r: R (R1, R2 without the band), Df
// (D_a f) and DR (D_b R_a, pair_index order).
struct Monogenic {
  Riesz R;
  std::vector<LD> Df[3], DR[6];
};
inline Monogenic monogenic(const Plans& P, const std::vector<LD>& r) {
  const int D = P.sh.dim;
  const int64_t N = P.sh.count();
  const Spectrum S = spectrum(P, r, 0);
  Monogenic M;
  M.R.nu = S.nu;
  M.R.Xb = S.X;
  const auto u = [&](int a, int64_t i) { return S.uk[a * N + i]; };
  for (int a = 0; a < D; ++a) {
    M.R.R1[a] = component(P, S, 1 << a, [&](int64_t i) { return u(a, i) / S.rad[i]; });
    M.Df[a] = component(P, S, 1 << a, [&](int64_t i) { return -u(a, i); });
    for (int b = a; b < D; ++b) {
      const int odd = a == b ? 0 : ((1 << a) | (1 << b));
      M.R.R2[pair_index(D, a, b)] = component(P, S, odd, [&](int64_t i) {
        const LD v = u(a, i) * u(b, i) / (S.rad[i] * S.rad[i]);
        return a == b ? -v : v;
      });
      M.DR[pair_index(D, a, b)] = component(P, S, odd, [&](int64_t i) {
        const LD v = u(a, i) * u(b, i) / S.rad[i];
        return a == b ? v : -v;
      });
    }
  }
  return M;
}

// The monogenic outputs at voxel i for the input value f and the unit direction n (1 in 1D).
struct MonogenicAt {
  LD q, a2, dnf, dnq, A, phase, freq;
};
inline MonogenicAt monogenic_at(const Monogenic& M, int D, int64_t i, LD f, const LD* n) {
  MonogenicAt o{};
  o.a2 = f * f;
  for (int a = 0; a < D; ++a) {
    o.q += n[a] * M.R.R1[a][i];
    o.a2 += M.R.R1[a][i] * M.R.R1[a][i];
    o.dnf += n[a] * M.Df[a][i];
    for (int b = 0; b < D; ++b) o.dnq += n[a] * n[b] * M.DR[pair_index(D, a, b)][i];
  }
  o.A = std::sqrt(o.a2);
  if (o.a2 > 0) {
    o.phase = std::atan2(o.q, f);
    o.freq = (f * o.dnq - o.q * o.dnf) / o.a2;
  }
  return o;
}

// T at voxel i into A (full symmetric D x D).
inline void tensor_at(const Riesz& R, int D, int64_t i, LD (&A)[3][3]) {
  for (int a = 0; a < 3; ++a) {
    for (int c = 0; c < 3; ++c) A[a][c] = 0;
  }
  for (int a = 0; a < D; ++a) {
    for (int c = a; c < D; ++c) {
      LD v = R.R1[a][i] * R.R1[c][i];
      for (int k = 0; k < D; ++k) v += h_at(R, D, a, k, i) * h_at(R, D, k, c, i);
      A[a][c] = A[c][a] = v;
    }
  }
}

// Cyclic Jacobi to convergence: eigenvalues mu descending, V[.][j] the eigenvector of mu[j].
inline void eig(int D, const LD (&A0)[3][3], LD (&mu)[3], LD (&V)[3][3]) {
  LD A[3][3], W[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A[i][j] = i < D && j < D ? A0[i][j] : 0;
      W[i][j] = i == j ? 1 : 0;
    }
  }
  for (int sweep = 0; sweep < 50; ++sweep) {
    LD off = 0, nrm = 0;
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < D; ++j) (i == j ? nrm : off) += A[i][j] * A[i][j];
    }
    if (off == 0 || off <= 1e-60L * nrm) break;
    for (int p = 0; p < D; ++p) {
      for (int q = p + 1; q < D; ++q) {
        if (A[p][q] == 0) continue;
        const LD th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const LD t = (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1));
        const LD cs = 1 / std::sqrt(t * t + 1), sn = t * cs;
        for (int k = 0; k < D; ++k) {
          const LD x = A[k][p], y = A[k][q];
          A[k][p] = cs * x - sn * y;
          A[k][q] = sn * x + cs * y;
        }
        for (int k = 0; k < D; ++k) {
          const LD x = A[p][k], y = A[q][k];
          A[p][k] = cs * x - sn * y;
          A[q][k] = sn * x + cs * y;
        }
        for (int k = 0; k < D; ++k) {
          const LD x = W[k][p], y = W[k][q];
          W[k][p] = cs * x - sn * y;
          W[k][q] = sn * x + cs * y;
        }
      }
    }
  }
  int order[3] = {0, 1, 2};
  std::sort(order, order + D, [&](int i, int j) { return A[i][i] > A[j][j]; });
  for (int j = 0; j < 3; ++j) {
    mu[j] = j < D ? A[order[j]][order[j]] : 0;
    for (int k = 0; k < 3; ++k) V[k][j] = j < D ? W[k][order[j]] : 0;
  }
}

// The coherence of the method from the eigenvalues.
inline LD coherence(LD mu1, LD mu2) {
  const LD s = mu1 + mu2;
  if (!(s > 0)) return 0;
  const LD c = (mu1 - mu2) / s;
  return c < 0 ? 0 : (c > 1 ? 1 : c);
}

}  // namespace orient_twin
