// SPDX-License-Identifier: Apache-2.0
// The orthonormal N-D DCT-II and its inverse (DCT-III) of a padded field batch, by Makhoul's
// algorithm on one N-D complex cuFFT plan, run forward and inverse in place:
//
//   forward  r --permute--> u (+ 0 i) --DFT--> V --post-twiddle--> X
//   inverse  X1, X2 --pre-twiddle--> V = V1 + i V2 --inverse DFT--> u1 + i u2 --unpermute--> r1, r2
//
// u is the field with every axis permuted (u[m] = v[2m] for m < ceil(n/2), u[n-1-m] = v[2m+1]),
// V its full complex spectrum (B, n_0..n_{D-1}), dense. The forward post-twiddle reads the half
// k_{D-1} <= n_{D-1}/2 of V; the inverse takes two real fields at once, one in the real part and
// one in the imaginary part (the spectra of real fields are Hermitian, so the two separate exactly
// in exact arithmetic). The twiddles combine the 2^D spectrum values of an orbit {+-k_0} x .. x
// {+-k_{D-1}} per thread. The inverse is normalized in the pre-twiddle (1/N).
//
// Complex transforms, not real ones: a batched complex transform computes every batch entry from
// that entry alone, so an element's result never depends on the other elements of the batch.
// cuFFT's batched real-to-complex transforms compute two batch entries together for many sizes (an
// odd last axis of 39 and more, for example), which made an element's rounding depend on its
// neighbour. float32 uses C2C, float64 Z2Z. The plan is made for one geometry and batch. The
// orientation (method/orientation.cu) runs its own permutations and pre-twiddle around the
// transforms; the forward post-twiddle is enqueue_dct_post.
#pragma once

#include "field/geom.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdint>

namespace eda {

// The layouts of one call, by value in every DCT kernel.
struct DctShape {
  int32_t dim;
  int32_t n[kMaxDim];
  int32_t nc;             // n_{D-1} / 2 + 1: the orbit indices along the last axis
  int32_t tw_off[kMaxDim];  // offset of axis a in the twiddle tables (n_a + 1 entries each)
  int64_t batch;
  int64_t pitch;
  int64_t field;          // real padded elements per element
  int64_t rows;           // padded rows per element
  int64_t cfield;         // complex elements per element: prod_a n_a (the dense spectrum)
  int64_t orbits;         // orbit threads per element: prod_{a<D-1} (n_a/2 + 1) * nc
};
DctShape dct_shape(const Geom& g);

// The cuFFT plan and twiddle tables of one geometry, batch and dtype.
class DctPlans {
 public:
  DctPlans(const Geom& g, bool f64, cudaStream_t stream);
  ~DctPlans();
  DctPlans(const DctPlans&) = delete;
  DctPlans& operator=(const DctPlans&) = delete;

  const DctShape& shape() const noexcept { return shape_; }
  bool f64() const noexcept { return f64_; }
  // Twiddles per axis a and k = 0..n_a: cos(pi k / (2 n_a)), sin(pi k / (2 n_a)), the
  // orthonormal scale sqrt((k ? 2 : 1) / n_a) and its inverse, in the dtype (float or double).
  const void* tw_cos() const noexcept { return tw_; }
  const void* tw_sin() const noexcept;
  const void* tw_scale() const noexcept;
  const void* tw_iscale() const noexcept;

  // The forward or inverse (unnormalized) DFT of V (B * cfield complex values) in place, on s.
  void exec_forward(cudaStream_t s, void* V) const;
  void exec_inverse(cudaStream_t s, void* V) const;

 private:
  void create(const Geom& g, cudaStream_t stream);
  void destroy() noexcept;  // what create() made, also when it threw part way
  void exec(cudaStream_t s, void* V, int direction, const char* what) const;
  DctShape shape_{};
  bool f64_ = false;
  cufftHandle plan_ = 0;
  bool have_plan_ = false;
  void* tw_ = nullptr;  // 4 tables of total (sum_a (n_a + 1)) entries
  size_t tw_entries_ = 0;
  int device_ = 0;
};

// The forward post-twiddle V (complex, B * cfield) -> X (real padded field) of D = 1, 2 or 3.
template <class T>
void enqueue_dct_post(cudaStream_t s, const DctPlans& p, const void* V, T* X);

}  // namespace eda
