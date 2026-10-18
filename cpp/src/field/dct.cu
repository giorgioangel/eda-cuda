// SPDX-License-Identifier: Apache-2.0
// The N-D Makhoul DCT (dct.hpp): the cuFFT plan, the twiddle tables and the forward post-twiddle.
#include "runtime/error.hpp"
#include "runtime/memory.hpp"
#include "field/dct.cuh"
#include "field/dct.hpp"
#include "decompose.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace eda {
namespace {

constexpr int kThreads = 256;

unsigned grid_for(int64_t work, int per_cta) {
  const int64_t g = (work + per_cta - 1) / per_cta;
  return static_cast<unsigned>(std::clamp<int64_t>(g, 1, int64_t{1} << 20));
}

template <class T>
DctTw<T> tables(const DctPlans& p) {
  return DctTw<T>{static_cast<const T*>(p.tw_cos()), static_cast<const T*>(p.tw_sin()),
                  static_cast<const T*>(p.tw_scale()), static_cast<const T*>(p.tw_iscale())};
}

template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_dct_post(DctShape d, DctTw<T> tw, const cplx_t<T>* __restrict__ V,
                                                         T* __restrict__ X) {
  const int64_t total = d.batch * d.orbits;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = idx / d.orbits;
    int32_t k[kMaxDim];
    dct_orbit<D>(d, idx - b * d.orbits, k);
    dct_post_orbit<T, D>(d, tw, k, V + b * d.cfield, X + b * d.field);
  }
}

template <class F>
void dispatch_dim(int dim, F&& f) {
  switch (dim) {
    case 1: f(std::integral_constant<int, 1>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    default: throw std::invalid_argument("eda: dct: D must be 1, 2 or 3");
  }
}

// Throws OutOfDeviceMemory for CUFFT_ALLOC_FAILED, std::runtime_error otherwise.
void cufft_check(cufftResult r, const char* what) {
  if (r == CUFFT_SUCCESS) return;
  if (r == CUFFT_ALLOC_FAILED) {
    throw OutOfDeviceMemory(std::string("eda: cuFFT could not allocate its work area (") + what +
                            "); decompose fewer elements per call");
  }
  throw std::runtime_error(std::string("eda: cuFFT error ") + std::to_string(static_cast<int>(r)) + " in " + what);
}

// Makes the plan of `handle` (the basic layout: dense batch entries). cuFFT reports a work area it
// cannot allocate as CUFFT_ALLOC_FAILED, or for some plans as CUFFT_INTERNAL_ERROR, also when the
// work area fits but the memory the plan needs beside it does not. A failure throws
// OutOfDeviceMemory when the work area (asked on a fresh handle) exceeds the free device memory, or
// when the same plan can be made without its work area (so the work area is what did not fit).
void make_plan(cufftHandle handle, int D, long long* n, cufftType type, long long batch, const char* what) {
  size_t ws = 0;
  const cufftResult r = cufftMakePlanMany64(handle, D, n, nullptr, 1, 0, nullptr, 1, 0, type, batch, &ws);
  if (r == CUFFT_SUCCESS || r == CUFFT_ALLOC_FAILED) return cufft_check(r, what);
  size_t need = 0, free_bytes = 0, total_bytes = 0;
  bool short_of_memory = false;
  cufftHandle probe = 0;
  if (cufftCreate(&probe) == CUFFT_SUCCESS) {
    short_of_memory =
        cufftGetSizeMany64(probe, D, n, nullptr, 1, 0, nullptr, 1, 0, type, batch, &need) == CUFFT_SUCCESS &&
        cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && need > free_bytes;
    if (!short_of_memory && cufftSetAutoAllocation(probe, 0) == CUFFT_SUCCESS) {
      size_t unused = 0;
      short_of_memory =
          cufftMakePlanMany64(probe, D, n, nullptr, 1, 0, nullptr, 1, 0, type, batch, &unused) == CUFFT_SUCCESS;
    }
    (void)cufftDestroy(probe);
  }
  (void)cudaGetLastError();
  if (short_of_memory) {
    throw OutOfDeviceMemory(std::string("eda: cuFFT could not allocate the work area of ") + what + " (" +
                            format_gib(static_cast<int64_t>(need)) + " and the plan's own memory, " +
                            format_gib(static_cast<int64_t>(free_bytes)) +
                            " free); decompose fewer elements per call");
  }
  cufft_check(r, what);
}

}  // namespace

DctShape dct_shape(const Geom& g) {
  DctShape d{};
  d.dim = g.dim;
  for (int a = 0; a < kMaxDim; ++a) d.n[a] = g.n[a];
  d.nc = g.n[g.dim - 1] / 2 + 1;
  int32_t off = 0;
  for (int a = 0; a < g.dim; ++a) {
    d.tw_off[a] = off;
    off += g.n[a] + 1;
  }
  d.batch = g.batch;
  d.pitch = g.pitch;
  d.field = g.field;
  d.rows = g.rows;
  d.cfield = g.n[g.dim - 1];
  d.orbits = d.nc;
  for (int a = 0; a < g.dim - 1; ++a) {
    d.cfield *= g.n[a];
    d.orbits *= g.n[a] / 2 + 1;
  }
  return d;
}

DctPlans::DctPlans(const Geom& g, bool f64, cudaStream_t stream) : shape_(dct_shape(g)), f64_(f64) {
  EDA_CUDA(cudaGetDevice(&device_));
  try {
    create(g, stream);
  } catch (...) {
    destroy();
    throw;
  }
}

void DctPlans::create(const Geom& g, cudaStream_t stream) {
  const bool f64 = f64_;
  const int D = g.dim;
  long long n[kMaxDim];
  for (int a = 0; a < D; ++a) n[a] = g.n[a];
  cufft_check(cufftCreate(&plan_), "cufftCreate");
  have_plan_ = true;
  make_plan(plan_, D, n, f64 ? CUFFT_Z2Z : CUFFT_C2C, g.batch, "the DCT plan");
  // Twiddles in double with the host's libm, rounded once to the dtype. They depend on n_a only,
  // so every element of every batch and chunk sees the same values.
  tw_entries_ = 0;
  for (int a = 0; a < D; ++a) tw_entries_ += static_cast<size_t>(g.n[a]) + 1;
  std::vector<double> cs(tw_entries_), sn(tw_entries_), sc(tw_entries_), isc(tw_entries_);
  for (int a = 0; a < D; ++a) {
    const int32_t na = g.n[a];
    for (int32_t k = 0; k <= na; ++k) {
      const size_t i = static_cast<size_t>(shape_.tw_off[a] + k);
      const double th = M_PI * static_cast<double>(k) / (2.0 * na);
      cs[i] = std::cos(th);
      sn[i] = std::sin(th);
      sc[i] = std::sqrt((k ? 2.0 : 1.0) / na);
      isc[i] = std::sqrt(na / (k ? 2.0 : 1.0));
    }
  }
  const size_t es = f64 ? sizeof(double) : sizeof(float);
  EDA_CUDA(cudaMalloc(&tw_, 4 * tw_entries_ * es));
  std::vector<unsigned char> host(4 * tw_entries_ * es);
  for (int t = 0; t < 4; ++t) {
    const std::vector<double>& v = t == 0 ? cs : (t == 1 ? sn : (t == 2 ? sc : isc));
    for (size_t i = 0; i < tw_entries_; ++i) {
      if (f64) {
        reinterpret_cast<double*>(host.data())[t * tw_entries_ + i] = v[i];
      } else {
        reinterpret_cast<float*>(host.data())[t * tw_entries_ + i] = static_cast<float>(v[i]);
      }
    }
  }
  EDA_CUDA(cudaMemcpyAsync(tw_, host.data(), host.size(), cudaMemcpyHostToDevice, stream));
  EDA_CUDA(cudaStreamSynchronize(stream));
}

DctPlans::~DctPlans() { destroy(); }

void DctPlans::destroy() noexcept {
  int previous = -1;
  const bool switched = cudaGetDevice(&previous) == cudaSuccess && previous != device_ && cudaSetDevice(device_) == cudaSuccess;
  if (have_plan_) (void)cufftDestroy(plan_);
  if (tw_) (void)cudaFree(tw_);
  have_plan_ = false;
  tw_ = nullptr;
  if (switched) (void)cudaSetDevice(previous);
}

const void* DctPlans::tw_sin() const noexcept {
  return static_cast<const char*>(tw_) + tw_entries_ * (f64_ ? sizeof(double) : sizeof(float));
}
const void* DctPlans::tw_scale() const noexcept {
  return static_cast<const char*>(tw_) + 2 * tw_entries_ * (f64_ ? sizeof(double) : sizeof(float));
}
const void* DctPlans::tw_iscale() const noexcept {
  return static_cast<const char*>(tw_) + 3 * tw_entries_ * (f64_ ? sizeof(double) : sizeof(float));
}

void DctPlans::exec(cudaStream_t s, void* V, int direction, const char* what) const {
  cufft_check(cufftSetStream(plan_, s), "cufftSetStream");
  if (f64_) {
    auto* z = static_cast<cufftDoubleComplex*>(V);
    cufft_check(cufftExecZ2Z(plan_, z, z, direction), what);
  } else {
    auto* c = static_cast<cufftComplex*>(V);
    cufft_check(cufftExecC2C(plan_, c, c, direction), what);
  }
}

void DctPlans::exec_forward(cudaStream_t s, void* V) const { exec(s, V, CUFFT_FORWARD, "the forward DCT"); }
void DctPlans::exec_inverse(cudaStream_t s, void* V) const { exec(s, V, CUFFT_INVERSE, "the inverse DCT"); }

template <class T>
void enqueue_dct_post(cudaStream_t s, const DctPlans& p, const void* V, T* X) {
  const DctShape& d = p.shape();
  dispatch_dim(d.dim, [&](auto D) {
    eda_dct_post<T, D()><<<grid_for(d.batch * d.orbits, kThreads), kThreads, 0, s>>>(
        d, tables<T>(p), static_cast<const cplx_t<T>*>(V), X);
  });
  EDA_CHECK_LAUNCH("eda_dct_post");
}

template void enqueue_dct_post<float>(cudaStream_t, const DctPlans&, const void*, float*);
template void enqueue_dct_post<double>(cudaStream_t, const DctPlans&, const void*, double*);

}  // namespace eda
