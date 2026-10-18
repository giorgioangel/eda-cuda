# eda-cuda user guide

This guide covers `eda.decompose`, `eda.monogenic` and the types around them, what a call
guarantees and what it does not, and how the tests check each claim. The [README](../README.md)
has the install steps and a short tour.

Contents: [The call](#the-call) · [Sifts per mode](#sifts-per-mode) · [The result](#the-result) ·
[DecompositionInfo](#decompositioninfo) · [Orientation](#orientation) · [Monogenic
signal](#monogenic-signal) · [Errors](#errors) · [Device arrays and
streams](#device-arrays-and-streams) · [Memory](#memory) · [Environment
variables](#environment-variables) · [Determinism and batches](#determinism-and-batches) ·
[Limits](#limits) · [Properties of the method](#properties-of-the-method)

## The call

```python
eda.decompose(x, axes=None, max_imf=-1, *, dtype=None, tie_tolerance=None, sifts=3,
              check_finite=True, errors="raise", stream=None, return_info=False,
              return_orientation=False)
```

`x` is a NumPy array (or anything `numpy.asarray` accepts), or a CUDA array that implements
DLPack or the CUDA array interface (CuPy, PyTorch and others).

| Argument | Rule |
|---|---|
| `axes` | An int or a sequence of one to three distinct ints, negative values allowed: the spatial axes, in the order the method sees them. The other axes form the batch. `None` takes every axis and needs `x.ndim <= 3`; for a larger array name the axes. Each spatial extent must be at least 2 and at most 8192 (float32) or 4096 (float64). |
| `max_imf` | At most this many IMFs per element. 0 or a negative value means no limit. Must be an integer, not a bool. |
| `dtype` | `"float32"` or `"float64"`. The compute dtype and the dtype of the result. By default float32 and float64 input keep their dtype. Any other real dtype (bool, integers, float16, and bfloat16 of a CUDA array) raises `TypeError` unless `dtype` casts it first. The cast rounds to nearest, as `numpy.ndarray.astype` does, so host and CUDA input with the same values give the same bits. Complex input raises `TypeError`. |
| `tie_tolerance` | A finite number >= 0, relative to the largest magnitude of the residual: two samples closer than this count as equal when extrema are counted. Default `2**-14` (float32) and `2**-30` (float64). A tolerance above the largest possible prominence (for example 2.5) finds no mode: the residue is the input. |
| `sifts` | The number of sifts per mode, an integer from 1 up; default 3. See [Sifts per mode](#sifts-per-mode). |
| `check_finite` | `True` (default): an element with inf or NaN fails with error code 1. `False` skips the check; a non-finite value then ends in code 2 or NaN rows. |
| `errors` | `"raise"` (default) or `"status"`; see [Errors](#errors). |
| `stream` | See [Device arrays and streams](#device-arrays-and-streams). Ignored, but type-checked, for host input. |
| `return_info` | `True` returns `(y, info)`. Must be a bool. |
| `return_orientation` | `True` also returns an `eda.Orientation`, the direction and coherence of every mode (see [Orientation](#orientation)), after `info` when `return_info` is set. Must be a bool; with one spatial axis it raises `ValueError`. |

Arguments of the wrong type raise `TypeError`, wrong values `ValueError`; the messages begin with
`eda:`.

## Sifts per mode

A sift subtracts the mean of the upper and lower envelopes from what is left of the input. The
first sift leaves part of every oscillation slower than the mode's window size in the mode; each
further sift takes out more of it. The number of sifts therefore sets how narrow a mode's band is.

- `sifts=3`, the default, is the value the method was designed and calibrated with. The
  continuation weight and the amplitude normalization that sifts 2 and later use were tuned for
  three sifts. Other counts are supported but were not calibrated.
- `sifts=1` is the single-sift form: smoother modes, with more of the next slower oscillation left
  in each.
- More than three sifts take out a little more of the slower oscillations and split noise and
  smooth fields into more modes.
- There is no option to sift until convergence, because this sift does not settle. It is not a
  contraction: during the development of the method, the envelope mean of an amplitude-modulated
  wave in 2-D and 3-D grew with every sift instead of shrinking. With this build, 40 sifts still
  change the first three modes by up to 48 % of the input's largest magnitude against 12 sifts, on
  the six 1-D and 2-D inputs tried.
- A sift that would make the mode or the remainder larger than the residual it started from ends
  the sifting of that mode early. `info.sifts` reports the sifts each mode completed, from 1 to
  `sifts`.
- The time per mode grows with the count.

Measured with this build on an NVIDIA RTX 4090 in float32 on 2026-10-16. The inputs are two
superposed plane waves, one 12 times slower than the other, on 4096 samples (1-D), 256 × 256 (2-D)
and 64 × 64 × 64 (3-D). The table gives the share of the slower wave left in the first mode, the
mean number of modes from eight white-noise signals of 4096 samples, and the time of one 2-D call
after a warm-up.

| `sifts` | slower wave in mode 1, 1-D | 2-D | 3-D | modes from white noise | 2-D time |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.3 % | 5.9 % | 12 % | 6.25 | 7.4 ms |
| 2 | 0.27 % | 0.50 % | 2.1 % | 7.1 | 9.8 ms |
| 3 | 0.21 % | 0.17 % | 0.68 % | 7.75 | 11.8 ms |
| 5 | 0.21 % | 0.15 % | 0.28 % | 8.6 | 16.4 ms |
| 8 | 0.20 % | 0.14 % | 0.22 % | 9.5 | 30.0 ms |

Every count keeps the guarantees of the default: the rows add up to the input up to the rounding
of the subtractions, an element's result does not depend on the rest of the batch, and host and
CUDA input give the same bits (`tests/python/test_sifts.py`).

## The result

`y` has the shape of `x` with a mode axis inserted before the first spatial axis:

```
y.shape == x.shape[:m] + (P,) + x.shape[m:],   m = min(axes)
```

For `x` of shape `(4, 128, 128)` and `axes=(1, 2)` the result is `(4, P, 128, 128)`; for a 1-D
`x` it is `(P, n)`. Along the mode axis each element holds its IMFs from the finest to the
coarsest, then its residue, then zero rows. `P` is one more than the largest IMF count among the
elements without an error, so an input without oscillation gives `P = 1` and the residue equals
the input. The method sees the spatial axes in the order of `axes`, and `y` keeps the axis order
of `x`: `axes=(1, 0)` gives, transposed back, the decomposition of the transposed array. The rows
of an element add up to the element, see [Properties of the method](#properties-of-the-method).

Host input returns a NumPy array. CUDA input returns an `eda.DeviceArray` on the device of the
input (below). The result is in the compute dtype.

The call returns `y`, then `info` if `return_info`, then the `Orientation` if
`return_orientation`: `y`, `(y, info)`, `(y, orientation)` or `(y, info, orientation)`.

## DecompositionInfo

`return_info=True` adds a frozen dataclass of NumPy arrays. With `B` the batch shape (empty for a
single field) and `K` the largest IMF count in the call, the arrays have shape `B` or `B + (K,)`.
In the `B + (K,)` arrays, entries beyond an element's `n_imfs`, and all entries of a failed
element, are 0; a failed element also has `n_imfs` 0.

| Field | Shape | Meaning |
|---|---|---|
| `n_imfs` | `B` | IMFs found per element (int32); the residue row follows them. |
| `error` | `B` | 0, 1 (inf or NaN in the input with `check_finite`) or 2 (a non-finite intermediate or output row, for example an overflow of the dtype's range). |
| `windows` | `B + (K,)` | The window size of each mode, in samples (int32). |
| `spacing` | `B + (K,)` | The mean spacing of the significant extrema from which the window size was computed, in samples (float64). |
| `sifts` | `B + (K,)` | The sifts each mode completed, 1 to the call's `sifts` (int32). |
| `energy` | `B + (K,)` | The sum of the squared IMF, in the units of `x` (float64). |
| `dtype` | | The compute dtype, `"float32"` or `"float64"`. |
| `axes` | | The spatial axes, normalized to non-negative values, in the order given. |

## Orientation

In 2-D and 3-D the method turns the window of each mode across the local stripes (see
[Direction](../README.md#direction) in the README): at the start of the mode it computes, from
the residual in the band of the mode's window size, the direction n across the stripes and their
coherence c.
`return_orientation=True` returns a copy of both as a frozen dataclass. With `m = min(axes)`, `K`
the largest IMF count in the call (the `K` of `DecompositionInfo`) and `D` the number of spatial
axes:

| Field | Shape | Meaning |
|---|---|---|
| `direction` | `x.shape[:m] + (K, D) + x.shape[m:]` | The unit direction n of each mode at every sample. Component `a` is along the `a`-th axis of `axes`. The sign is the canonical one of `eda.monogenic`: the first component whose magnitude is above the dtype's default tie tolerance (2^-14 float32, 2^-30 float64) is positive. |
| `coherence` | `x.shape[:m] + (K,) + x.shape[m:]` | The coherence c of each mode, from 0 (no direction) to 1 (perfect stripes). |

Mode `k` is the mode of IMF row `k`. Rows beyond an element's `n_imfs`, and all rows of a failed
element, are 0. Host input gives NumPy arrays and CUDA input `eda.DeviceArray`s, in the compute
dtype. Asking for the orientation changes nothing else: the rows and the information are the same
bits with and without it (`tests/python/test_orientation_output.py`). On the device the
orientation takes `D + 1` values per sample and mode, beside the output rows (see
[Memory](#memory)). Like the rows, this room is set aside for the largest number of modes the
call could find, not only for the modes it finds. So when little device memory is free, asking
for the orientation can make the call work in smaller chunks and run slower; its results stay
the same.

## Monogenic signal

```python
eda.monogenic(x, axes=None, *, dtype=None, check_finite=True, stream=None) -> eda.Monogenic
```

The monogenic signal describes a field around every sample as a local wave: how strong it is (the
amplitude), where in its cycle the sample sits (the phase), how fast it oscillates (the
frequency) and, in 2-D and 3-D, which way it runs (the direction) and how clearly it has one
direction (the coherence). In 1-D it is the analytic signal. A typical input is the modes of a
decomposition, `eda.monogenic(y[:n], axes=...)`, whose amplitude and frequency then show how each
mode changes from place to place. The [README](../README.md#amplitude-phase-frequency-and-direction)
shows these quantities on its 1-D and 2-D examples.

`x`, `axes`, `dtype`, `check_finite` and `stream` follow the rules of `eda.decompose`: the same
input types and casts, the same axis rules and limits (the other axes form a batch), and the same
stream handling. Every field is processed on its own and gives the same bits alone or at any
position of any batch. With `check_finite=True` a field that holds inf or NaN raises `ValueError`
(naming its batch indices); with `False` its outputs are not defined (NaN in practice).

The result is a frozen dataclass. With `m = min(axes)` and `D` the number of spatial axes:

| Field | Shape | Meaning |
|---|---|---|
| `amplitude` | `x.shape` | A, the strength of the local wave. |
| `phase` | `x.shape` | φ in (−π, π]: the position in the local wave's cycle. |
| `frequency` | `x.shape` | ν, in cycles per sample along the direction. |
| `direction` | `x.shape[:m] + (D,) + x.shape[m:]` | n, component `a` along the `a`-th axis of `axes`; `None` in 1-D. |
| `coherence` | `x.shape` | c in [0, 1]; `None` in 1-D. |

Host input gives NumPy arrays and CUDA input `eda.DeviceArray`s, complete on return, in the
compute dtype.

**Definitions.** At every border the field is continued by its mirror image at half a sample, as
in the method, so nothing assumes that it is periodic (the transforms run on the DCT). The Riesz
transforms R_a generalize the Hilbert transform: on a plane wave f = cos θ, θ = 2π k·x + θ₀,

- R_a f = (k_a/|k|) sin θ, the first-order transform, a vector R1 = (R_a f);
- R_a R_b f = −(k_a k_b/|k|²) cos θ, the second-order transform, a matrix R2 = (R_a R_b f);
- ∂_b f = −2π k_b sin θ and ∂_b R_a f = 2π k_b (k_a/|k|) cos θ, the derivatives.

They act on the whole spectrum of the field and remove its mean (all of them are 0 at frequency
0). Then, at every sample:

- the direction n is the top eigenvector of the energy tensor T = R1 R1ᵀ + R2 R2ᵀ (A²k̂k̂ᵀ for a
  plane wave of amplitude A, at every phase), with the canonical sign: the first component, in
  the order of `axes`, whose magnitude is above the dtype's default tie tolerance (2^-14 float32,
  2^-30 float64) is positive. The tolerance keeps a component that is zero up to rounding, as
  the other components of stripes along one axis are, from deciding the sign. In 1-D n = 1;
- the coherence is c = (μ₁ − μ₂)/(μ₁ + μ₂) from the two largest eigenvalues of T, clipped to
  [0, 1] (0 where both are 0);
- q = n·R1, the part of the first-order transform along n;
- the amplitude is A = √(f² + |R1|²), with all of R1;
- the phase is φ = atan2(q, f), so φ = θ for the plane wave when n = k̂;
- the frequency is ν = (f ∂_n q − q ∂_n f)/(2π A²), the rate at which the phase changes along n,
  with ∂_n f = Σ_b n_b ∂_b f and ∂_n q = Σ_ab n_a n_b ∂_b R_a f (n taken as locally constant). It
  is |k| for the plane wave, and in 1-D the classical instantaneous frequency of the analytic
  signal.

f is the field as given: a constant offset adds to f in A and φ but not to the transforms. Where
the amplitude vanishes the phase and the frequency are 0: precisely, where A² of the field scaled
by a power of two to max |x| in [1, 2) is not above the floor the method divides by (FLT_MIN for
float32, 1e-300 for float64). As in the method's orientation, the transforms run in the compute
dtype and the multipliers and the arithmetic at every sample in double; every output is rounded
once to the compute dtype. In float32 the amplitude can exceed the float32 range (about 3.4e38)
for input close to that range, and is then inf: unlike `decompose`, which reports an overflow as
error code 2, `monogenic` does not report it.

**What the tests check** (`tests/python/test_monogenic.py`, and against a long-double twin in
`cpp/tests/test_orientation.cu`):

- DCT basis fields cos(πm(x + ½)/n), in 1-D and along one axis of a 2-D or 3-D field: A = 1,
  ν = m/(2n) and φ = πm(x + ½)/n at every sample, and n the unit vector of that axis with c = 1,
  within a few units of roundoff (at most 7.5 eps measured).
- A plane wave that is not a DCT basis field bends at the mirror, and the Riesz transforms reach
  far, so its values hold only away from the borders: two periods from every border, A is 1
  within 3.2e-3 and ν is |k| within 1.9e-4 on 128 × 128 (k = (0.031, 0.047)), and within 3.0e-4
  and 1.0e-4 on 48³ (k = (0.05, 0.07, 0.09)).
- A NumPy implementation of the same definitions with the FFT of the mirrored field agrees within
  a few eps of the field's scale on smooth random fields in 1-D, 2-D and 3-D.
- Bit for bit: an element alone and at any position of a batch, host and CUDA input, and a gain
  by a power of two (A scales; the phase, the frequency, n and c keep their bits).

A call keeps, per element of a chunk, 4 working fields in the compute dtype in 1-D and 3·D + D(D +
1)/2 + 2 in 2-D and 3-D (11 and 17), and the output of 3 (1-D) or D + 4 (2-D, 3-D) values per
sample on the device; a batch that does not fit is processed in chunks, as `decompose` does.
Measured with this build on an RTX 4090 in float32 (median of 10 calls after a warm-up): a 1024 ×
1024 field takes 3.7 ms as a CuPy array and 7.9 ms as a NumPy array (with the copies), a 128³
field 16.7 ms and 26.1 ms.

## Errors

| Exception | When |
|---|---|
| `TypeError`, `ValueError` | An argument is wrong (above), or `EDA_*` environment variables are wrong. |
| `eda.DecompositionError` (a `ValueError`) | With `errors="raise"`, at least one element failed. `indices` holds the flat C-order batch indices of the failed elements, `codes` their codes and `info` the `DecompositionInfo` of the call. The message lists the first indices of each code. |
| `eda.CudaError` (a `RuntimeError`) | A CUDA call failed, or no usable GPU exists. `code` and `name` hold the `cudaError_t` and its name when known. |
| `MemoryError` | The device memory does not suffice, see [Memory](#memory). |
| `RuntimeError` | An internal check failed (error code 255); this is a bug in eda, please report it. |

With `errors="status"` the call returns: the rows of a failed element are NaN and `info.error`
holds its code. Both exception classes of eda can be pickled.

`eda.device_count()` is 0 without a driver or device. `eda.device_info(i)` returns the properties of
a visible device (`ValueError` for one that is not visible) and `eda.build_info()` how the
extension was built (CUDA versions, architectures, compilers, the cuFFT version mapped into the
process); `build_info` and `device_count` work without a GPU. `import eda` needs neither a GPU nor
the extension; the extension loads at the first call that needs it.

## Device arrays and streams

CUDA input is read where it is, without a copy, when its layout allows it. Another dtype, or batch
axes that do not merge into one stride, are converted on the GPU first. The result is an
`eda.DeviceArray` with `shape`, `strides` (bytes), `dtype`, `device`, `ndim`, `size` and `nbytes`.
It implements DLPack (`__dlpack__`, `__dlpack_device__`) and the CUDA array interface (version 3),
so libraries read it without a copy:

```python
cupy.asarray(y)          cupy.from_dlpack(y)
torch.from_dlpack(y)     torch.as_tensor(y)
```

It owns its memory; users do not construct it. Its data is complete when the call returns, so
any stream may read it.

**Streams.** The call runs on a stream of its own, never on the caller's. Before it reads a CUDA
array it waits for the producer's pending writes:

- An object with both DLPack and the CUDA array interface is read through DLPack. The call asks the
  producer for the data with `__dlpack__(stream=s)`, where `s` is `stream` when given and
  otherwise the legacy default stream (handle 1), and waits for `s`. The producer makes `s` wait
  for its current stream, so data written on another stream needs `stream=` or a block that makes
  that stream current (`with s:` in CuPy, `torch.cuda.stream(s)` in PyTorch).
- An object with only the CUDA array interface is waited for on the stream it reports (version 3;
  `None` reports none and the call does not wait). A producer of version 2 or lower has no stream:
  the call then waits for the legacy default stream, or only for `stream` when it is given.
  Stream handle 0 is refused in the interface; 1 is the legacy default stream and 2 the
  per-thread default stream of the calling thread.
- `stream` is an integer handle (0 means the legacy default stream), an object with
  `__cuda_stream__`, a PyTorch stream or a CuPy stream. It must be a live stream of the device of
  `x`: CUDA does not validate handles, and an unknown one can crash the process.
- PyTorch exports a CUDA tensor only while the tensor's device is the current device (make it
  current with `torch.cuda.device(i)`) and refuses `stream=2`; the call then raises `TypeError`.

The call may be made from several threads; each thread keeps its own workspace.

## Memory

eda keeps the workspace of its calls in a memory pool of its own for later calls, until the
interpreter exits. A call frees it early when its result would not fit beside it. A batch that does
not fit at once is processed in chunks sized by the free device memory; because an element does not
depend on the others, the chunking does not change the result.

The memory of a dropped `DeviceArray` is freed by the next eda call on any thread, or at exit, not at
the moment it is dropped.

`MemoryError` (not `CudaError`) is raised when the output rows (with `return_orientation`, the
rows and their orientation) do not fit on the device, when one element's workspace does not fit,
or when a cuFFT plan cannot get its memory (the plan is made once
more after the memory pools give back what they hold). The message names the sizes. Before it
raises, the call gives back to the device what the calling thread's eda state does not use: its own
buffers, the workspace and cuFFT plans that thread keeps, and the free memory of eda's pools. The
workspaces and plans of other threads, the memory of calls still running there and the memory of
live `DeviceArray` results stay. The next call then works.

## Environment variables

There are two variables, both meant for the tests and the sanitizers; normal use needs neither.
They take `1/0`, `true/false`, `on/off` or `yes/no`. The library rejects any other `EDA_*`
variable and any other value with a `ValueError` at the first call, so that a misspelled switch is
not ignored.

| Variable | Default | Effect |
|---|---|---|
| `EDA_SANITIZE` | `0` | Sanitizer mode: every workspace slice is a device allocation of its own, allocated afresh at each use and filled with `0xFF` bytes (NaN as float); the output rows are zeroed before use; the tests shrink their shapes. Slower and uses more memory. |
| `EDA_ARENA_POISON` | `1` | In `EDA_SANITIZE` mode, fill new slices with `0xFF`. `0` turns the fill off, which `compute-sanitizer --tool initcheck` needs (it counts a fill as initialization). |

`tools/sanitize.sh` sets both as needed, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Determinism and batches

- **Deterministic.** The same input, arguments, GPU, cuFFT and build give the same bits, in the
  same process or another, from any thread, and after the plan cache and the memory pools have
  been through other shapes.
- **An element does not see the batch.** Its rows, windows, sifts, spacing and energy are the
  same bits alone, at any position of any batch, in any order of the batch, with any chunking,
  whatever the other elements hold, for every shape and dtype. The orientation step in 2-D and
  3-D uses complex-to-complex cuFFT transforms for this reason: cuFFT's real-to-complex plans
  can compute two entries of a batch together, and then an entry's rounding depends on its
  neighbour.
- **The orientation and the monogenic signal.** The `Orientation` of a call and every output of
  `eda.monogenic` are deterministic and independent of the batch in the same way
  (`tests/python/test_orientation_output.py`, `tests/python/test_monogenic.py`).
- **Across cuFFT versions, within rounding.** Bit for bit holds for a given GPU, build and cuFFT;
  another cuFFT version agrees within rounding. `tests/python/test_parity.py` checks the record
  `tests/data/parity_v2.json` bit for bit when the process maps the record's cuFFT file
  (`nvidia-cufft` 12.0.0.61) and otherwise within tolerance: decisions equal on every case, rows
  within 10 times the measured sensitivity on the stable cases. On an RTX 4090 all 287 cases
  were bit for bit equal with eda-cuda and cuFFT 12.0.0.61 and 12.4.0.43, and with eda-cuda-cu12
  and cuFFT 11.4.1.4.
- **Not promised.** Another GPU model, cuFFT version or build can differ in the last bits
  (decisions such as the number of IMFs can change when an input sits on a threshold).

`tests/python/test_properties.py` checks each of these, including batches of 70 elements (more than
one orientation group), every batch position of sizes whose cuFFT plans pair entries, chunk sizes
of 7, 64 and 65, a permuted batch, repeats after the plan cache has evicted the shape, and threads.

## Limits

- Spatial extent per axis: 2 to 8192 samples in float32, 2 to 4096 in float64. A pass of the
  method stages a whole line in shared memory. Larger axes raise `ValueError`. `eda.monogenic`
  takes the same limits.
- One to three spatial axes; more batch axes are free.
- Real float32 or float64 data (other real dtypes through `dtype=`).
- The device must have compute capability 8.0 or newer and a driver for CUDA 13.0 or newer
  (`eda-cuda`) or CUDA 12.3 or newer (`eda-cuda-cu12`); an older driver raises `eda.CudaError`
  with the version needed. Each wheel carries code for 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0, PTX for
  12.0, and needs Linux x86-64 with glibc 2.28 or newer.
- CUDA graphs are not used yet; every launch is issued from the host.
- Not invariant under a mirror of a single axis in 2-D and 3-D, nor, in any dimension, under adding
  a constant (below).
- The method is empirical. Nothing guarantees that a mode holds a single oscillation: oscillations
  of similar scale can share a mode, and one oscillation can be split over two (mode mixing).

## Properties of the method

All of these are checked in `tests/python/test_properties.py` and `test_golden.py`. `u` is the unit
roundoff of the compute dtype (2^-24 float32, 2^-53 float64), `P` the number of rows of an element
and `D` the number of spatial axes.

### Exact

- **Gain by a power of two.** For `x * 2**j` the rows are the rows of `x` times `2**j`, bit for
  bit; windows, sifts and spacing are equal and the energy is multiplied by `4**j`. This also
  holds with `tie_tolerance` 0 or `2**-10`, because it is relative. It needs every value to stay
  normal or zero at both scales.
- **Sign.** `-y(-x) == y(x)` in value (the bytes differ only in the sign of zeros).
- **One-dimensional mirror.** `x[::-1]` gives the mirrored rows bit for bit, with equal windows,
  sifts and spacing, for inputs whose extrema are not plateaus of even length (a plateau's peak
  sits at its middle sample, which a mirror moves for an even plateau, so exact ties can change
  the decisions). The energy agrees up to the rounding of a sum in another order.
- **`max_imf`.** For every `m` below the number of IMFs, `max_imf=m` gives the first `m` rows,
  windows, sifts, spacing and energy of the unlimited call bit for bit; the residue differs
  from the sum of the unlimited tail only by rounding.
- **No oscillation.** A constant, zero or subnormal-constant input has no IMF and its residue is
  the input bit for bit.
- **Batch, chunk, repeat, thread.** Bitwise, as above.

### Within a derived bound

- **Reconstruction.** Each mode is committed as `r_next = fl(r - h)`, one rounding per mode. Let
  `t_k` be the sum of rows `k` to `P - 1` (the IMFs from `k` on and the residue) and
  `S = sum_{k=1}^{P-1} |t_k|`. Then at every sample

  `|sum of the rows - x| <= u * S / (1 - P u) * (1 + 2**-20) + P * 2**-63 * sum_k |y_k|`

  (the last term allows for the long-double sum used to check it). The test asserts this at every
  sample for D = 1 to 3 and both dtypes. In other words, the error is at most a few units of
  roundoff times the size of the partial sums, times the number of modes.
- **Axis permutation and point reflection (all axes flipped), 2-D and 3-D.** The decisions (IMF
  count, windows, sifts) are equal and the rows differ by rounding only, because the box sums,
  the cuFFT transforms and the interpolation round in another order. The test ceilings are in
  units of `2u * max|x| * (P - 1)`: the largest values measured over the test family were 7.9 for
  permutations, 7.2 for point reflections (238 for 3-D float32) and 3.7 for an offset, and each
  ceiling sits at eight or more times that (four for 3-D float32). They are regression ceilings,
  not bounds of the method. `axes=(1, 0)` against a transposed copy is bitwise.
- **Golden corpus.** On the 9 float64 cases of `tests/data/golden` (made by the predecessor
  implementation) the decisions (IMF count, windows, sifts) are equal and the rows agree within
  10 times the case's measured sensitivity to input noise of relative size 1e-12; the worst
  difference measured is 1.2e-14 of `max|x|`. In float32 the decisions agree and the
  rows within 2e-3 of `max|x|` (printed, not asserted).

### Not symmetries

- **A mirror of a single axis in 2-D and 3-D.** The noise level pools the signed differences of
  all axes, so a one-axis mirror changes it. The results are close in practice, but not
  guaranteed to be.
- **A constant offset.** The tie tolerance and the sift guard are relative to `max|r|`, so
  `x + c` can take other windows and sifts. On the golden chirp with `c = 0.75 max|x|` the windows
  change from 19, 21, 51, 129, 513 to 19, 21, 43, 105, 341.
- **Mirrors of inputs with exact ties** (quantized data), see the 1-D mirror above.

The single-axis mirror and the offset follow from the method; changing either would change its
results.
