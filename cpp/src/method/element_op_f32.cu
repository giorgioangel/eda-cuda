// SPDX-License-Identifier: Apache-2.0
// The element operators and the envelope mean for float (element_op.cuh). One translation unit
// per dtype keeps the compile time of the 9 gather instantiations per dtype in parallel builds.
#include "method/element_op.cuh"

namespace eda {

template void enqueue_element_op<float>(cudaStream_t, const Geom&, ElemOp, const ElemView<float>&,
                                        const float*, float*, float*, float*, float*);
template void enqueue_envelope_mean<float>(cudaStream_t, const Geom&, const ElemView<float>&, const float*,
                                           float*, float*, float*, float*, float*);

}  // namespace eda
