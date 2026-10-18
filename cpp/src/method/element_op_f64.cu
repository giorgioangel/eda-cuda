// SPDX-License-Identifier: Apache-2.0
// The element operators and the envelope mean for double (element_op.cuh). One translation unit
// per dtype keeps the compile time of the 9 gather instantiations per dtype in parallel builds.
#include "method/element_op.cuh"

namespace eda {

template void enqueue_element_op<double>(cudaStream_t, const Geom&, ElemOp, const ElemView<double>&,
                                         const double*, double*, double*, double*, double*);
template void enqueue_envelope_mean<double>(cudaStream_t, const Geom&, const ElemView<double>&, const double*,
                                            double*, double*, double*, double*, double*);

}  // namespace eda
