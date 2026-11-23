// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REDUCTION_ARM_H
#define LAYER_REDUCTION_ARM_H

#include "reduction.h"

namespace ncnn {

class Reduction_arm : public Reduction
{
public:
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_ARM_H
