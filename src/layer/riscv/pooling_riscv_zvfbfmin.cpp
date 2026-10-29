// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling_riscv.h"

#include <float.h>

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "pooling_bf16s.h"

void pooling_bf16s_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, const Option& opt, const Pooling* layer)
{
    pooling_bf16s(bottom_blob, top_blob, opt, layer);
}

#endif // NCNN_BF16

} // namespace ncnn
