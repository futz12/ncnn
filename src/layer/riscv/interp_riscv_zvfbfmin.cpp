// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "interp_riscv.h"

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if NCNN_BF16

#include "interp_bf16s.h"

void interp_bf16s_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, int outw, int outh, const Option& opt, const Interp* layer)
{
    interp_bf16s_core(bottom_blob, top_blob, outw, outh, opt, layer);
}

#endif // NCNN_BF16

} // namespace ncnn
