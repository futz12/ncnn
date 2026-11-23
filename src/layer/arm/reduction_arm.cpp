// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_arm.h"

#include "cpu.h"

#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#include "reduction_arm_fp32.h"

int Reduction_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    bool reduce_w, reduce_h, reduce_d, reduce_c;
    int outdims, outw, outh, outd, outc;
    resolve_reduce_flags_and_output_shape(bottom_blob, reduce_w, reduce_h, reduce_d, reduce_c, outdims, outw, outh, outd, outc);

    if (outdims == 0)
        top_blob.create(1, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 1)
        top_blob.create(outw, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 2)
        top_blob.create(outw, outh, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 3)
        top_blob.create(outw, outh, outc, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 4)
        top_blob.create(outw, outh, outd, outc, bottom_blob.elemsize, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    int ret = reduction_fp32(bottom_blob, top_blob, reduce_w, reduce_h, reduce_d, reduce_c, operation, opt);
    if (ret != 0)
        return ret;

    float coeff = this->coeff;
    if (operation == ReductionOp_MEAN)
    {
        int scale = 1;
        if (reduce_w) scale *= bottom_blob.w;
        if (reduce_h) scale *= bottom_blob.h;
        if (reduce_d) scale *= bottom_blob.d;
        if (reduce_c) scale *= bottom_blob.c;
        coeff /= scale;
    }

    const int size = top_blob.w * top_blob.h * top_blob.d;
    if (operation == ReductionOp_LogSum)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            float* ptr = top_blob.channel(q);
            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                vst1q_f32(ptr, log_ps(_p));
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                ptr[0] = logf(ptr[0]);
                ptr++;
            }
        }
    }

    if (operation == ReductionOp_L2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            float* ptr = top_blob.channel(q);
            int i = 0;
#if __ARM_NEON
            const float32x4_t _zero = vdupq_n_f32(0.f);
            const float32x4_t _min = vdupq_n_f32(FLT_MIN);
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vbslq_f32(vcgtq_f32(_p, _min), vsqrtq_f32(_p), _zero);
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                ptr[0] = sqrtf(ptr[0] < FLT_MIN ? 0.f : ptr[0]);
                ptr++;
            }
        }
    }

    if (coeff != 1.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            float* ptr = top_blob.channel(q);
            int i = 0;
#if __ARM_NEON
            const float32x4_t _coeff = vdupq_n_f32(coeff);
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmulq_f32(_p, _coeff);
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                ptr[0] *= coeff;
                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
