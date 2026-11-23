// Copyright 2026 <pchar.cn> futz12
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
    // use the generic kernel until the remaining axis layouts have ARM kernels that are not slower
    if (bottom_blob.dims == 4 && (operation == ReductionOp_SUMSQ || operation == ReductionOp_L2))
        return Reduction::forward(bottom_blob, top_blob, opt);

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

    if (operation == ReductionOp_MAX && outdims == 2 && bottom_blob.dims == 3 && reduce_c && !reduce_w && !reduce_h && (bottom_blob.w * bottom_blob.h) % 4 == 0)
    {
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int channels = bottom_blob.c;
        const int n = w * h;
        float* outptr = top_blob;

#if __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < n; i += 4)
        {
            float32x4_t _max = vdupq_n_f32(-FLT_MAX);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                _max = vmaxq_f32(_max, vld1q_f32(ptr + i));
            }

            if (i + 3 < n)
            {
                vst1q_f32(outptr + i, _max);
            }
            else
            {
                for (int j = 0; i + j < n; j++)
                    outptr[i + j] = vgetq_lane_f32(_max, j);
            }
        }
#else
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < n; i++)
        {
            float max_value = -FLT_MAX;
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                max_value = ptr[i] > max_value ? ptr[i] : max_value;
            }
            outptr[i] = max_value;
        }
#endif // __ARM_NEON

        return 0;
    }

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
