// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

namespace reduction_arm_functor {

#include "reduction_arm_functor.h"

} // namespace reduction_arm_functor

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"

static NCNN_FORCEINLINE float reduction_horizontal_sum(float32x4_t v)
{
#if __aarch64__
    return vaddvq_f32(v);
#else
    float32x2_t v2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    v2 = vpadd_f32(v2, v2);
    return vget_lane_f32(v2, 0);
#endif
}
#endif // __ARM_NEON

template<typename Op, typename Op2>
static NCNN_FORCEINLINE float reduction(float v0, const float* ptr, int size)
{
    const Op op;
    const Op2 op2;
    float sum = v0;

    int i = 0;
#if __ARM_NEON
    float32x4_t _sum4 = vdupq_n_f32(v0);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        _sum4 = op.func_pack4(_sum4, _p);
        ptr += 4;
    }
    sum = op2.func(sum, op2.reduce_pack4(_sum4));
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        sum = op.func(sum, *ptr);
        ptr++;
    }

    return sum;
}

template<typename Op>
static NCNN_FORCEINLINE void reduction_vector(const float* ptr, float* outptr, int size)
{
    const Op op;

    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = vld1q_f32(outptr);
        _outp = op.func_pack4(_outp, _p);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = op.func(*outptr, *ptr);
        ptr++;
        outptr++;
    }
}

static void reduction_collapse_axes(int& w, int& h, int& d, bool& reduce_w, bool& reduce_h, bool& reduce_d)
{
    if (reduce_w == reduce_h)
    {
        w *= h;
        h = d;
        reduce_h = reduce_d;
        d = 1;
        reduce_d = false;
    }
    if (reduce_h == reduce_d)
    {
        h *= d;
        d = 1;
        reduce_d = false;
    }
    if (reduce_w == reduce_h)
    {
        w *= h;
        h = 1;
        reduce_h = false;
    }
}

template<typename Op, typename Op2>
static int reduction_op(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, float v0, const Option& opt)
{
    const Op op;
    const Op2 op2;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    const int channels = a.c;

    // removing c from a 4d tensor makes d the output channel axis
    // keep these planes separate because the output may have channel padding
    if (a.dims == 4 && reduce_c && !reduce_w && !reduce_h && !reduce_d && b.dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int z = 0; z < d; z++)
        {
            Mat out = b.channel(z);
            out.fill(v0);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q).depth(z);
                reduction_vector<Op>(ptr, out, w * h);
            }
        }
        return 0;
    }

    reduction_collapse_axes(w, h, d, reduce_w, reduce_h, reduce_d);

    if (reduce_w && h == 1 && d == 1)
    {
        Mat sums = b;
        if (reduce_c && channels > 1)
        {
            sums.create(channels, a.elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            float* outptr = sums.dims >= 3 ? sums.channel(q) : (float*)sums + q;
            outptr[0] = reduction<Op, Op2>(v0, ptr, w);
        }

        if (reduce_c && channels > 1)
            b[0] = reduction<Op2, Op2>(v0, sums, channels);
        return 0;
    }

    // each contiguous row produces one output, with no partial sums to merge
    if (reduce_w && !reduce_h && !reduce_d && !reduce_c)
    {
        const int rows = h * d;
        const size_t out_cstep = b.dims >= 3 ? b.cstep : rows;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < channels * rows; i++)
        {
            const int q = i / rows;
            const int y = i % rows;
            const float* ptr = (const float*)a + q * a.cstep + y * w;
            float* outptr = (float*)b + q * out_cstep;
            outptr[y] = reduction<Op, Op2>(v0, ptr, w);
        }
        return 0;
    }

    const int outw = reduce_w ? 1 : w;
    const int outh = reduce_h ? 1 : h;
    const int outd = reduce_d ? 1 : d;
    const int outc = reduce_c ? 1 : channels;
    const int size = b.w * b.h * b.d;

    const int reduced_h = reduce_h ? h : 1;
    const int reduced_d = reduce_d ? d : 1;
    const int reduced_c = reduce_c ? channels : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < outc * outd * outh; i++)
    {
        const int q = i / (outd * outh);
        const int z = i / outh % outd;
        const int y = i % outh;
        const size_t offset = (size_t)i * outw;
        float* outptr = (float*)b + offset / size * b.cstep + offset % size;
        const float* ptr = (const float*)a + q * a.cstep + (z * h + y) * w;

        if (reduce_w)
        {
            float sum = v0;
#if __ARM_NEON
            int jb = 0;
            for (; jb + 15 < w; jb += 16)
            {
                float32x4_t _acc0 = vdupq_n_f32(v0);
                float32x4_t _acc1 = vdupq_n_f32(v0);
                float32x4_t _acc2 = vdupq_n_f32(v0);
                float32x4_t _acc3 = vdupq_n_f32(v0);
                for (int qc = 0; qc < reduced_c; qc++)
                {
                    for (int zd = 0; zd < reduced_d; zd++)
                    {
                        const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                        for (int yh = 0; yh < reduced_h; yh++)
                        {
                            _acc0 = op.func_pack4(_acc0, vld1q_f32(ptr0 + jb));
                            _acc1 = op.func_pack4(_acc1, vld1q_f32(ptr0 + jb + 4));
                            _acc2 = op.func_pack4(_acc2, vld1q_f32(ptr0 + jb + 8));
                            _acc3 = op.func_pack4(_acc3, vld1q_f32(ptr0 + jb + 12));
                            ptr0 += w;
                        }
                    }
                }
                sum = op2.func(sum, op2.reduce_pack4(_acc0));
                sum = op2.func(sum, op2.reduce_pack4(_acc1));
                sum = op2.func(sum, op2.reduce_pack4(_acc2));
                sum = op2.func(sum, op2.reduce_pack4(_acc3));
            }
            for (; jb + 3 < w; jb += 4)
            {
                float32x4_t _acc = vdupq_n_f32(v0);
                for (int qc = 0; qc < reduced_c; qc++)
                {
                    for (int zd = 0; zd < reduced_d; zd++)
                    {
                        const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                        for (int yh = 0; yh < reduced_h; yh++)
                        {
                            _acc = op.func_pack4(_acc, vld1q_f32(ptr0 + jb));
                            ptr0 += w;
                        }
                    }
                }
                sum = op2.func(sum, op2.reduce_pack4(_acc));
            }
            for (; jb < w; jb++)
            {
                for (int qc = 0; qc < reduced_c; qc++)
                {
                    for (int zd = 0; zd < reduced_d; zd++)
                    {
                        const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                        for (int yh = 0; yh < reduced_h; yh++)
                        {
                            sum = op.func(sum, ptr0[jb]);
                            ptr0 += w;
                        }
                    }
                }
            }
#else
            for (int qc = 0; qc < reduced_c; qc++)
            {
                for (int zd = 0; zd < reduced_d; zd++)
                {
                    const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                    for (int yh = 0; yh < reduced_h; yh++)
                    {
                        float v = reduction<Op, Op2>(v0, ptr0, w);
                        sum = op2.func(sum, v);
                        ptr0 += w;
                    }
                }
            }
#endif // __ARM_NEON
            outptr[0] = sum;
        }
        else
        {
#if __ARM_NEON
            int j = 0;
            for (; j + 15 < outw; j += 16)
            {
                float32x4_t _acc0 = vdupq_n_f32(v0);
                float32x4_t _acc1 = vdupq_n_f32(v0);
                float32x4_t _acc2 = vdupq_n_f32(v0);
                float32x4_t _acc3 = vdupq_n_f32(v0);
                for (int qc = 0; qc < reduced_c; qc++)
                {
                    for (int zd = 0; zd < reduced_d; zd++)
                    {
                        const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                        for (int yh = 0; yh < reduced_h; yh++)
                        {
                            _acc0 = op.func_pack4(_acc0, vld1q_f32(ptr0 + j));
                            _acc1 = op.func_pack4(_acc1, vld1q_f32(ptr0 + j + 4));
                            _acc2 = op.func_pack4(_acc2, vld1q_f32(ptr0 + j + 8));
                            _acc3 = op.func_pack4(_acc3, vld1q_f32(ptr0 + j + 12));
                            ptr0 += w;
                        }
                    }
                }
                vst1q_f32(outptr + j, _acc0);
                vst1q_f32(outptr + j + 4, _acc1);
                vst1q_f32(outptr + j + 8, _acc2);
                vst1q_f32(outptr + j + 12, _acc3);
            }
            for (; j + 3 < outw; j += 4)
            {
                float32x4_t _acc = vdupq_n_f32(v0);
                for (int qc = 0; qc < reduced_c; qc++)
                {
                    for (int zd = 0; zd < reduced_d; zd++)
                    {
                        const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                        for (int yh = 0; yh < reduced_h; yh++)
                        {
                            _acc = op.func_pack4(_acc, vld1q_f32(ptr0 + j));
                            ptr0 += w;
                        }
                    }
                }
                vst1q_f32(outptr + j, _acc);
            }
            for (; j < outw; j++)
            {
                float sum = v0;
                for (int qc = 0; qc < reduced_c; qc++)
                {
                    for (int zd = 0; zd < reduced_d; zd++)
                    {
                        const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                        for (int yh = 0; yh < reduced_h; yh++)
                        {
                            sum = op.func(sum, ptr0[j]);
                            ptr0 += w;
                        }
                    }
                }
                outptr[j] = sum;
            }
#else
            for (int j = 0; j < outw; j++)
                outptr[j] = v0;
            for (int qc = 0; qc < reduced_c; qc++)
            {
                for (int zd = 0; zd < reduced_d; zd++)
                {
                    const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                    for (int yh = 0; yh < reduced_h; yh++)
                    {
                        reduction_vector<Op>(ptr0, outptr, outw);
                        ptr0 += w;
                    }
                }
            }
#endif // __ARM_NEON
        }
    }

    return 0;
}

static float reduction_sumexp(const float* ptr, int size, float max_value)
{
    float sum = 0.f;

    int i = 0;
#if __ARM_NEON
    float32x4_t _max4 = vdupq_n_f32(max_value);
    float32x4_t _sum4 = vdupq_n_f32(0.f);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        _sum4 = vaddq_f32(_sum4, exp_ps(vsubq_f32(_p, _max4)));
        ptr += 4;
    }
    sum = reduction_horizontal_sum(_sum4);
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        sum += expf(*ptr - max_value);
        ptr++;
    }

    return sum;
}

static void reduction_vector_sumexp(const float* ptr, const float* maxptr, float* outptr, int size)
{
    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _max = vld1q_f32(maxptr);
        float32x4_t _outp = vld1q_f32(outptr);
        _outp = vaddq_f32(_outp, exp_ps(vsubq_f32(_p, _max)));
        vst1q_f32(outptr, _outp);
        ptr += 4;
        maxptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr += expf(*ptr - *maxptr);
        ptr++;
        maxptr++;
        outptr++;
    }
}

static void reduction_vector_addlog(const float* ptr, float* outptr, int size)
{
    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = vld1q_f32(outptr);
        _outp = vaddq_f32(_outp, log_ps(_p));
        vst1q_f32(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr += logf(*ptr);
        ptr++;
        outptr++;
    }
}

static int reduction_logsumexp(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, const Option& opt)
{
    using namespace reduction_arm_functor;

    // subtracting the max keeps every exp argument non-positive
    int ret = reduction_op<reduction_op_max, reduction_op_max>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, -FLT_MAX, opt);
    if (ret != 0)
        return ret;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    const int channels = a.c;

    // the max is read back from b, the exp sums go to a buffer of the same shape
    Mat sums;

    // removing c from a 4d tensor makes d the output channel axis
    // keep these planes separate because the output may have channel padding
    if (a.dims == 4 && reduce_c && !reduce_w && !reduce_h && !reduce_d && b.dims == 3)
    {
        sums.create_like(b, opt.workspace_allocator);
        if (sums.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int z = 0; z < d; z++)
        {
            Mat out = b.channel(z);
            Mat sum = sums.channel(z);
            sum.fill(0.f);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q).depth(z);
                reduction_vector_sumexp(ptr, out, sum, w * h);
            }
            reduction_vector_addlog(sum, out, w * h);
        }
        return 0;
    }

    reduction_collapse_axes(w, h, d, reduce_w, reduce_h, reduce_d);

    if (reduce_w && h == 1 && d == 1 && reduce_c && channels > 1)
    {
        // every channel shares the global max, so the channel sums add up directly
        sums.create(channels, a.elemsize, opt.workspace_allocator);
        if (sums.empty())
            return -100;

        const float max_value = b[0];

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            sums[q] = reduction_sumexp(ptr, w, max_value);
        }

        b[0] = max_value + logf(reduction<reduction_op_add, reduction_op_add>(0.f, sums, channels));
        return 0;
    }

    const int outw = reduce_w ? 1 : w;
    const int outh = reduce_h ? 1 : h;
    const int outd = reduce_d ? 1 : d;
    const int outc = reduce_c ? 1 : channels;
    const int size = b.w * b.h * b.d;

    const int reduced_h = reduce_h ? h : 1;
    const int reduced_d = reduce_d ? d : 1;
    const int reduced_c = reduce_c ? channels : 1;

    if (!reduce_w)
    {
        sums.create_like(b, opt.workspace_allocator);
        if (sums.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < outc * outd * outh; i++)
    {
        const int q = i / (outd * outh);
        const int z = i / outh % outd;
        const int y = i % outh;
        const size_t offset = (size_t)i * outw;
        float* outptr = (float*)b + offset / size * b.cstep + offset % size;
        const float* ptr = (const float*)a + q * a.cstep + (z * h + y) * w;

        if (reduce_w)
        {
            const float max_value = outptr[0];
            float sum = 0.f;
            for (int qc = 0; qc < reduced_c; qc++)
            {
                for (int zd = 0; zd < reduced_d; zd++)
                {
                    const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                    for (int yh = 0; yh < reduced_h; yh++)
                    {
                        sum += reduction_sumexp(ptr0, w, max_value);
                        ptr0 += w;
                    }
                }
            }
            outptr[0] = max_value + logf(sum);
        }
        else
        {
            float* sumptr = (float*)sums + offset / size * sums.cstep + offset % size;
            for (int j = 0; j < outw; j++)
                sumptr[j] = 0.f;
            for (int qc = 0; qc < reduced_c; qc++)
            {
                for (int zd = 0; zd < reduced_d; zd++)
                {
                    const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                    for (int yh = 0; yh < reduced_h; yh++)
                    {
                        reduction_vector_sumexp(ptr0, outptr, sumptr, outw);
                        ptr0 += w;
                    }
                }
            }
            reduction_vector_addlog(sumptr, outptr, outw);
        }
    }

    return 0;
}

static int reduction_fp32(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, int operation, const Option& opt)
{
    using namespace reduction_arm_functor;

    if (operation == Reduction::ReductionOp_SUM || operation == Reduction::ReductionOp_MEAN || operation == Reduction::ReductionOp_LogSum)
        return reduction_op<reduction_op_add, reduction_op_add>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 0.f, opt);

    if (operation == Reduction::ReductionOp_ASUM || operation == Reduction::ReductionOp_L1)
        return reduction_op<reduction_op_asum, reduction_op_add>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 0.f, opt);

    if (operation == Reduction::ReductionOp_SUMSQ || operation == Reduction::ReductionOp_L2)
        return reduction_op<reduction_op_sumsq, reduction_op_add>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 0.f, opt);

    if (operation == Reduction::ReductionOp_MAX)
        return reduction_op<reduction_op_max, reduction_op_max>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, -FLT_MAX, opt);

    if (operation == Reduction::ReductionOp_MIN)
        return reduction_op<reduction_op_min, reduction_op_min>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, FLT_MAX, opt);

    if (operation == Reduction::ReductionOp_PROD)
        return reduction_op<reduction_op_mul, reduction_op_mul>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 1.f, opt);

    if (operation == Reduction::ReductionOp_LogSumExp)
        return reduction_logsumexp(a, b, reduce_w, reduce_h, reduce_d, reduce_c, opt);

    return -1;
}
