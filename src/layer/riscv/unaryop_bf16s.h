// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void unaryop_bf16s_zvfbfmin(Mat& bottom_top_blob, int op_type, const Option& opt);
#endif

#if __riscv_vector
// bf16 data is loaded as e16m4 and widened to f32m8 so that the same functors
// (written for f32m8) can be reused by the fp32 and bf16 paths
static inline vfloat32m8_t unaryop_bfloat2float_rvv(vuint16m4_t v, size_t vl)
{
#if NCNN_ZVFBFMIN && __riscv_zvfbfmin
    return __riscv_vfwcvtbf16_f_f_v_f32m8(__riscv_vreinterpret_v_u16m4_bf16m4(v), vl);
#elif __riscv_xtheadvector
    return __riscv_vfwcvt_f_f_v_f32m8(__riscv_vreinterpret_v_u16m4_f16m4(v), vl);
#else
    return __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vsll_vx_u32m8(__riscv_vwcvtu_x_x_v_u32m8(v, vl), 16, vl));
#endif
}

static inline vuint16m4_t unaryop_float2bfloat_rvv(vfloat32m8_t v, size_t vl)
{
#if NCNN_ZVFBFMIN && __riscv_zvfbfmin
    return __riscv_vreinterpret_v_bf16m4_u16m4(__riscv_vfncvtbf16_f_f_w_bf16m4_rm(v, __RISCV_FRM_RNE, vl));
#elif __riscv_xtheadvector
    return __riscv_vreinterpret_v_f16m4_u16m4(__riscv_vfncvt_f_f_w_f16m4(v, vl));
#else
    return __riscv_vnclipu_wx_u16m4(__riscv_vreinterpret_v_f32m8_u32m8(v), 16, __RISCV_VXRM_RNE, vl);
#endif
}

template<typename Op>
static int unary_op_inplace_bf16s(Mat& a, const Option& opt)
{
    Op op;

    const int w = a.w;
    const int h = a.h;
    const int d = a.d;
    const int channels = a.c;
    const int size = w * h * d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat32m8_t _p = unaryop_bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            _p = op(_p, vl);
            __riscv_vse16_v_u16m4(ptr, unaryop_float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        // restore FXCR while retaining accrued exception flags, including FE
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
    }

    return 0;
}
#endif // __riscv_vector

static int unaryop_bf16s(Mat& bottom_top_blob, int op_type, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        unaryop_bf16s_zvfbfmin(bottom_top_blob, op_type, opt);
        return 0;
    }
#endif

#if __riscv_vector
    using namespace unaryop_functor;

    if (op_type == UnaryOp::Operation_ABS)
        return unary_op_inplace_bf16s<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_NEG)
        return unary_op_inplace_bf16s<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_FLOOR)
        return unary_op_inplace_bf16s<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_CEIL)
        return unary_op_inplace_bf16s<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SQUARE)
        return unary_op_inplace_bf16s<unary_op_square>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SQRT)
        return unary_op_inplace_bf16s<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_RSQRT)
        return unary_op_inplace_bf16s<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_EXP)
        return unary_op_inplace_bf16s<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_LOG)
        return unary_op_inplace_bf16s<unary_op_log>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SIN)
        return unary_op_inplace_bf16s<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_COS)
        return unary_op_inplace_bf16s<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_TAN)
        return unary_op_inplace_bf16s<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ASIN)
        return unary_op_inplace_bf16s<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ACOS)
        return unary_op_inplace_bf16s<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ATAN)
        return unary_op_inplace_bf16s<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_RECIPROCAL)
        return unary_op_inplace_bf16s<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_TANH)
        return unary_op_inplace_bf16s<unary_op_tanh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_LOG10)
        return unary_op_inplace_bf16s<unary_op_log10>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ROUND)
    {
        // round to nearest even
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        int ret = unary_op_inplace_bf16s<unary_op_round>(bottom_top_blob, opt);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return ret;
    }

    if (op_type == UnaryOp::Operation_TRUNC)
        return unary_op_inplace_bf16s<unary_op_trunc>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SIGN)
        return unary_op_inplace_bf16s<unary_op_sign>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_EXPM1)
        return unary_op_inplace_bf16s<unary_op_expm1>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SINH)
        return unary_op_inplace_bf16s<unary_op_sinh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ASINH)
        return unary_op_inplace_bf16s<unary_op_asinh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_COSH)
        return unary_op_inplace_bf16s<unary_op_cosh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ACOSH)
        return unary_op_inplace_bf16s<unary_op_acosh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ATANH)
        return unary_op_inplace_bf16s<unary_op_atanh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_LOG1P)
        return unary_op_inplace_bf16s<unary_op_log1p>(bottom_top_blob, opt);

    return 0;
#else  // __riscv_vector
    // no vector intrinsics available, convert to fp32 and reuse the generic implementation
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    Mat bottom_top_blob_fp32;
    if (dims == 1)
        bottom_top_blob_fp32.create(w, 4u, elempack, opt.workspace_allocator);
    if (dims == 2)
        bottom_top_blob_fp32.create(w, h, 4u, elempack, opt.workspace_allocator);
    if (dims == 3)
        bottom_top_blob_fp32.create(w, h, channels, 4u, elempack, opt.workspace_allocator);
    if (dims == 4)
        bottom_top_blob_fp32.create(w, h, d, channels, 4u, elempack, opt.workspace_allocator);
    if (bottom_top_blob_fp32.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_top_blob.channel(q);
        float* outptr = bottom_top_blob_fp32.channel(q);

        for (int i = 0; i < size; i++)
        {
            outptr[i] = bfloat16_to_float32(ptr[i]);
        }
    }

    UnaryOp unaryop;
    unaryop.op_type = op_type;
    int ret = unaryop.forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob_fp32.channel(q);
        unsigned short* outptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            outptr[i] = float32_to_bfloat16(ptr[i]);
        }
    }

    return 0;
#endif // __riscv_vector
}
