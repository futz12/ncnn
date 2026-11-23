// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

struct reduction_op_add
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x + y;
    }
#if __ARM_NEON
    NCNN_FORCEINLINE float32x4_t func_pack4(const float32x4_t& x, const float32x4_t& y) const
    {
        return vaddq_f32(x, y);
    }
    NCNN_FORCEINLINE float reduce_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vaddvq_f32(x);
#else
        float32x2_t _x = vadd_f32(vget_low_f32(x), vget_high_f32(x));
        _x = vpadd_f32(_x, _x);
        return vget_lane_f32(_x, 0);
#endif
    }
#endif // __ARM_NEON
};

struct reduction_op_mul
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x * y;
    }
#if __ARM_NEON
    NCNN_FORCEINLINE float32x4_t func_pack4(const float32x4_t& x, const float32x4_t& y) const
    {
        return vmulq_f32(x, y);
    }
    NCNN_FORCEINLINE float reduce_pack4(const float32x4_t& x) const
    {
        float v = vgetq_lane_f32(x, 0);
        v *= vgetq_lane_f32(x, 1);
        v *= vgetq_lane_f32(x, 2);
        v *= vgetq_lane_f32(x, 3);
        return v;
    }
#endif // __ARM_NEON
};

struct reduction_op_max
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x > y ? x : y;
    }
#if __ARM_NEON
    NCNN_FORCEINLINE float32x4_t func_pack4(const float32x4_t& x, const float32x4_t& y) const
    {
        return vmaxq_f32(x, y);
    }
    NCNN_FORCEINLINE float reduce_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vmaxvq_f32(x);
#else
        float32x2_t _x = vpmax_f32(vget_low_f32(x), vget_high_f32(x));
        _x = vpmax_f32(_x, _x);
        return vget_lane_f32(_x, 0);
#endif
    }
#endif // __ARM_NEON
};

struct reduction_op_min
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x < y ? x : y;
    }
#if __ARM_NEON
    NCNN_FORCEINLINE float32x4_t func_pack4(const float32x4_t& x, const float32x4_t& y) const
    {
        return vminq_f32(x, y);
    }
    NCNN_FORCEINLINE float reduce_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vminvq_f32(x);
#else
        float32x2_t _x = vpmin_f32(vget_low_f32(x), vget_high_f32(x));
        _x = vpmin_f32(_x, _x);
        return vget_lane_f32(_x, 0);
#endif
    }
#endif // __ARM_NEON
};

struct reduction_op_asum
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x + fabsf(y);
    }
#if __ARM_NEON
    NCNN_FORCEINLINE float32x4_t func_pack4(const float32x4_t& x, const float32x4_t& y) const
    {
        return vaddq_f32(x, vabsq_f32(y));
    }
    NCNN_FORCEINLINE float reduce_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vaddvq_f32(x);
#else
        float32x2_t _x = vadd_f32(vget_low_f32(x), vget_high_f32(x));
        _x = vpadd_f32(_x, _x);
        return vget_lane_f32(_x, 0);
#endif
    }
#endif // __ARM_NEON
};

struct reduction_op_sumsq
{
    NCNN_FORCEINLINE float func(const float& x, const float& y) const
    {
        return x + y * y;
    }
#if __ARM_NEON
    NCNN_FORCEINLINE float32x4_t func_pack4(const float32x4_t& x, const float32x4_t& y) const
    {
        return vmlaq_f32(x, y, y);
    }
    NCNN_FORCEINLINE float reduce_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vaddvq_f32(x);
#else
        float32x2_t _x = vadd_f32(vget_low_f32(x), vget_high_f32(x));
        _x = vpadd_f32(_x, _x);
        return vget_lane_f32(_x, 0);
#endif
    }
#endif // __ARM_NEON
};
