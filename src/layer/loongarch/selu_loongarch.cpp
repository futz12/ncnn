// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_loongarch.h"

namespace ncnn {

SELU_loongarch::SELU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
}

int SELU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;
    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = (expf(ptr[i]) - 1.f) * alphaxlambda;
            else
                ptr[i] *= lambda;
        }
    }

    return 0;
}

} // namespace ncnn
