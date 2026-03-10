// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_mips.h"

namespace ncnn {

ELU_mips::ELU_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

int ELU_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = alpha * (expf(ptr[i]) - 1.f);
        }
    }

    return 0;
}

} // namespace ncnn
