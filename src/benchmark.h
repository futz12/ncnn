// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_BENCHMARK_H
#define NCNN_BENCHMARK_H

#include "layer.h"
#include "mat.h"
#include "platform.h"

#include <string>
#include <vector>

namespace ncnn {

// get now timestamp in ms
NCNN_EXPORT double get_current_time();

// sleep milliseconds
NCNN_EXPORT void sleep(unsigned long long int milliseconds = 1000);

#if NCNN_BENCHMARK

struct LayerBenchStat
{
    std::string type;
    std::string name;
    int count = 0;
    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;
};

struct LayerTypeStat
{
    std::string type;
    int count = 0;
    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;
};

NCNN_EXPORT void reset_layer_benchmark();
NCNN_EXPORT void set_layer_benchmark_active(bool active);
NCNN_EXPORT bool is_layer_benchmark_active();
NCNN_EXPORT bool is_verbose_layer_benchmark();
NCNN_EXPORT void record_layer_benchmark(const std::string& type, const std::string& name, double duration_ms);
NCNN_EXPORT void print_layer_benchmark_summary(int top_k = 15);
NCNN_EXPORT std::vector<LayerTypeStat> get_layer_type_benchmark_stats();
NCNN_EXPORT std::vector<LayerBenchStat> get_layer_benchmark_stats();

NCNN_EXPORT void benchmark(const Layer* layer, double start, double end);
NCNN_EXPORT void benchmark(const Layer* layer, const Mat& bottom_blob, Mat& top_blob, double start, double end);

#endif // NCNN_BENCHMARK

} // namespace ncnn

#endif // NCNN_BENCHMARK_H
