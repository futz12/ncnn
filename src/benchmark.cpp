// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "benchmark.h"

#if (__cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)) && !defined(__riscv) && !NCNN_SIMPLESTL
#define USE_CXX11_CLOCK 1
#else
#define USE_CXX11_CLOCK 0
#endif

#if USE_CXX11_CLOCK
#include <chrono>
#if NCNN_THREADS
#include <thread>
#endif
#include <numeric>
#include <algorithm>
#endif // USE_CXX11_CLOCK

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else                 // _WIN32
#include <sys/time.h> //gettimeofday()
#include <unistd.h>   // sleep()
#endif                // _WIN32

#if NCNN_BENCHMARK
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/deconvolution.h"
#include "layer/deconvolutiondepthwise.h"
#include "layer/convolution3d.h"
#include "layer/convolutiondepthwise3d.h"
#include "layer/deconvolution3d.h"
#include "layer/deconvolutiondepthwise3d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <map>
#include <vector>
#include <algorithm>

namespace ncnn {

static std::mutex g_layer_bench_mutex;
static bool g_layer_bench_active = false;
static std::map<std::string, LayerBenchStat> g_layer_stats;
static std::map<std::string, LayerTypeStat> g_layer_type_stats;

bool is_verbose_layer_benchmark()
{
    static int verbose = -1;
    if (verbose == -1)
    {
        const char* env = getenv("NCNN_BENCHMARK_VERBOSE");
        verbose = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    return verbose == 1;
}

void reset_layer_benchmark()
{
    std::lock_guard<std::mutex> lock(g_layer_bench_mutex);
    g_layer_stats.clear();
    g_layer_type_stats.clear();
}

void set_layer_benchmark_active(bool active)
{
    std::lock_guard<std::mutex> lock(g_layer_bench_mutex);
    g_layer_bench_active = active;
}

bool is_layer_benchmark_active()
{
    return g_layer_bench_active;
}

void record_layer_benchmark(const std::string& type, const std::string& name, double duration_ms)
{
    std::lock_guard<std::mutex> lock(g_layer_bench_mutex);
    auto& ls = g_layer_stats[name];
    if (ls.count == 0)
    {
        ls.type = type;
        ls.name = name;
        ls.min_ms = duration_ms;
        ls.max_ms = duration_ms;
    }
    else
    {
        if (duration_ms < ls.min_ms) ls.min_ms = duration_ms;
        if (duration_ms > ls.max_ms) ls.max_ms = duration_ms;
    }
    ls.count++;
    ls.total_ms += duration_ms;

    auto& ts = g_layer_type_stats[type];
    if (ts.count == 0)
    {
        ts.type = type;
        ts.min_ms = duration_ms;
        ts.max_ms = duration_ms;
    }
    else
    {
        if (duration_ms < ts.min_ms) ts.min_ms = duration_ms;
        if (duration_ms > ts.max_ms) ts.max_ms = duration_ms;
    }
    ts.count++;
    ts.total_ms += duration_ms;
}

std::vector<LayerTypeStat> get_layer_type_benchmark_stats()
{
    std::lock_guard<std::mutex> lock(g_layer_bench_mutex);
    std::vector<LayerTypeStat> res;
    for (const auto& kv : g_layer_type_stats)
    {
        res.push_back(kv.second);
    }
    std::sort(res.begin(), res.end(), [](const LayerTypeStat& a, const LayerTypeStat& b) {
        return a.total_ms > b.total_ms;
    });
    return res;
}

std::vector<LayerBenchStat> get_layer_benchmark_stats()
{
    std::lock_guard<std::mutex> lock(g_layer_bench_mutex);
    std::vector<LayerBenchStat> res;
    for (const auto& kv : g_layer_stats)
    {
        res.push_back(kv.second);
    }
    std::sort(res.begin(), res.end(), [](const LayerBenchStat& a, const LayerBenchStat& b) {
        return a.total_ms > b.total_ms;
    });
    return res;
}

void print_layer_benchmark_summary(int top_k)
{
    auto type_stats = get_layer_type_benchmark_stats();
    auto layer_stats = get_layer_benchmark_stats();
    if (type_stats.empty())
    {
        fprintf(stderr, "[Layer Benchmark] No layer profiling data recorded.\n");
        return;
    }

    double total_layer_time = 0.0;
    for (const auto& ts : type_stats) total_layer_time += ts.total_ms;

    fprintf(stderr, "--------------------------------------------------------------------------------\n");
    fprintf(stderr, "                     DECODER LAYER BREAKDOWN BY OP TYPE                        \n");
    fprintf(stderr, "--------------------------------------------------------------------------------\n");
    fprintf(stderr, "%-22s | %8s | %12s | %12s | %8s\n", "Layer Type", "Calls", "Total (ms)", "Avg (ms)", "Time %");
    fprintf(stderr, "-----------------------+----------+--------------+--------------+---------\n");
    for (const auto& ts : type_stats)
    {
        double pct = total_layer_time > 0 ? (ts.total_ms / total_layer_time * 100.0) : 0.0;
        double avg = ts.count > 0 ? (ts.total_ms / ts.count) : 0.0;
        fprintf(stderr, "%-22s | %8d | %12.2f | %12.3f | %6.1f%%\n",
                ts.type.c_str(), ts.count, ts.total_ms, avg, pct);
    }
    fprintf(stderr, "-----------------------+----------+--------------+--------------+---------\n");
    fprintf(stderr, "%-22s | %8s | %12.2f | %12s | 100.0%%\n", "TOTAL", "-", total_layer_time, "-");
    fprintf(stderr, "--------------------------------------------------------------------------------\n");

    if (top_k > 0 && !layer_stats.empty())
    {
        int show_cnt = std::min((int)layer_stats.size(), top_k);
        fprintf(stderr, " TOP %d SLOWEST INDIVIDUAL LAYERS:\n", show_cnt);
        fprintf(stderr, " %-4s | %-28s | %-16s | %10s | %10s | %6s\n",
                "#", "Layer Name", "Type", "Total (ms)", "Avg (ms)", "Calls");
        fprintf(stderr, "------+------------------------------+------------------+------------+------------+-------\n");
        for (int i = 0; i < show_cnt; i++)
        {
            const auto& ls = layer_stats[i];
            double avg = ls.count > 0 ? (ls.total_ms / ls.count) : 0.0;
            fprintf(stderr, " %-4d | %-28s | %-16s | %10.2f | %10.3f | %6d\n",
                    i + 1, ls.name.c_str(), ls.type.c_str(), ls.total_ms, avg, ls.count);
        }
        fprintf(stderr, "--------------------------------------------------------------------------------\n");
    }
}

} // namespace ncnn

#endif // NCNN_BENCHMARK

namespace ncnn {

double get_current_time()
{
#if USE_CXX11_CLOCK
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
#else
#ifdef _WIN32
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
#else  // _WIN32
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif // _WIN32
#endif
}

void sleep(unsigned long long int milliseconds)
{
#if USE_CXX11_CLOCK && NCNN_THREADS
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
#else
#ifdef _WIN32
    Sleep(milliseconds);
#elif defined(__unix__) || defined(__APPLE__)
    usleep(milliseconds * 1000);
#elif _POSIX_TIMERS
    struct timespec ts;
    ts.tv_sec = milliseconds * 0.001;
    ts.tv_nsec = 0;
    nanosleep(&ts, &ts);
#else
    // TODO How to handle it ?
#endif
#endif
}

#if NCNN_BENCHMARK

void benchmark(const Layer* layer, double start, double end)
{
    if (is_layer_benchmark_active())
    {
        record_layer_benchmark(layer->type, layer->name, end - start);
    }

    if (is_verbose_layer_benchmark())
    {
        fprintf(stderr, "%-24s %-30s %8.2lfms", layer->type.c_str(), layer->name.c_str(), end - start);
        fprintf(stderr, "    |");
        fprintf(stderr, "\n");
    }
}

void benchmark(const Layer* layer, const Mat& bottom_blob, Mat& top_blob, double start, double end)
{
    if (is_layer_benchmark_active())
    {
        record_layer_benchmark(layer->type, layer->name, end - start);
    }

    if (is_verbose_layer_benchmark())
    {
        fprintf(stderr, "%-24s %-30s %8.2lfms", layer->type.c_str(), layer->name.c_str(), end - start);

        char in_shape_str[64] = {'\0'};
        char out_shape_str[64] = {'\0'};

        if (bottom_blob.dims == 1)
        {
            sprintf(in_shape_str, "[%3d *%d]", bottom_blob.w, bottom_blob.elempack);
        }
        if (bottom_blob.dims == 2)
        {
            sprintf(in_shape_str, "[%3d, %3d *%d]", bottom_blob.w, bottom_blob.h, bottom_blob.elempack);
        }
        if (bottom_blob.dims == 3)
        {
            sprintf(in_shape_str, "[%3d, %3d, %3d *%d]", bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elempack);
        }
        if (bottom_blob.dims == 4)
        {
            sprintf(in_shape_str, "[%3d, %3d, %3d, %3d *%d]", bottom_blob.w, bottom_blob.h, bottom_blob.d, bottom_blob.c, bottom_blob.elempack);
        }

        if (top_blob.dims == 1)
        {
            sprintf(out_shape_str, "[%3d *%d]", top_blob.w, top_blob.elempack);
        }
        if (top_blob.dims == 2)
        {
            sprintf(out_shape_str, "[%3d, %3d *%d]", top_blob.w, top_blob.h, top_blob.elempack);
        }
        if (top_blob.dims == 3)
        {
            sprintf(out_shape_str, "[%3d, %3d, %3d *%d]", top_blob.w, top_blob.h, top_blob.c, top_blob.elempack);
        }
        if (top_blob.dims == 4)
        {
            sprintf(out_shape_str, "[%3d, %3d, %3d, %3d *%d]", top_blob.w, top_blob.h, top_blob.d, top_blob.c, top_blob.elempack);
        }

        fprintf(stderr, "    | %22s -> %-22s", in_shape_str, out_shape_str);

        if (layer->type == "Convolution")
        {
            fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                    ((Convolution*)layer)->kernel_w,
                    ((Convolution*)layer)->kernel_h,
                    ((Convolution*)layer)->stride_w,
                    ((Convolution*)layer)->stride_h);
        }
        else if (layer->type == "ConvolutionDepthWise")
        {
            fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                    ((ConvolutionDepthWise*)layer)->kernel_w,
                    ((ConvolutionDepthWise*)layer)->kernel_h,
                    ((ConvolutionDepthWise*)layer)->stride_w,
                    ((ConvolutionDepthWise*)layer)->stride_h);
        }
        else if (layer->type == "Deconvolution")
        {
            fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                    ((Deconvolution*)layer)->kernel_w,
                    ((Deconvolution*)layer)->kernel_h,
                    ((Deconvolution*)layer)->stride_w,
                    ((Deconvolution*)layer)->stride_h);
        }
        else if (layer->type == "DeconvolutionDepthWise")
        {
            fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                    ((DeconvolutionDepthWise*)layer)->kernel_w,
                    ((DeconvolutionDepthWise*)layer)->kernel_h,
                    ((DeconvolutionDepthWise*)layer)->stride_w,
                    ((DeconvolutionDepthWise*)layer)->stride_h);
        }
        else if (layer->type == "Convolution3D")
        {
            fprintf(stderr, "     kernel: %1d x %1d x %1d    stride: %1d x %1d x %1d",
                    ((Convolution3D*)layer)->kernel_w,
                    ((Convolution3D*)layer)->kernel_h,
                    ((Convolution3D*)layer)->kernel_d,
                    ((Convolution3D*)layer)->stride_w,
                    ((Convolution3D*)layer)->stride_h,
                    ((Convolution3D*)layer)->stride_d);
        }
        else if (layer->type == "ConvolutionDepthWise3D")
        {
            fprintf(stderr, "     kernel: %1d x %1d x %1d    stride: %1d x %1d x %1d",
                    ((ConvolutionDepthWise3D*)layer)->kernel_w,
                    ((ConvolutionDepthWise3D*)layer)->kernel_h,
                    ((ConvolutionDepthWise3D*)layer)->kernel_d,
                    ((ConvolutionDepthWise3D*)layer)->stride_w,
                    ((ConvolutionDepthWise3D*)layer)->stride_h,
                    ((ConvolutionDepthWise3D*)layer)->stride_d);
        }
        else if (layer->type == "Deconvolution3D")
        {
            fprintf(stderr, "     kernel: %1d x %1d x %1d    stride: %1d x %1d x %1d",
                    ((Deconvolution3D*)layer)->kernel_w,
                    ((Deconvolution3D*)layer)->kernel_h,
                    ((Deconvolution3D*)layer)->kernel_d,
                    ((Deconvolution3D*)layer)->stride_w,
                    ((Deconvolution3D*)layer)->stride_h,
                    ((Deconvolution3D*)layer)->stride_d);
        }
        else if (layer->type == "DeconvolutionDepthWise3D")
        {
            fprintf(stderr, "     kernel: %1d x %1d x %1d    stride: %1d x %1d x %1d",
                    ((DeconvolutionDepthWise3D*)layer)->kernel_w,
                    ((DeconvolutionDepthWise3D*)layer)->kernel_h,
                    ((DeconvolutionDepthWise3D*)layer)->kernel_d,
                    ((DeconvolutionDepthWise3D*)layer)->stride_w,
                    ((DeconvolutionDepthWise3D*)layer)->stride_h,
                    ((DeconvolutionDepthWise3D*)layer)->stride_d);
        }
        fprintf(stderr, "\n");
    }
}

#endif // NCNN_BENCHMARK

} // namespace ncnn
