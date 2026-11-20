// Isolated GPU timing + output check of the turbo-int4 prefill attention (split + reduce) variants (ATTN.md).
//   prefill_attn_bench FLATTENED.metal HISTORY[,HISTORY...] [ROWS] [VARIANT...]
// FLATTENED = flatten_metal.py RUNTIME tools/prefill_attn_variants.metal. One 2048-row chunk (ROWS) after HISTORY
// committed tokens, the grid PagedAttention::addPrefill encodes (32 / tiles splits, at least 1: 1 at 2048 rows). Random bf16 queries, random
// int4 K/V with per-token scales chosen so the softmax is peaked (running-max rescales happen), finite everywhere.
// Each variant: best-of-REPEATS GPU ms of the split and of the reduce, TFLOP/s on causal FLOPs
// (4 x 24 x 256 x sum of causal keys), and its output against the production kernels' output (max |diff|, and the
// largest diff relative to the row's max |value|). VARIANT = name:split:reduce:rows (rows per query tile).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kHeads = 4, kGroup = 6, kDim = 256, kPageTokens = 32;
constexpr uint32_t kKeyData = kHeads * kPageTokens * kDim / 2, kScales = kHeads * kPageTokens * 4;

struct Variant { std::string name, split, reduce; uint32_t rows; };

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, const std::string &name) {
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
  id<MTLComputePipelineState> state = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
  if (!state) { fprintf(stderr, "pipeline %s: %s\n", name.c_str(), error ? error.localizedDescription.UTF8String : "missing"); exit(1); }
  return state;
}

double runTimed(id<MTLCommandQueue> queue, void (^encode)(id<MTLComputeCommandEncoder>), int repeats) {
  double best = 1e30;
  for (int i = 0; i < repeats; ++i) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode(encoder);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.error) { fprintf(stderr, "GPU error: %s\n", command.error.localizedDescription.UTF8String); exit(1); }
    best = std::min(best, (command.GPUEndTime - command.GPUStartTime) * 1000.0);
  }
  return best;
}

float bf16(uint16_t v) { uint32_t b = uint32_t(v) << 16; float f; memcpy(&f, &b, 4); return f; }
}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: prefill_attn_bench FLATTENED.metal HISTORY[,HISTORY...] [ROWS] [name:split:reduce:rows ...]\n"); return 2; }
    std::vector<uint32_t> histories;
    { std::stringstream list(argv[2]); std::string item; while (std::getline(list, item, ',')) histories.push_back(uint32_t(atoi(item.c_str()))); }
    const uint32_t rows = argc > 3 ? uint32_t(atoi(argv[3])) : 2048;
    const int repeats = getenv("BENCH_REPEATS") ? atoi(getenv("BENCH_REPEATS")) : 3;
    std::vector<Variant> variants = {{"production", "prefill_attention_turbo_int4_split", "prefill_attention_reduce_turbo_int4", 8}};
    for (int i = 4; i < argc; ++i) {
      std::stringstream spec(argv[i]); Variant v; std::string r;
      std::getline(spec, v.name, ':'); std::getline(spec, v.split, ':'); std::getline(spec, v.reduce, ':'); std::getline(spec, r, ':');
      v.rows = uint32_t(atoi(r.c_str()));
      variants.push_back(v);
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr, "compile: %s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::mt19937 rng(5);
    const uint32_t stride = (rows + 31) / 32 * 32;
    const uint64_t queryBytes = uint64_t(kHeads) * stride * kGroup * kDim * 2;
    id<MTLBuffer> queries = [device newBufferWithLength:queryBytes options:MTLResourceStorageModeShared];
    {
      std::normal_distribution<float> normal(0, 1);
      auto *q = static_cast<uint16_t *>(queries.contents);
      for (uint64_t i = 0; i < queryBytes / 2; ++i) { float f = normal(rng); uint32_t b; memcpy(&b, &f, 4); q[i] = uint16_t(b >> 16); }
    }
    uint32_t minRows = 1000;
    for (const Variant &v : variants) minRows = std::min(minRows, v.rows);
    const uint64_t slots = uint64_t((rows + minRows - 1) / minRows) * kHeads;  // 1 split at 2048 rows
    const uint64_t fused = uint64_t(minRows) * kGroup;
    id<MTLBuffer> partials = [device newBufferWithLength:slots * fused * kDim * 4 * 32 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> statistics = [device newBufferWithLength:slots * fused * 2 * 4 * 32 options:MTLResourceStorageModePrivate];
    std::vector<id<MTLBuffer>> outputs;
    for (size_t i = 0; i < variants.size(); ++i)
      outputs.push_back([device newBufferWithLength:queryBytes options:MTLResourceStorageModeShared]);

    for (uint32_t history : histories) {
      const uint32_t pages = (history + rows + kPageTokens - 1) / kPageTokens;
      const uint64_t layerPage = 2ull * (kKeyData + kScales);
      const uint64_t bytes = (pages * layerPage + 16383) / 16384 * 16384;
      id<MTLBuffer> extent = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      auto *base = static_cast<uint8_t *>(extent.contents);
      for (uint64_t i = 0; i < bytes; i += 4) { uint32_t r = rng(); memcpy(base + i, &r, 4); }
      // Key scales: scores of a few units (peaked softmax, frequent rescales); value scales ~ int4 norms.
      std::uniform_real_distribution<float> keyScale(0.2f, 0.9f), valueScale(0.01f, 0.05f);
      auto *ks = reinterpret_cast<float *>(base + uint64_t(pages) * kKeyData);
      auto *vs = reinterpret_cast<float *>(base + uint64_t(pages) * (2ull * kKeyData + kScales));
      for (uint64_t i = 0; i < uint64_t(pages) * kScales / 4; ++i) { ks[i] = keyScale(rng); vs[i] = valueScale(rng); }
      std::vector<uint64_t> table(pages);
      for (uint32_t p = 0; p < pages; ++p) table[p] = extent.gpuAddress | p;
      id<MTLBuffer> pageTable = [device newBufferWithBytes:table.data() length:pages * 8 options:MTLResourceStorageModeShared];
      double keys = 0;
      for (uint32_t r = 0; r < rows; ++r) keys += double(history) + r + 1;
      const double flop = 4.0 * kHeads * kGroup * kDim * keys;
      printf("history %u + chunk %u rows (%u pages):\n", history, rows, pages);
      double productionMs = 0;
      for (size_t index = 0; index < variants.size(); ++index) {
        const Variant &variant = variants[index];
        id<MTLComputePipelineState> ps = pipeline(device, library, variant.split);
        id<MTLComputePipelineState> pr = pipeline(device, library, variant.reduce);
        const uint32_t tiles = (rows + variant.rows - 1) / variant.rows;
        // PagedAttention::prefillSplits: 32 split slots shared out over the chunk's tiles.
        const uint32_t splits = std::clamp(32u / tiles, 1u, 32u);
        struct { uint32_t committed, rows, stride, entries, extentPages, offset, splits; } params = {
            history, rows, stride, pages, pages, 0, splits};
        id<MTLBuffer> output = outputs[index];
        void (^split)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> e) {
          [e useResource:extent usage:MTLResourceUsageRead];
          [e setComputePipelineState:ps];
          [e setBuffer:queries offset:0 atIndex:0]; [e setBuffer:partials offset:0 atIndex:1];
          [e setBuffer:statistics offset:0 atIndex:2]; [e setBuffer:pageTable offset:0 atIndex:3];
          [e setBytes:&params length:sizeof(params) atIndex:4];
          [e dispatchThreadgroups:MTLSizeMake(kHeads, tiles, splits) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        void (^reduce)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> e) {
          [e setComputePipelineState:pr];
          [e setBuffer:partials offset:0 atIndex:0]; [e setBuffer:statistics offset:0 atIndex:1];
          [e setBuffer:output offset:0 atIndex:2]; [e setBytes:&params length:sizeof(params) atIndex:3];
          [e dispatchThreadgroups:MTLSizeMake(kHeads, variant.rows * kGroup, tiles) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        const double splitMs = runTimed(queue, split, repeats);
        runTimed(queue, ^(id<MTLComputeCommandEncoder> e) { split(e); reduce(e); }, 1);  // output of this variant
        const double reduceMs = runTimed(queue, reduce, repeats);
        const double totalMs = splitMs + reduceMs;
        if (index == 0) productionMs = totalMs;
        double maxDiff = 0, maxRelative = 0;
        uint64_t argmaxFlips = 0, rowsChecked = 0;
        const auto *a = static_cast<const uint16_t *>(outputs[0].contents);
        const auto *b = static_cast<const uint16_t *>(output.contents);
        for (uint32_t head = 0; head < kHeads; ++head)
          for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t g = 0; g < kGroup; ++g) {
              const uint64_t at = ((uint64_t(head) * stride + r) * kGroup + g) * kDim;
              double rowMax = 0, rowDiff = 0; uint32_t ia = 0, ib = 0;
              for (uint32_t d = 0; d < kDim; ++d) {
                const float x = bf16(a[at + d]), y = bf16(b[at + d]);
                rowMax = std::max(rowMax, double(std::fabs(x)));
                rowDiff = std::max(rowDiff, double(std::fabs(x - y)));
                if (x > bf16(a[at + ia])) ia = d;
                if (y > bf16(b[at + ib])) ib = d;
              }
              maxDiff = std::max(maxDiff, rowDiff);
              if (rowMax > 0) maxRelative = std::max(maxRelative, rowDiff / rowMax);
              argmaxFlips += ia != ib; ++rowsChecked;
            }
        printf("  %-12s split %8.2f ms  reduce %6.2f ms  total %8.2f ms  %5.1f TFLOP/s  %+6.1f%%  max|diff| %.3g  rel %.3g  argmax flips %llu/%llu  tg %lu\n",
               variant.name.c_str(), splitMs, reduceMs, totalMs, flop / (totalMs * 1e-3) / 1e12,
               (totalMs / productionMs - 1) * 100, maxDiff, maxRelative, (unsigned long long)argmaxFlips,
               (unsigned long long)rowsChecked, (unsigned long)ps.staticThreadgroupMemoryLength);
        fflush(stdout);
      }
    }
  }
  return 0;
}
