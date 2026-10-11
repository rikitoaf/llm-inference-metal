// GPU timing of Splash paged attention, INT8 vs TurboQuant, outside the engine.
//   attn_perf FLATTENED.metal HISTORY [ROWS]
// Pages are filled with random bytes (finite scales and norms): only the timing is meaningful.
// Prefill: one chunk of ROWS rows (default 2048) after HISTORY tokens, split + reduce, as
// PagedAttention::addPrefill encodes it. Verify: one lane of 8 rows after HISTORY tokens.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kHeads = 4, kGroup = 6, kDim = 256, kPageTokens = 32;

struct Format {
  const char *name;
  uint32_t keyData, keyScales, valueData, valueScales;
  const char *prefillSplit, *prefillReduce, *verifySplit, *verifyReduce;
};
const Format kFormats[] = {
    {"int8", kHeads * kPageTokens * kDim, kHeads * kPageTokens * 4, kHeads * kPageTokens * kDim,
     kHeads * kPageTokens * 4, "prefill_attention_q8_split", "prefill_attention_reduce",
     "verify_attention_q8_split", "verify_attention_reduce"},
    {"turbo", kHeads * kPageTokens * 8 * 4 * 4, kHeads * kPageTokens * 2 * 2, kHeads * kPageTokens * 8 * 3 * 4,
     kHeads * kPageTokens * 2 * 2, "prefill_attention_turbo_split", "prefill_attention_reduce_turbo",
     "verify_attention_turbo_split", "verify_attention_reduce_turbo"},
    {"tq-i4", kHeads * kPageTokens * kDim / 2, kHeads * kPageTokens * 4, kHeads * kPageTokens * kDim / 2,
     kHeads * kPageTokens * 4, "prefill_attention_turbo_int4_split", "prefill_attention_reduce_turbo_int4",
     "verify_attention_turbo_int4_split", "verify_attention_reduce_turbo_int4"},
};

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, const char *name) {
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:@(name)];
  id<MTLComputePipelineState> state = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
  if (!state) { fprintf(stderr, "pipeline %s: %s\n", name, error ? error.localizedDescription.UTF8String : "missing"); exit(1); }
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
}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: attn_perf FLATTENED.metal HISTORY [ROWS]\n"); return 2; }
    const uint32_t history = uint32_t(atoi(argv[2]));
    const uint32_t rows = argc > 3 ? uint32_t(atoi(argv[3])) : 2048;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr, "compile: %s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::mt19937 rng(5);

    const uint32_t pages = (history + rows + kPageTokens - 1) / kPageTokens;
    const uint32_t tiles = (rows + 7) / 8;
    const uint32_t prefillSplits = std::clamp(32u / tiles, 1u, 32u);
    const uint32_t verifyPages = (history + 8 + kPageTokens - 1) / kPageTokens;
    const uint32_t verifySplits = std::min(128u, std::max(32u, (verifyPages + 15) / 16));
    const uint32_t fused = 8 * kGroup;

    id<MTLBuffer> queries = [device newBufferWithLength:uint64_t(kHeads) * std::max(rows, 32u) * kGroup * kDim * 2
                                                options:MTLResourceStorageModePrivate];
    id<MTLBuffer> output = [device newBufferWithLength:queries.length options:MTLResourceStorageModePrivate];
    const uint64_t slots = std::max(uint64_t(tiles) * kHeads * prefillSplits, uint64_t(kHeads) * verifySplits);
    id<MTLBuffer> partials = [device newBufferWithLength:slots * fused * kDim * 4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> statistics = [device newBufferWithLength:slots * fused * 2 * 4 options:MTLResourceStorageModePrivate];
    {  // random bf16 queries
      std::vector<uint16_t> q(queries.length / 2);
      std::normal_distribution<float> normal(0, 1);
      for (auto &v : q) { float f = normal(rng); uint32_t b; memcpy(&b, &f, 4); v = uint16_t(b >> 16); }
      id<MTLBuffer> staging = [device newBufferWithBytes:q.data() length:queries.length options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> c = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [c blitCommandEncoder];
      [blit copyFromBuffer:staging sourceOffset:0 toBuffer:queries destinationOffset:0 size:queries.length];
      [blit endEncoding]; [c commit]; [c waitUntilCompleted];
    }

    printf("history %u + chunk %u rows: %u pages; prefill %u tiles x %u splits; verify %u splits\n",
           history, rows, pages, tiles, prefillSplits, verifySplits);
    for (const Format &format : kFormats) {
      const uint64_t layerPage = uint64_t(format.keyData) + format.keyScales + format.valueData + format.valueScales;
      const uint64_t bytes = (pages * layerPage + 16383) / 16384 * 16384;
      id<MTLBuffer> extent = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      auto *base = static_cast<uint8_t *>(extent.contents);
      for (uint64_t i = 0; i < bytes; i += 4) { uint32_t r = rng(); memcpy(base + i, &r, 4); }
      // finite scales / norms in the two scale regions (each holds `pages` pages)
      const uint64_t keyScales = uint64_t(pages) * format.keyData;
      const uint64_t valueScales = uint64_t(pages) * (format.keyData + format.keyScales + format.valueData);
      for (uint64_t region : {keyScales, valueScales}) {
        if (strcmp(format.name, "turbo")) {
          auto *s = reinterpret_cast<float *>(base + region);
          for (uint64_t i = 0; i < uint64_t(pages) * format.keyScales / 4; ++i) s[i] = 0.01f;
        } else {
          auto *s = reinterpret_cast<uint16_t *>(base + region);
          for (uint64_t i = 0; i < uint64_t(pages) * format.keyScales / 2; ++i) s[i] = 0x3c00;  // half 1.0
        }
      }
      std::vector<uint64_t> table(pages);
      for (uint32_t p = 0; p < pages; ++p) table[p] = extent.gpuAddress | p;
      id<MTLBuffer> pageTable = [device newBufferWithBytes:table.data() length:pages * 8 options:MTLResourceStorageModeShared];

      struct { uint32_t committed, rows, stride, entries, extentPages, offset, splits; } prefill = {
          history, rows, (rows + 31) / 32 * 32, pages, pages, 0, prefillSplits};
      struct { uint32_t committed, entries, extentPages, offset, splitCount, slotSplits; } verify = {
          history, verifyPages, pages, 0, verifySplits, verifySplits};
      id<MTLComputePipelineState> ps = pipeline(device, library, format.prefillSplit);
      id<MTLComputePipelineState> pr = pipeline(device, library, format.prefillReduce);
      id<MTLComputePipelineState> vs = pipeline(device, library, format.verifySplit);
      id<MTLComputePipelineState> vr = pipeline(device, library, format.verifyReduce);

      const double prefillMs = runTimed(queue, ^(id<MTLComputeCommandEncoder> e) {
        [e useResource:extent usage:MTLResourceUsageRead];
        [e setComputePipelineState:ps];
        [e setBuffer:queries offset:0 atIndex:0]; [e setBuffer:partials offset:0 atIndex:1];
        [e setBuffer:statistics offset:0 atIndex:2]; [e setBuffer:pageTable offset:0 atIndex:3];
        [e setBytes:&prefill length:sizeof(prefill) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(kHeads, tiles, prefillSplits) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:pr];
        [e setBuffer:partials offset:0 atIndex:0]; [e setBuffer:statistics offset:0 atIndex:1];
        [e setBuffer:output offset:0 atIndex:2]; [e setBytes:&prefill length:sizeof(prefill) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(kHeads, fused, tiles) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }, 3);
      const double verifyMs = runTimed(queue, ^(id<MTLComputeCommandEncoder> e) {
        [e useResource:extent usage:MTLResourceUsageRead];
        [e setComputePipelineState:vs];
        [e setBuffer:queries offset:0 atIndex:0]; [e setBuffer:partials offset:0 atIndex:1];
        [e setBuffer:statistics offset:0 atIndex:2];
        for (uint32_t i = 3; i <= 6; ++i) [e setBuffer:pageTable offset:0 atIndex:i];
        [e setBytes:&verify length:sizeof(verify) atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(kHeads, verifySplits, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:vr];
        [e setBuffer:partials offset:0 atIndex:0]; [e setBuffer:statistics offset:0 atIndex:1];
        [e setBuffer:output offset:0 atIndex:2]; [e setBytes:&verify length:sizeof(verify) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(kHeads, fused, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }, 5);
      printf("  %-5s layer: prefill chunk %8.2f ms (%6.0f rows/s), verify step %6.3f ms   [tg mem split %lu]\n",
             format.name, prefillMs, rows / (prefillMs / 1000.0), verifyMs, (unsigned long)ps.staticThreadgroupMemoryLength);
    }
  }
  return 0;
}
