// GPU side of tools/turbo_attention_test.py: one verify lane of TurboQuant attention over a
// paged history, exactly as the engine encodes it (store, query rotation, split, reduce).
//   turbo_attn_harness FLATTENED.metal WORKDIR COMMITTED SPLITS
// Reads keys.f32 [head][token][dim] and values.f32 [head][dim][token] for COMMITTED + 8 tokens
// (stride = tokens rounded up to 32) and queries.f32 [head][32 rows][6][dim]; writes output.f32.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t kHeads = 4, kGroup = 6, kDim = 256, kRows = 8, kStride = 32, kPageTokens = 32;
constexpr uint32_t kTurboLayerPage = kHeads * kPageTokens * (8 * 4 * 4 + 2 * 2 + 8 * 3 * 4 + 2 * 2);
constexpr uint32_t kInt4LayerPage = kHeads * kPageTokens * (128 + 4 + 128 + 4);

std::vector<float> readFloats(NSString *path, size_t count) {
  std::vector<float> data(count);
  FILE *f = fopen(path.UTF8String, "rb");
  if (!f || fread(data.data(), sizeof(float), count, f) != count) {
    fprintf(stderr, "cannot read %s\n", path.UTF8String);
    exit(1);
  }
  fclose(f);
  return data;
}

id<MTLBuffer> bf16Buffer(id<MTLDevice> device, const std::vector<float> &values) {
  id<MTLBuffer> buffer = [device newBufferWithLength:values.size() * 2 options:MTLResourceStorageModeShared];
  auto *out = static_cast<uint16_t *>(buffer.contents);
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t bits;
    memcpy(&bits, &values[i], 4);
    out[i] = uint16_t(bits >> 16);
  }
  return buffer;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  id<MTLComputePipelineState> state = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
  if (!state) {
    fprintf(stderr, "pipeline %s: %s\n", name.UTF8String, error ? error.localizedDescription.UTF8String : "missing");
    exit(1);
  }
  return state;
}
}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 5) { fprintf(stderr, "usage: turbo_attn_harness FLATTENED.metal WORKDIR COMMITTED SPLITS [int4]\n"); return 2; }
    const bool int4 = argc > 5 && !strcmp(argv[5], "int4");
    const uint32_t kLayerPage = int4 ? kInt4LayerPage : kTurboLayerPage;
    NSString *work = @(argv[2]);
    const uint32_t committed = uint32_t(atoi(argv[3])), splits = uint32_t(atoi(argv[4]));
    const uint32_t tokens = committed + kRows;
    const uint32_t stride = (tokens + 31) / 32 * 32;
    const uint32_t pages = (tokens + kPageTokens - 1) / kPageTokens;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr, "compile: %s\n", error.localizedDescription.UTF8String); return 1; }

    id<MTLBuffer> chunkKeys = bf16Buffer(device, readFloats([work stringByAppendingPathComponent:@"keys.f32"], kHeads * stride * kDim));
    id<MTLBuffer> chunkValues = bf16Buffer(device, readFloats([work stringByAppendingPathComponent:@"values.f32"], kHeads * stride * kDim));
    id<MTLBuffer> queries = bf16Buffer(device, readFloats([work stringByAppendingPathComponent:@"queries.f32"], kHeads * kStride * kGroup * kDim));

    // One extent of `pages` pages, garbage-filled; pages in reverse order in the table so that
    // logical and physical positions differ.
    id<MTLBuffer> extent = [device newBufferWithLength:(uint64_t(pages) * kLayerPage + 16383) / 16384 * 16384
                                               options:MTLResourceStorageModeShared];
    memset(extent.contents, 0x7F, extent.length);
    std::vector<uint64_t> table(pages);
    for (uint32_t page = 0; page < pages; ++page) table[page] = extent.gpuAddress | (pages - 1 - page);
    id<MTLBuffer> pageTable = [device newBufferWithBytes:table.data() length:pages * 8 options:MTLResourceStorageModeShared];

    struct { uint32_t committed, chunk, stride, entries, extentPages, offset; } store = {0, tokens, stride, pages, pages, 0};
    struct { uint32_t committed, entries, extentPages, offset, splitCount, slotSplits; } verify = {committed, pages, pages, 0, splits, splits};
    const uint32_t vectors = kHeads * kStride * kGroup;
    const uint32_t fused = kRows * kGroup;
    id<MTLBuffer> partials = [device newBufferWithLength:uint64_t(kHeads) * splits * fused * kDim * 4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> statistics = [device newBufferWithLength:uint64_t(kHeads) * splits * fused * 2 * 4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> output = [device newBufferWithLength:uint64_t(vectors) * kDim * 2 options:MTLResourceStorageModeShared];
    memset(output.contents, 0, output.length);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder useResource:extent usage:MTLResourceUsageRead | MTLResourceUsageWrite];

    [encoder setComputePipelineState:pipeline(device, library, int4 ? @"prefill_attention_turbo_int4_store" : @"prefill_attention_turbo_store")];
    [encoder setBuffer:chunkKeys offset:0 atIndex:0];
    [encoder setBuffer:chunkValues offset:0 atIndex:1];
    [encoder setBuffer:pageTable offset:0 atIndex:2];
    [encoder setBytes:&store length:sizeof(store) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(2 * tokens * kHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:pipeline(device, library, int4 ? @"turbo_rotate_queries_256" : @"turbo_rotate_queries")];
    [encoder setBuffer:queries offset:0 atIndex:0];
    [encoder setBytes:&vectors length:4 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(int4 ? (vectors + 7) / 8 : (vectors * 2 + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:pipeline(device, library, int4 ? @"verify_attention_turbo_int4_split" : @"verify_attention_turbo_split")];
    [encoder setBuffer:queries offset:0 atIndex:0];
    [encoder setBuffer:partials offset:0 atIndex:1];
    [encoder setBuffer:statistics offset:0 atIndex:2];
    for (uint32_t index = 3; index <= 6; ++index) [encoder setBuffer:pageTable offset:0 atIndex:index];
    [encoder setBytes:&verify length:sizeof(verify) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(kHeads, splits, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:pipeline(device, library, int4 ? @"verify_attention_reduce_turbo_int4" : @"verify_attention_reduce_turbo")];
    [encoder setBuffer:partials offset:0 atIndex:0];
    [encoder setBuffer:statistics offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&verify length:sizeof(verify) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(kHeads, fused, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.error) { fprintf(stderr, "GPU error: %s\n", command.error.localizedDescription.UTF8String); return 1; }

    std::vector<float> result(uint64_t(vectors) * kDim);
    const auto *out = static_cast<const uint16_t *>(output.contents);
    for (size_t i = 0; i < result.size(); ++i) {
      uint32_t bits = uint32_t(out[i]) << 16;
      memcpy(&result[i], &bits, 4);
    }
    FILE *f = fopen([work stringByAppendingPathComponent:@"output.f32"].UTF8String, "wb");
    fwrite(result.data(), 4, result.size(), f);
    fclose(f);
    printf("attention harness: %u history tokens + %u rows over %u pages, %u splits\n", committed, kRows, pages, splits);
  }
  return 0;
}
