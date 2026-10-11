// GPU side of tools/turbo_kernel_test.py: runs the TurboQuant store, the dequantizers and the
// query rotation on one KV page (4 heads x 32 tokens x 256 dims) with Metal's runtime compiler.
//   turbo_harness FLATTENED.metal WORKDIR   (reads keys.f32 values.f32 queries.f32, writes *_out.f32)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t kHeads = 4, kTokens = 32, kDim = 256, kQueries = 40;
// SplashTurboPageBytes<4>: key planes, norms, value planes, norms of one layer page.
constexpr uint32_t kKeyData = kHeads * kTokens * 8 * 4 * 4;
constexpr uint32_t kNorms = kHeads * kTokens * 2 * 2;
constexpr uint32_t kValueData = kHeads * kTokens * 8 * 3 * 4;

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

void writeFloats(NSString *path, const float *data, size_t count) {
  FILE *f = fopen(path.UTF8String, "wb");
  fwrite(data, sizeof(float), count, f);
  fclose(f);
}

uint16_t toBf16(float value) {   // inputs are bf16-representable already
  uint32_t bits;
  memcpy(&bits, &value, 4);
  return uint16_t(bits >> 16);
}

float fromBf16(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  memcpy(&result, &bits, 4);
  return result;
}

id<MTLBuffer> bf16Buffer(id<MTLDevice> device, const std::vector<float> &values) {
  id<MTLBuffer> buffer = [device newBufferWithLength:values.size() * 2 options:MTLResourceStorageModeShared];
  auto *out = static_cast<uint16_t *>(buffer.contents);
  for (size_t i = 0; i < values.size(); ++i) out[i] = toBf16(values[i]);
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
    if (argc < 3) { fprintf(stderr, "usage: turbo_harness FLATTENED.metal WORKDIR [int4]\n"); return 2; }
    const bool int4 = argc > 3 && !strcmp(argv[3], "int4");
    NSString *work = @(argv[2]);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { fprintf(stderr, "compile: %s\n", error.localizedDescription.UTF8String); return 1; }

    auto keys = readFloats([work stringByAppendingPathComponent:@"keys.f32"], kHeads * kTokens * kDim);
    auto values = readFloats([work stringByAppendingPathComponent:@"values.f32"], kHeads * kTokens * kDim);
    auto queries = readFloats([work stringByAppendingPathComponent:@"queries.f32"], kQueries * kDim);
    id<MTLBuffer> chunkKeys = bf16Buffer(device, keys);
    id<MTLBuffer> chunkValues = bf16Buffer(device, values);
    id<MTLBuffer> queryBuffer = bf16Buffer(device, queries);

    // One extent of one page; garbage first, so nothing passes by accident.
    const uint32_t layerPage = kKeyData + kNorms + kValueData + kNorms;
    id<MTLBuffer> extent = [device newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    memset(extent.contents, 0xA5, extent.length);
    id<MTLBuffer> pageTable = [device newBufferWithLength:8 options:MTLResourceStorageModeShared];
    const uint64_t entry = extent.gpuAddress | 0;   // page 0 of the extent
    memcpy(pageTable.contents, &entry, 8);
    if (extent.gpuAddress % 16384) { fprintf(stderr, "extent not 16 KiB aligned\n"); return 1; }

    struct { uint32_t committed, chunk, stride, entries, extentPages, offset; } params = {0, kTokens, kTokens, 1, 1, 0};
    struct { uint32_t extentPages, offset; } layer = {1, 0};
    uint32_t vectors = kQueries;

    id<MTLBuffer> keysOut = [device newBufferWithLength:kHeads * kTokens * kDim * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> valuesOut = [device newBufferWithLength:kHeads * kTokens * kDim * 4 options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder useResource:extent usage:MTLResourceUsageRead | MTLResourceUsageWrite];

    [encoder setComputePipelineState:pipeline(device, library, int4 ? @"prefill_attention_turbo_int4_store" : @"prefill_attention_turbo_store")];
    [encoder setBuffer:chunkKeys offset:0 atIndex:0];
    [encoder setBuffer:chunkValues offset:0 atIndex:1];
    [encoder setBuffer:pageTable offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(2 * kTokens * kHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    if (!int4) {
      [encoder setComputePipelineState:pipeline(device, library, @"test_turbo_dump")];
      [encoder setBuffer:pageTable offset:0 atIndex:0];
      [encoder setBytes:&layer length:sizeof(layer) atIndex:1];
      [encoder setBuffer:keysOut offset:0 atIndex:2];
      [encoder setBuffer:valuesOut offset:0 atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    [encoder setComputePipelineState:pipeline(device, library, int4 ? @"turbo_rotate_queries_256" : @"turbo_rotate_queries")];
    [encoder setBuffer:queryBuffer offset:0 atIndex:0];
    [encoder setBytes:&vectors length:4 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(int4 ? (kQueries + 7) / 8 : (kQueries * 2 + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.error) { fprintf(stderr, "GPU error: %s\n", command.error.localizedDescription.UTF8String); return 1; }

    {
      FILE *f = fopen([work stringByAppendingPathComponent:@"extent.bin"].UTF8String, "wb");
      fwrite(extent.contents, 1, extent.length, f);
      fclose(f);
    }
    writeFloats([work stringByAppendingPathComponent:@"keys_out.f32"], static_cast<float *>(keysOut.contents), kHeads * kTokens * kDim);
    writeFloats([work stringByAppendingPathComponent:@"values_out.f32"], static_cast<float *>(valuesOut.contents), kHeads * kTokens * kDim);
    std::vector<float> rotated(kQueries * kDim);
    const auto *q = static_cast<const uint16_t *>(queryBuffer.contents);
    for (size_t i = 0; i < rotated.size(); ++i) rotated[i] = fromBf16(q[i]);
    writeFloats([work stringByAppendingPathComponent:@"queries_out.f32"], rotated.data(), rotated.size());
    printf("harness: stored %u rows, layer page %u bytes, dumped %u heads, rotated %u queries\n",
           2 * kTokens * kHeads, layerPage, kHeads, kQueries);
  }
  return 0;
}
