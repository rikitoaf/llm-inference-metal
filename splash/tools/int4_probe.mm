#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@"int4_probe.metal" encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions *o = [MTLCompileOptions new]; o.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
    if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"int4_probe"] error:&err];
    const int M = 48, N = 32, D = 256;
    id<MTLBuffer> q = [dev newBufferWithLength:M * D * 2 options:MTLResourceStorageModeShared];
    uint16_t *qp = (uint16_t *)q.contents; memset(qp, 0, M * D * 2);
    for (int m = 0; m < M; ++m) qp[m * D + m] = 0x3f80;            // bf16 1.0 at dimension m
    id<MTLBuffer> k = [dev newBufferWithLength:N * D / 2 options:MTLResourceStorageModeShared];
    uint8_t *kp = (uint8_t *)k.contents;
    for (int b = 0; b < N * D / 2; ++b) kp[b] = uint8_t(((b * 3) & 0xF) | (((b * 5 + 1) & 0xF) << 4));
    id<MTLBuffer> out = [dev newBufferWithLength:M * N * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> qu = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [qu commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps]; [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:k offset:0 atIndex:1]; [e setBuffer:out offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const float *s = (const float *)out.contents;   // [m][n]
    int lowFirst = 0, highFirst = 0, total = 0;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        const int elem = n * D + m, byte = elem / 2;
        const int lo = kp[byte] & 0xF, hi = kp[byte] >> 4;
        auto sx = [](int v) { return v >= 8 ? v - 16 : v; };
        const int wantLow = sx(elem % 2 == 0 ? lo : hi), wantHigh = sx(elem % 2 == 0 ? hi : lo);
        const float got = s[m * N + n];
        lowFirst += got == wantLow; highFirst += got == wantHigh; ++total;
      }
    printf("elements %d: match low-nibble-first signed %d, high-nibble-first signed %d\n", total, lowFirst, highFirst);
    printf("sample: m=0 n=0 got %.1f (byte 0x%02x), m=1 n=0 got %.1f\n", s[0], kp[0], s[N]);
  }
  return 0;
}
