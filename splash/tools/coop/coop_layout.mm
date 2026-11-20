// coop_layout FILE.metal KERNEL: prints matmul2d cooperative operand layouts per lane.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *e = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&e];
    MTLCompileOptions *o = [MTLCompileOptions new]; o.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&e];
    if (!lib) { fprintf(stderr, "%s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(argv[2])] error:&e];
    if (!ps) { fprintf(stderr, "%s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLBuffer> b = [dev newBufferWithLength:32 * 1024 * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue]; id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:ps]; [en setBuffer:b offset:0 atIndex:0];
    [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [en endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    const int *v = (const int *)b.contents;
    for (int lane = 0; lane < 32; ++lane) {
      const int *x = v + lane * 1024; int rc = x[0], lc = x[1], at = 2;
      printf("lane %2d right(%d):", lane, rc); for (int i = 0; i < rc; ++i, at += 3) printf(" %s(%d,%d)", x[at] ? "" : "x", x[at+1], x[at+2]);
      printf("\n        left(%d):", lc); for (int i = 0; i < lc; ++i, at += 3) printf(" %s(%d,%d)", x[at] ? "" : "x", x[at+1], x[at+2]);
      printf("\n");
    }
  }
}
