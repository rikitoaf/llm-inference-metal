// Probe: compile a flattened Splash kernel source with Metal's runtime compiler and build pipelines.
//   rtc_probe FILE.metal [kernel ...]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: rtc_probe FILE.metal [kernel...]\n"); return 2; }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    if (!src) { fprintf(stderr, "read: %s\n", error.localizedDescription.UTF8String); return 1; }
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;  // the offline build does not pass -ffast-math
    auto t0 = std::chrono::steady_clock::now();
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:options error:&error];
    auto t1 = std::chrono::steady_clock::now();
    if (!lib) { fprintf(stderr, "compile failed: %s\n", error.localizedDescription.UTF8String); return 1; }
    printf("compiled in %.2f s, %lu functions\n", std::chrono::duration<double>(t1 - t0).count(), (unsigned long)lib.functionNames.count);
    for (int i = 2; i < argc; ++i) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(argv[i])];
      if (!fn) { printf("  %s: MISSING\n", argv[i]); continue; }
      auto p0 = std::chrono::steady_clock::now();
      id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&error];
      auto p1 = std::chrono::steady_clock::now();
      printf("  %s: %s (%.2f s, max threads %lu, tg mem %lu)\n", argv[i], ps ? "pipeline OK" : error.localizedDescription.UTF8String,
             std::chrono::duration<double>(p1 - p0).count(), ps ? (unsigned long)ps.maxTotalThreadsPerThreadgroup : 0UL,
             ps ? (unsigned long)ps.staticThreadgroupMemoryLength : 0UL);
    }
  }
  return 0;
}
