#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/KvExtent.h"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace splash::kv {

// Selected once for a runtime and its entire page pool. Weight storage is
// independent of the KV format; requests never change it while serving.
// Turbo is TurboQuant (metal/kernels/common/turbo_kv.h): keys at 4 bits and
// values at 3 bits per element on Lloyd-Max codebooks, rotated by a fixed
// randomized Walsh-Hadamard transform per 128 dimensions, with one fp16 norm
// per 128 dimensions. TurboInt4 rotates each head vector by a 256-point
// randomized Hadamard transform and stores keys and values as symmetric int4
// with an INT8-style fp32 scale per (token, head), which MPP reads natively.
enum class Format : uint32_t { Int8 = 1, BFloat16 = 2, Turbo = 3, TurboInt4 = 4 };

[[nodiscard]] constexpr bool validFormat(Format format) noexcept {
  return format == Format::Int8 || format == Format::BFloat16 ||
         format == Format::Turbo || format == Format::TurboInt4;
}

[[nodiscard]] constexpr std::string_view formatName(Format format) noexcept {
  switch (format) {
  case Format::Int8: return "int8";
  case Format::BFloat16: return "bf16";
  case Format::Turbo: return "turbo";
  case Format::TurboInt4: return "turbo-int4";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view storageFormatName(Format format) noexcept {
  switch (format) {
  case Format::Int8:
    return "q8s8_f32_scale_per_token_head_k_token_major_v_dimension_major";
  case Format::BFloat16:
    return "bf16_k_token_major_v_dimension_major";
  case Format::Turbo:
    return "turboquant_k4_v3_wht128_fp16_norm_bitplane_token_major";
  case Format::TurboInt4:
    return "turboquant_wht256_int4_f32_scale_per_token_head_token_major";
  }
  return "invalid";
}

// TurboQuant geometry: 128 dimensions share one rotation and one norm; 4-bit
// keys and 3-bit values are bit-planes of 32 dimensions in one uint each.
inline constexpr uint32_t kTurboGroup = 128;
inline constexpr uint32_t kTurboKeyBits = 4;
inline constexpr uint32_t kTurboValueBits = 3;

// A page whose content goes to another page.
struct PageCopy final {
  uint32_t from = 0;
  uint32_t to = 0;
};

// The memory of the engine's page pool: extents of extentPages() pages,
// pageCount() a whole number of them. Only KvPool allocates and releases
// them. Implementations provide Metal storage or deterministic test storage.
class ExtentStorage {
public:
  virtual ~ExtentStorage() = default;
  [[nodiscard]] virtual uint32_t pageCount() const noexcept = 0;
  [[nodiscard]] virtual uint64_t bytesPerPage() const noexcept = 0;
  [[nodiscard]] virtual uint32_t extentPages() const noexcept = 0;
  // Allocates extent `extent`, which is not allocated: granted, or the
  // refusal's cause with nothing allocated. std::logic_error for an
  // allocated extent.
  [[nodiscard]] virtual metal::AllocationResult allocateExtent(uint32_t extent) = 0;
  // Releases allocated extent `extent` at once. std::logic_error, changing
  // nothing, for an unallocated extent or while a command is in flight (a
  // command reaches extents through its page tables without retaining them).
  virtual void releaseExtent(uint32_t extent) = 0;
  // Copies each page's content onto its destination, both in allocated
  // extents: how the pool moves the pages of an extent it empties.
  // std::logic_error, copying nothing, for a page of an unallocated extent
  // or while a command is in flight (it reaches both pages through its
  // tables and may still write the source).
  virtual void copyPages(std::span<const PageCopy> copies) = 0;
};

// Shared cache format and execution limits; model dimensions live in Layout.
inline constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
inline constexpr uint32_t kMaximumLogicalTokens =
    SPLASH_MAXIMUM_CONTEXT_TOKENS;
inline constexpr uint32_t kMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;
inline constexpr int32_t kQuantizedMinimum = -127;
inline constexpr int32_t kQuantizedMaximum = 127;
// Every tensor region of an extent starts on this boundary. The attention
// kernels were tuned on it, and regions aligned to less cost Apple10's 35B
// verify kernel a fixed ~25 µs per dispatch.
inline constexpr uint64_t kExtentRegionAlignmentBytes = 64 * 1024;
// The size a pool aims its extents at (Layout::extentPagesFor stays within
// half and one and a half times it): each extent costs the serving loop an
// allocation and a release, and an extent returns memory only once all of
// its pages are free.
inline constexpr uint64_t kAllocationExtentTargetBytes = 128ull * 1024 * 1024;

namespace detail {

[[nodiscard]] constexpr uint64_t gcd(uint64_t left, uint64_t right) noexcept {
  while (right) {
    const uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

[[nodiscard]] constexpr uint64_t lcm(uint64_t left, uint64_t right) noexcept {
  return left && right ? left / gcd(left, right) * right : 0;
}

// Pages whose bytes of one tensor fill whole region alignment units.
[[nodiscard]] constexpr uint64_t pagesForAlignedRegion(
    uint64_t bytesPerPage) noexcept {
  return bytesPerPage
             ? kExtentRegionAlignmentBytes /
                   gcd(kExtentRegionAlignmentBytes, bytesPerPage)
             : 0;
}

} // namespace detail

// Physical KV geometry: Page32, either BF16, per-(token, head) symmetric INT8,
// or TurboQuant (4-bit keys, 3-bit values, fp16 norms per 128 dimensions).
// Layer and head counts vary by target.
struct Layout final {
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  Format format = Format::Int8;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return attentionLayers && kvHeads && headDimension && validFormat(format) &&
           (format != Format::Turbo || headDimension % kTurboGroup == 0) &&
           (format != Format::TurboInt4 || headDimension == 256);
  }
  [[nodiscard]] constexpr uint32_t elementsPerScale() const noexcept {
    return format == Format::Int8 || format == Format::TurboInt4 ? headDimension
           : format == Format::Turbo                             ? kTurboGroup
                                                                 : 0;
  }
  [[nodiscard]] constexpr uint64_t elementsPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * headDimension;
  }
  [[nodiscard]] constexpr uint64_t scalesPerTensorLayerPage() const noexcept {
    return format == Format::Int8 || format == Format::TurboInt4
               ? uint64_t{kPageTokens} * kvHeads
               : 0;
  }
  // INT8 and BF16 keys and values share one data and one scale geometry
  // per layer page; TurboQuant keys take more bits than its values.
  [[nodiscard]] constexpr uint64_t dataBytesPerLayerPage() const noexcept {
    return keyDataBytesPerLayerPage();
  }
  [[nodiscard]] constexpr uint64_t scaleBytesPerLayerPage() const noexcept {
    return keyScaleBytesPerLayerPage();
  }
  [[nodiscard]] constexpr uint64_t keyDataBytesPerLayerPage() const noexcept {
    if (format == Format::Turbo)
      return elementsPerLayerPage() * kTurboKeyBits / 8;
    if (format == Format::TurboInt4)
      return elementsPerLayerPage() / 2;
    return elementsPerLayerPage() * (format == Format::Int8 ? 1 : 2);
  }
  [[nodiscard]] constexpr uint64_t valueDataBytesPerLayerPage() const noexcept {
    if (format == Format::Turbo)
      return elementsPerLayerPage() * kTurboValueBits / 8;
    return keyDataBytesPerLayerPage();
  }
  // INT8: one fp32 scale per (token, head); TurboQuant: one fp16 norm per
  // (token, head, 128 dimensions); keys and values alike.
  [[nodiscard]] constexpr uint64_t keyScaleBytesPerLayerPage() const noexcept {
    if (format == Format::Turbo)
      return uint64_t{kPageTokens} * kvHeads * (headDimension / kTurboGroup) * 2;
    return scalesPerTensorLayerPage() * sizeof(float);
  }
  [[nodiscard]] constexpr uint64_t valueScaleBytesPerLayerPage() const noexcept {
    return keyScaleBytesPerLayerPage();
  }
  // The four tensors' bytes in one layer page, as the kernels place them.
  [[nodiscard]] constexpr SplashKvPageGeometry geometry() const noexcept {
    return {static_cast<uint32_t>(keyDataBytesPerLayerPage()),
            static_cast<uint32_t>(keyScaleBytesPerLayerPage()),
            static_cast<uint32_t>(valueDataBytesPerLayerPage()),
            static_cast<uint32_t>(valueScaleBytesPerLayerPage())};
  }
  [[nodiscard]] constexpr uint64_t bytesPerLayerPage() const noexcept {
    return keyDataBytesPerLayerPage() + keyScaleBytesPerLayerPage() +
           valueDataBytesPerLayerPage() + valueScaleBytesPerLayerPage();
  }
  [[nodiscard]] constexpr uint64_t bytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * bytesPerLayerPage();
  }

  // An extent holds a whole number of these pages, so that every tensor
  // region starts 64 KiB-aligned. The INT8 scales are the tightest
  // constraint: 4 heads require 128 pages and 2 heads require 256. BF16
  // needs only 1 or 2 pages. This is allocation geometry only; prefix
  // matching remains Page32 in both cases.
  [[nodiscard]] constexpr uint32_t extentAlignmentPages() const noexcept {
    if (format == Format::BFloat16)
      return static_cast<uint32_t>(
          detail::pagesForAlignedRegion(dataBytesPerLayerPage()));
    if (format == Format::Turbo)
      return static_cast<uint32_t>(detail::lcm(
          detail::lcm(detail::pagesForAlignedRegion(keyDataBytesPerLayerPage()),
                      detail::pagesForAlignedRegion(valueDataBytesPerLayerPage())),
          detail::pagesForAlignedRegion(keyScaleBytesPerLayerPage())));
    return static_cast<uint32_t>(detail::lcm(
        detail::pagesForAlignedRegion(dataBytesPerLayerPage()),
        detail::pagesForAlignedRegion(scaleBytesPerLayerPage())));
  }

  // Extents hold whole alignment units, between half and one and a half
  // times the allocation target; a unit larger than that is an extent on its
  // own. These are the smallest and the largest sizes.
  [[nodiscard]] constexpr uint32_t minimumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units =
        std::max<uint64_t>(1, (kAllocationExtentTargetBytes / 2 + unit - 1) / unit);
    return static_cast<uint32_t>(units * extentAlignmentPages());
  }
  [[nodiscard]] constexpr uint32_t maximumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units = kAllocationExtentTargetBytes * 3 / 2 / unit;
    return std::max(static_cast<uint32_t>(units * extentAlignmentPages()),
                    minimumExtentPages());
  }

  // The extent size of a pool that holds `pages` pages: all of a pool's
  // extents hold the same number of pages, because kernels find a layer's
  // region from that number. Of the sizes above that fit the pool, the one
  // that leaves the fewest pages over, and the one nearest the target on a
  // tie. Zero when the pool holds fewer pages than the smallest size.
  [[nodiscard]] constexpr uint32_t extentPagesFor(uint64_t pages) const noexcept {
    const uint32_t step = extentAlignmentPages();
    const uint64_t smallest = minimumExtentPages();
    if (!smallest || pages < smallest)
      return 0;
    const uint64_t largest = std::min<uint64_t>(maximumExtentPages(), pages);
    uint64_t best = 0, bestLeft = 0, bestDistance = 0;
    for (uint64_t extent = smallest; extent <= largest; extent += step) {
      const uint64_t bytes = extent * bytesPerModelPage();
      const uint64_t distance = bytes > kAllocationExtentTargetBytes
                                    ? bytes - kAllocationExtentTargetBytes
                                    : kAllocationExtentTargetBytes - bytes;
      if (!best || pages % extent < bestLeft ||
          (pages % extent == bestLeft && distance < bestDistance)) {
        best = extent;
        bestLeft = pages % extent;
        bestDistance = distance;
      }
    }
    return static_cast<uint32_t>(best);
  }

  bool operator==(const Layout &) const = default;
};

} // namespace splash::kv
