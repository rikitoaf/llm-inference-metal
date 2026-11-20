#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <filesystem>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::model {

struct DFlashDraftRingLayer final {
  // K is [head][ring_position][dimension].
  metal::MetalBuffer keys;
  // V is [head][dimension][ring_position].
  metal::MetalBuffer values;
};

// Draft-owned persistent context, paired with target state in composite caches.
class DFlashDraftRing final {
public:
  DFlashDraftRing(metal::MetalBackend &backend,
                  std::shared_ptr<StateAllocationTracker> tracker,
                  DraftStateLayout layout,
                  std::string_view label);
  ~DFlashDraftRing();
  DFlashDraftRing(const DFlashDraftRing &) = delete;
  DFlashDraftRing &operator=(const DFlashDraftRing &) = delete;

  [[nodiscard]] const std::vector<DFlashDraftRingLayer> &layers() const noexcept {
    return layers_;
  }

private:
  std::shared_ptr<StateAllocationTracker> tracker_;
  std::vector<DFlashDraftRingLayer> layers_;
  uint64_t actualAllocatedBytes_ = 0;
};

struct DFlashDraftLayout final {
  uint32_t layers = 5;
  uint32_t hiddenSize = 5120;
  uint32_t vocabularySize = 248320;
  uint32_t dynamicSize = 1280;
  uint32_t qkvSize = 6144;
  uint32_t attentionSize = 4096;
  uint32_t intermediateSize = 17408;
  uint32_t attentionHeadDimension = 128;
  float rotaryTheta = 10'000'000.0F;
  uint32_t targetHiddenSize = 25600;
  uint32_t selectorRank = 256;
  uint32_t kvHeads = 8;

  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return {layers, kvHeads, attentionHeadDimension};
  }
  [[nodiscard]] constexpr ops::DraftAttentionShape attentionShape() const noexcept {
    return {hiddenSize, dynamicSize, qkvSize, attentionSize,
            attentionSize / attentionHeadDimension, kvHeads,
            attentionHeadDimension};
  }
  // The key and value columns of the fused QKV projection, all the context
  // writers read.
  [[nodiscard]] constexpr uint32_t contextKvSize() const noexcept {
    return qkvSize - attentionSize;
  }

  bool operator==(const DFlashDraftLayout &) const = default;
};

struct DFlashDecodeBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer dynamic;
  metal::MetalBuffer convolved;
  metal::MetalBuffer proposalQkv;
  metal::MetalBuffer attention;
  metal::MetalBuffer projected;
  metal::MetalBuffer residual;
  metal::MetalBuffer intermediate;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer gateScratch;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashContextBuffers final {
  ops::LinearScratch linearScratch{};
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer contextKv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer retainedCounts;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashPrefillSpan final {
  uint32_t compactRow = 0;
  uint32_t rows = 0;
  uint32_t startPosition = 0;
  std::span<const DFlashDraftRingLayer> ring;
};

struct DFlashPrefillBuffers final {
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer contextKv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
};

struct DFlashDraftLayerWeights final {
  ops::NormWeights inputNorm;
  metal::MetalBuffer attentionConvolution;
  ops::Projection attentionDynamic;
  ops::Projection qkvProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Projection outputProjection;
  ops::NormWeights postAttentionNorm;
  metal::MetalBuffer mlpConvolution;
  ops::Projection mlpDynamic;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

struct DFlashDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<DFlashDraftLayerWeights> layers;
  ops::Projection contextProjection;
  ops::NormWeights hiddenNorm;
  ops::NormWeights finalNorm;
  ops::Projection selectorProjection;
  metal::MetalBuffer predecessorCodebook;
  metal::MetalBuffer successorCodebook;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

inline constexpr std::string_view kDFlashLayerMagic = "MDFD0004";

// A Splash package's draft files: layer-<N>.bin and model.bin.
struct PackedDraftFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const DFlashDraftLayout &layout;
  [[nodiscard]] WeightFile layer(uint32_t index) const;
  [[nodiscard]] WeightFile model() const;
};

class DraftCheckpointLoader;

// The files a draft is read from: a package's packed files, or the images
// DraftCheckpointLoader writes from a DFlash2 checkpoint.
using DraftFiles = std::variant<PackedDraftFiles, std::reference_wrapper<DraftCheckpointLoader>>;

[[nodiscard]] DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend, const DraftFiles &files,
                       DFlashDraftLayout layout);

// The draft's vocabulary head over a shortlist of the target head's rows (SPLASH_DRAFT_HEAD_ROWS): tokens
// [0, split) and the tail [tail, vocabulary), zero-copy row views of the GGUF head (ops::blockRows). The draft's
// logits rows then hold split + vocabulary - tail columns, which the selector maps back to token ids
// (ops::DraftLogitsColumns). It only changes what the draft proposes: the target scores every proposal with its
// full head, so the output distribution is the target's either way.
struct DFlashHeadShortlist final {
  // Rows [0, split) into columns [0, split), and rows [tail, vocabulary) into a view of the logits from column
  // split on, both projections of the shortlist's width.
  ops::Projection head;
  ops::Projection tailHead;
  ops::DraftLogitsColumns columns;
  [[nodiscard]] uint32_t width() const noexcept { return head.outputSize; }
  [[nodiscard]] bool covers(uint32_t token) const noexcept {
    return token < columns.split || token >= columns.tail;
  }
};

// The shortlist of `rows` leading tokens (rounded up to whole QUANT_TILE_ROWS tiles) and the tile holding
// `firstAddedToken` to the head's end, or none: rows 0, a head that is not one quantized GGUF tensor of every
// vocabulary row, or a shortlist that would not leave a tile out.
[[nodiscard]] std::optional<DFlashHeadShortlist>
dflashHeadShortlist(const metal::MetalBackend &backend, const ops::Projection &vocabularyProjection, uint32_t rows,
                    uint32_t firstAddedToken);

// Builds the draft layer graph and its proposal selection
// (ops::DraftSelector) from packed buffers and persistent context; the
// target's sampling and acceptance policy remain outside the model.
class DFlashDraft final {
public:
  DFlashDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
               const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;

  // `shortlist` when its columns map onto the draft's vocabulary and its head runs the full head's plan
  // configuration at `lanes` lanes, so that each logit it scores is the full head's bits (and its scratch is no
  // larger); null otherwise, and for no shortlist. A batch passes the result to both addDecode and addSelection.
  [[nodiscard]] const DFlashHeadShortlist *headShortlist(const ops::Projection &vocabularyProjection,
                                                         uint32_t lanes,
                                                         const DFlashHeadShortlist *shortlist) const;
  // With a shortlist, the head scores its tokens only (the full head otherwise).
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths,
                 const DFlashHeadShortlist *shortlist = nullptr) const;
  void addSelection(metal::CommandGraph &graph,
                    const ops::DraftSelectorBuffers &buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    uint32_t treeMask = 0,
                    const DFlashHeadShortlist *shortlist = nullptr) const;
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions) const;

private:
  const DFlashDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::DraftSelector selector_;
  // Each layer's key and value rows of its QKV projection, views of its
  // planes, which the context writers project with.
  std::vector<ops::Projection> contextKvProjections_;
};

} // namespace splash::model
