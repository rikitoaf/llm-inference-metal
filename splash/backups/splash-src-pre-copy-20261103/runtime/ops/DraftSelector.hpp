#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Sampling.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary], or [rows][the shortlist's width] (DraftLogitsColumns).
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
  // With both: draft_select_tree writes every lane's tree (SplashDraftTree), a chain for lanes outside add's tree
  // mask, and the tree lanes' target RoPE positions.
  metal::MetalBuffer draftTrees;
  metal::MetalBuffer targetPositions;
};

// The draft's selector codebooks, which score the edge from a proposal
// position's predecessor candidate to each of its candidates.
struct DraftCodebooks final {
  metal::MetalBuffer predecessor;
  metal::MetalBuffer successor;
};

// The tokens the logits rows hold (DraftLogitsParams). split 0 (the default):
// every token of the vocabulary in order. Otherwise a shortlisted head
// (model::DFlashHeadShortlist): columns [0, split) hold tokens [0, split) and
// the columns from split on the tokens from tail to the vocabulary's end.
struct DraftLogitsColumns final {
  uint32_t split = 0;
  uint32_t tail = 0;
};

// The DFlash draft's proposal policy (draft_select_* in
// metal/kernels/decode/sampling.metal): each lane keeps the
// SPLASH_DRAFT_CANDIDATES most likely draft tokens of every proposal
// position, scores each candidate with its edge from the previous position's
// choice, and walks the SPLASH_DRAFT_PROPOSAL_TOKENS positions greedily or,
// for a sampling lane, drawing at its temperature.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  // treeMask: the lanes whose proposal is a draft tree (draft_select_tree) rather than DFlash's chain. columns:
  // the tokens buffers.logits holds; candidates are token ids either way.
  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
           const DraftCodebooks &codebooks, std::span<const uint32_t> anchors,
           std::span<const SamplingPolicy> policies, uint32_t treeMask = 0,
           DraftLogitsColumns columns = {}) const;

private:
  uint32_t vocabulary_ = 0;
};

} // namespace splash::ops
