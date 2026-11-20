#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"

#include <cstdlib>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

// The draft samples its proposal at this fraction of the request's temperature.
// draft_select_dflash stores the q it sampled from, and acceptance and the
// correction draw read that same q (sampling.metal accept_sampled_lane,
// vocabulary_draw), so the output distribution stays exactly the target's; only
// the proposal gets less random. When the draft's scores are flatter than the
// target's, a sharper proposal is accepted more often.
// SPLASH_DRAFT_TEMPERATURE_SCALE in (0, 1]; 1 (the default) is upstream behaviour.
float draftTemperatureScale() {
  static const float scale = [] {
    const char *text = std::getenv("SPLASH_DRAFT_TEMPERATURE_SCALE");
    if (!text || !*text) return 1.0f;
    char *end = nullptr;
    const float value = std::strtof(text, &end);
    return end && *end == '\0' && value > 0.0f && value <= 1.0f ? value : 1.0f;
  }();
  return scale;
}

} // namespace

DraftSelector::DraftSelector(uint32_t vocabulary) : vocabulary_(vocabulary) {
  if (!vocabulary)
    throw std::invalid_argument("invalid draft selector vocabulary");
}

DraftSelectorWorkspace DraftSelector::workspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * SPLASH_DRAFT_CANDIDATES;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kShards * sizeof(uint32_t),
          candidates * (kShards + SPLASH_DRAFT_CANDIDATES) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float),
          candidates * sizeof(float)};
}

void DraftSelector::add(metal::CommandGraph &graph,
                        const DraftSelectorBuffers &buffers,
                        const DraftCodebooks &codebooks,
                        std::span<const uint32_t> anchors,
                        std::span<const SamplingPolicy> policies,
                        uint32_t treeMask, DraftLogitsColumns columns) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  if (columns.split && (columns.split > columns.tail || columns.tail >= vocabulary_))
    throw std::invalid_argument("invalid draft logits shortlist");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  // The top-16 scan reads the shortlist's columns and writes token ids; the codebooks stay indexed by token.
  const DraftLogitsParams logits =
      columns.split ? DraftLogitsParams{columns.split + (vocabulary_ - columns.tail), columns.split,
                                        columns.tail - columns.split}
                    : DraftLogitsParams{vocabulary_, vocabulary_, 0};
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.vocabulary = vocabulary_;
  params.tree_mask = treeMask;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature * draftTemperatureScale();
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            logits, {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add("draft_select_dflash",
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities},
            params, {lanes, 1, 1}, {1, 1, 1});
  if (buffers.draftTrees && buffers.targetPositions)
    graph.add("draft_select_tree",
              {buffers.candidates, buffers.unary, buffers.partialValues,
               buffers.proposedTokens, buffers.proposalProbabilities,
               buffers.draftTrees, buffers.targetPositions},
              params, {lanes, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
