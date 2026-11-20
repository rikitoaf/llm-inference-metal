#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The uniforms one lane draws a decode cycle with, in [0, 1): the first
// token after a prompt draws SPLASH_UNIFORM_INITIAL; the draft's sampled
// proposal at position p draws SPLASH_UNIFORM_PROPOSALS + p; acceptance tests
// draft token p against SPLASH_UNIFORM_ACCEPTANCE + p; and a sampled verify
// row draws its correction, or the bonus token after the whole draft, with
// SPLASH_UNIFORM_CORRECTION. Lane l's uniforms start at
// l * SPLASH_SAMPLING_UNIFORMS.
#define SPLASH_UNIFORM_INITIAL 0u
#define SPLASH_UNIFORM_PROPOSALS 1u
#define SPLASH_UNIFORM_ACCEPTANCE                                          \
  (SPLASH_UNIFORM_PROPOSALS + SPLASH_DRAFT_PROPOSAL_TOKENS)
#define SPLASH_UNIFORM_CORRECTION                                          \
  (SPLASH_UNIFORM_ACCEPTANCE + SPLASH_DRAFT_PROPOSAL_TOKENS)
#define SPLASH_SAMPLING_UNIFORMS (SPLASH_UNIFORM_CORRECTION + 1u)

// One target-policy dispatch over lanes of SPLASH_TARGET_VERIFY_ROWS logits
// rows, SPLASH_TARGET_VERIFY_ROWS + 1 constraint-mask rows and
// SPLASH_SAMPLING_UNIFORMS uniforms each. Selected row s is row s % rows of
// lane s / rows: its logits row is lane * SPLASH_TARGET_VERIFY_ROWS +
// logits_row + s % rows, its mask row lane * (SPLASH_TARGET_VERIFY_ROWS + 1) +
// mask_row + s % rows, and its draw takes uniform
// lane * SPLASH_SAMPLING_UNIFORMS + uniform; workspaces and output tokens are
// indexed by s. Rows below drafted_rows follow draft token s % rows (verify
// input row s % rows + 1).
struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t rows;
  uint32_t logits_row;
  uint32_t mask_row;
  uint32_t uniform;
  uint32_t drafted_rows;
  uint32_t top_k[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  float top_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  float min_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  // Lanes that sample; the others take the argmax.
  uint32_t sampling_mask;
  uint32_t constrained_mask;
  // Lanes that ignore end-of-sequence: they never select a stop token.
  uint32_t exclude_stop_mask;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  // Lanes whose verify rows are a draft tree (SplashDraftTree): a sampled row at depth d draws with uniform
  // SPLASH_UNIFORM_ACCEPTANCE + d (the deepest, d = SPLASH_DRAFT_PROPOSAL_TOKENS, with SPLASH_UNIFORM_CORRECTION),
  // so the draws along an accepted path are independent; tree acceptance reads none of these uniforms itself.
  uint32_t tree_mask;
};

static_assert(sizeof(TargetSamplingParams) == 116,
              "Target sampling parameters are 116 bytes on both sides");

// One shard's share of a sampled row's softmax denominator: the largest
// logit it admits, the sum of exp((logit - maximum) / temperature) over its
// admitted tokens, and how many it admits.
struct TargetShardMass {
  float maximum;
  float sum;
  uint32_t admitted;
};

static_assert(sizeof(TargetShardMass) == 12,
              "Target shard masses are 12 bytes on both sides");

// A sampled row's selection over the whole vocabulary. The search records
// the row's largest admitted logit and where its min-p/top-k/top-p
// distribution ends in the order of the logits (the key and id of its last
// token, metal/kernels/decode/sampling.metal). The draw writes a drafted
// row's target probability of its draft token; the token it draws goes to
// the output tokens: for a verify row with a draft token, the correction
// acceptance takes if it rejects that token (a draw from the residual
// distribution); otherwise a draw from the row's distribution.
struct TargetVocabularyRow {
  float maximum;
  uint32_t end_key;
  uint32_t end_last;
  float draft_probability;
};

static_assert(sizeof(TargetVocabularyRow) == 16,
              "Target vocabulary rows are 16 bytes on both sides");

// One range of the vocabulary in such a row's draw, which one simdgroup of
// the row's groups sums: the kept weight of its tokens other than the
// draft's candidates, and one past the last of those with weight.
struct TargetVocabularyRange {
  float rest;
  uint32_t after;
};

static_assert(sizeof(TargetVocabularyRange) == 8,
              "Target vocabulary ranges are 8 bytes on both sides");

// A penalized request's word for each vocabulary token, in its state lane's
// row of the penalty table (ops::Sampling::rebuildPenaltyWords): the
// prompt bit marks a prompt token, and the count is how often the target
// selected it.
#define SPLASH_PENALTY_PROMPT_BIT 0x80000000u
#define SPLASH_PENALTY_COUNT_MASK 0x7fffffffu

// The penalized lanes of one penalty dispatch. Each entry names the lane of
// its logits and the penalty table row it reads; rows penalizes that many
// rows of the lane's SPLASH_TARGET_VERIFY_ROWS, from row_offset.
// repetition_inverse is 1 / repetition, saturated to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t rows;
  uint32_t row_offset;
  uint32_t entries;
  uint32_t logits_lane[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[SPLASH_MAXIMUM_BATCH_WIDTH];
  float presence[SPLASH_MAXIMUM_BATCH_WIDTH];
  float frequency[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SamplingPenaltyParams) == 112,
              "Sampling penalty parameters are 112 bytes on both sides");

// The batched selector and acceptance kernels' grids cover exactly the
// dispatch's lanes: per-lane arrays hold those lanes, and entries past them
// are zero and unread.
struct SelectorBatchParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t sampling_mask;
  uint32_t vocabulary;
  // Lanes whose proposal is a draft tree (draft_select_tree); the others keep DFlash's chain.
  uint32_t tree_mask;
};

static_assert(sizeof(SelectorBatchParams) == 48,
              "Draft selector parameters are 48 bytes on both sides");

// The draft logits rows draft_select_top16_sharded scans: `width` columns a
// row, column c holding token c below `split` and token c + `offset` from it.
// The full vocabulary head is width = split = the vocabulary and offset 0; a
// shortlisted draft head (SPLASH_DRAFT_HEAD_ROWS, model::DFlashHeadShortlist)
// scores tokens [0, split) and [split + offset, vocabulary).
struct DraftLogitsParams {
  uint32_t width;
  uint32_t split;
  uint32_t offset;
};

static_assert(sizeof(DraftLogitsParams) == 12,
              "Draft logits parameters are 12 bytes on both sides");

// One lane's draft tree (draft_select_tree). Verify row r holds a node: row 0 the anchor, rows 1..7 the proposals
// in the order they were chosen, so a parent's row is below its children's. parent[r] is its parent's row (row 0:
// none, 0xffffffff), depth[r] its distance from the anchor (its RoPE position is the anchor's plus depth[r]) and
// bit j of ancestors[r] is set when row j is r or an ancestor of r: the verify rows r attends to. Acceptance writes
// path[d], the row of the accepted node at depth d (path[0] = 0), for the commits. A lane without a tree carries the
// chain: parent r - 1, depth r, ancestors 2^(r+1) - 1, path r.
struct SplashDraftTree {
  uint32_t parent[SPLASH_TARGET_VERIFY_ROWS];
  uint32_t depth[SPLASH_TARGET_VERIFY_ROWS];
  uint32_t ancestors[SPLASH_TARGET_VERIFY_ROWS];
  uint32_t path[SPLASH_TARGET_VERIFY_ROWS];
};

static_assert(sizeof(SplashDraftTree) == 16 * SPLASH_TARGET_VERIFY_ROWS,
              "A draft tree is four words per verify row on both sides");

struct AcceptBatchParams {
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t sampling_mask;
};

static_assert(sizeof(AcceptBatchParams) == 28,
              "Batched acceptance parameters are 28 bytes on both sides");
