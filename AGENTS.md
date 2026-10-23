# AGENTS.md

Instructions for AI agents reviewing this repository.

## What this is

A design-phase research project: a paired-view (external + internal camera)
avatar motion-replay pipeline and external-video translator for producing
estimated joint trajectories and first-person video, on hobbyist hardware and a 4-GPU
local box. Not a product. No guarantees, no roadmap promises.

## Before doing anything

1. Read DESIGN.md fully. It is the single source of truth for the thesis.
2. Read docs/first-training-roadmap.md and docs/v0.1-experiment-contract.md.
   Phase 1 is source inspection and one faithful head-and-hand replay; older
   arm-first PDFs and reference recommendations are historical.
3. Read the open questions in README.md. Answer one or critique one; do not
   restate the design back to us.

## Ground rules

- Disagree loudly where you think the design is wrong. A flaw found here
  costs nothing; the same flaw found after buying hardware costs thousands.
- Cite sources for prior-art claims (paper title + year is enough).
- Distinguish manufacturer claims from independently demonstrated results,
  especially for robot hardware specs.
- Budget reality: the operator is early-career. Recommend the cheapest path
  that preserves the science. Never recommend a purchase above ~$500
  without justifying why nothing cheaper answers the question.
- Compute constraints: 4x AMD V620 (ROCm, not CUDA), 150W cap each. Assume
  CUDA-only toolchains need workarounds; flag them early.
- Keep proposals in issues/PRs against DESIGN.md rather than editing the
  design doc directly, unless asked to.

## What we will not debate

- That paired observation-action data is the bottleneck. Argue about how to
  get it, not whether it matters.
- Hardware purchases are deferred; no physical arm is required for Phase 1.
- Never present source inspection, replay, or training gates as completed without
  measured evidence. Translator inputs must not leak tracked poses or joint labels.

## Style

Plain engineering prose. No marketing voice. Numbers over adjectives. If a
claim can't be checked, say it's a guess.
