# Game 1 Qualitative Review Plan (April 12, 2026)

This document records the Game 1-first qualitative review plan for the current `N = 2` paper results.

## Goal

Make the Game 1 (`N = 2` item allocation) results feel less bland by adding:

- vivid transcript examples
- concrete failure modes
- a more paper-convincing mix of quantitative and qualitative evidence

The broader research question is not just whether stronger models get better payoffs, but whether there are capability-linked shifts in *how* negotiations unfold.

Key motivating hypothesis:

- GPT-5-nano may be disadvantaged against some weaker models not because those models are strategically better overall, but because they can be more stubborn, repetitive, or degenerate in ways that interact badly with the baseline.

## Scope

Phase 1 covers Game 1 only.

If the workflow is useful, it should then be reused for:

- Game 2 diplomacy
- Game 3 cofunding

Game 3 already has structured qualitative metrics. Games 1 and 2 mainly require transcript-plus-proposal review.

## Deliverables

Phase 1 should produce:

- a run-level Game 1 strategy table
- a stratified set of exemplar runs
- new figures/tables showing negotiation styles and failure modes
- a markdown memo with candidate paper claims, caveats, and evidence spans
- paper-ready prose for a Game 1 qualitative subsection

## Method

The review has two layers.

### 1. Deterministic feature extraction

Extract proposal- and outcome-based features from every Game 1 run, including:

- proposal repetition rate
- parse-fallback / proposer-gets-all rate
- all-items-to-self demand rate
- first concession round
- concession size over time
- whether the final accepted deal came from the adversary, the baseline, or neither
- no-consensus / max-round failure
- top-item conflict / overlap structure

These features give broad coverage across the full batch and avoid relying on impressionistic transcript reading.

### 2. LLM-assisted transcript classification

Run Claude Haiku 4.5 over a broad stratified sample of Game 1 runs, with cached JSON outputs, to assign semantic labels that deterministic features cannot capture cleanly.

Initial taxonomy:

- `opening_style`
  - `parser_or_degenerate`
  - `maximalist_anchor`
  - `targeted_anchor`
  - `balanced_tradeoff`
  - `cooperative_exploration`
- `adaptation_style`
  - `rigid_repetition`
  - `incremental_concession`
  - `responsive_tradeoff`
  - `oscillating_or_incoherent`
  - `minimal_evidence`
- `failure_mode`
  - `none`
  - `repetitive_deadlock`
  - `top_item_conflict`
  - `parser_failure`
  - `incompatible_fairness_frame`
  - `late_round_brinkmanship`
  - `other`
- `resolution_driver`
  - `adversary_frame_accepted`
  - `baseline_frame_accepted`
  - `hybrid_compromise`
  - `no_resolution`

The LLM should also assign:

- `stubbornness_scores` for baseline and adversary on a 1-5 scale
- a short `surprising_feature`
- short `evidence` spans

## Sampling Policy

Use both broad coverage and exemplars.

Broad coverage:

- stratify by adversary tier and competition level
- classify multiple runs per stratum where available

Targeted exemplars:

- highest-stubbornness runs
- no-consensus runs
- strongest and weakest payoff-gap runs
- models of special interest such as Llama 1B, 3B, 8B, GPT-5.4 High, and DeepSeek R1-0528

The report should always distinguish:

- broad sampled evidence
- deliberately selected showcase cases

## Planned Outputs

Outputs should be written to:

- `analysis/game1_qualitative_review/`
- `visualization/figures/game1_qualitative_review/`

Expected files:

- `run_features.csv`
- `adversary_summary.csv`
- `label_packets.jsonl`
- `llm_labels.jsonl`
- `labeled_runs.csv`
- `game1_qualitative_review.md`
- Game 1 qualitative figures

## Candidate Figures

Initial figure set:

- adversary stubbornness proxy vs competition and tier
- adversary stubbornness proxy vs consensus outcome
- LLM opening-style prevalence by tier
- LLM failure modes by competition level

Optional later additions:

- accepted-proposal source by capability tier
- first-concession timing vs Elo
- model-by-competition heatmap for no-consensus or stubbornness

## Intended Claims To Test

1. Some low-capability models are harder for GPT-5-nano because they are more rigid or degenerate, not because they bargain more skillfully.
2. There may be capability inflection points where negotiation style changes qualitatively, for example from parser-fallback maximalism to stable package bargaining.
3. Competition level changes not just payoffs, but the frequency of deadlock, repeated anchors, and last-round collapse.
4. The strongest adversaries may succeed with fewer rounds and more coherent trade framing, while some weaker adversaries succeed mainly when the baseline fails to adapt to stubborn behavior.

## Risks / Caveats

- LLM labels are supporting evidence, not ground truth.
- Cherry-picking risk must be called out explicitly.
- Deterministic features should be the main full-batch evidence.
- Transcript labels should be cached and auditable.

## Execution Order

1. Build full-batch deterministic feature extractor for Game 1.
2. Build stratified label packets.
3. Run Claude Haiku 4.5 classification with caching.
4. Generate figures/tables.
5. Draft internal memo and paper-ready prose.
6. Reuse the workflow for Games 2 and 3 if useful.
