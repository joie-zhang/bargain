# Game 1 Initial Findings (April 12, 2026)

This note summarizes the first broad-coverage qualitative pass for Game 1 item allocation.

Data sources:

- full deterministic feature extraction over 890 completed runs
- Claude Haiku 4.5 labels for 118 stratified/exemplar runs

## Main Takeaways

1. The strongest evidence for a qualitative capability inflection currently comes from the bottom of the model ladder.

- In the full batch at competition `1.0`, the lowest consensus rates are:
  - `llama-3.2-1b-instruct`: `0.50`
  - `llama-3.2-3b-instruct`: `0.75`
  - `o3-mini-high`: `0.75`
  - `qwen2.5-72b-instruct`: `0.75`
  - `gpt-4o-2024-05-13`: `0.75`
- By contrast, `llama-3.1-8b-instruct` reaches full consensus in the sampled high-competition slice and does **not** look like a simple continuation of the 1B/3B pattern.

2. The 1B/3B Llama models look qualitatively different from the 8B model.

- Full-batch stubbornness proxy at competition `1.0`:
  - `llama-3.2-3b-instruct`: `0.81`
  - `llama-3.2-1b-instruct`: `0.50`
  - `llama-3.1-8b-instruct`: not among the worst cases
- In the labeled subset, 1B/3B runs are repeatedly tagged as:
  - `parser_failure`
  - `repetitive_deadlock`
  - `adversary_more_stubborn`
- The 8B runs are much more likely to look like ordinary negotiation rather than pathological repetition.

3. “Weak models hurt GPT-5-nano because they are stubborn” is partly true, but not universally true.

- In the 118 labeled runs:
  - `adversary_more_stubborn`: `25`
  - `baseline_more_stubborn`: `12`
  - `both_stubborn`: `9`
  - `neither`: `72`
- So stubbornness matters, but it is not the dominant explanation for most runs.
- The current picture is:
  - most negotiations still look normal
  - the interesting failures are concentrated in a small but real subset

4. Most labeled runs are not failures; they are straightforwardly successful bargaining.

- Label distribution:
  - `failure_mode = none`: `88 / 118`
  - `repetitive_deadlock`: `16 / 118`
  - `parser_failure`: `7 / 118`
  - `top_item_conflict`: `5 / 118`
- This means the paper story should not overstate dysfunction.
- A better framing is:
  - most pairings converge via recognizable tradeoff reasoning
  - a minority of pairings, concentrated in certain low-capability or awkward settings, fail in qualitatively revealing ways

5. Stronger adversaries often look more coherent rather than merely more stubborn.

- In the labeled sample, stronger models such as `gpt-5.4-high` and `deepseek-r1-0528` are mostly tagged with:
  - `failure_mode = none`
  - `adaptation_style = responsive_tradeoff`
  - `opening_style = targeted_anchor` or `balanced_tradeoff`
- This supports a narrative where strong models win by coherent package bargaining, not just by refusing to move.

## Candidate Paper Claims

These are plausible claims to explore next, not finalized claims yet.

1. There is a low-capability regime where negotiation failures are driven by degenerate or repetitive proposal behavior rather than strategically sophisticated bargaining.
2. The transition from `llama-3.2-3b` / `llama-3.2-1b` to `llama-3.1-8b` may already show a qualitative shift from parser-like stubbornness to ordinary trade negotiation.
3. Capability appears to improve *strategic coherence* more reliably than it increases raw stubbornness.
4. High competition exposes a specific family of failures: repeated anchors, parser fallback, and unresolved deadlock around contested high-value items.

## Good Exemplar Directions

The strongest current case-study candidates are:

- `llama-3.2-3b-instruct` high-competition no-consensus runs
- `llama-3.2-1b-instruct` high-competition deadlocks
- one or two `gpt-5.4-high` or `deepseek-r1-0528` runs that show fast, coherent package bargaining
- one “baseline more stubborn” counterexample, to avoid a one-sided narrative

## What To Do Next

1. Read 8-12 exemplar transcripts manually and pull exact paper quotes.
2. Tighten the stubbornness proxy so parser fallback and strategic rigidity are separated more explicitly.
3. Expand the label cache from 118 runs toward the full selected packet set if needed.
4. Draft a short Game 1 subsection that contrasts:
   - normal successful bargaining
   - coherent strong-model anchoring
   - weak-model repetitive failure modes
