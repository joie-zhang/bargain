#!/usr/bin/env python3
"""Prepare the complete nine-seed Gemini TTC corpus for Codex adjudication.

Each of the 648 completed rollouts receives its own manifest/output chunk so
one exhaustive Codex worker can be assigned to exactly one rollout.
"""

from __future__ import annotations

from pathlib import Path

import prepare_ttc_claude_codex_adjudication as pipeline


pipeline.OUTPUT_ROOT = (
    pipeline.PROJECT_ROOT
    / "analysis/ttc_gemini_nine_seed_codex_adjudication_20260809"
)
pipeline.TARGET_FAMILY = "gemini-3-flash"
pipeline.SOURCE_CONFIG_IDS = tuple(range(145, 217))
pipeline.CHUNK_SIZE = 1


def write_gemini_instructions() -> None:
    original = (
        pipeline.ORIGINAL_ROOT / "TTC_LLM_ADJUDICATION_INSTRUCTIONS.md"
    ).read_text(encoding="utf-8")
    addendum = """# Nine-seed Gemini TTC Codex Adjudication

This bundle uses the exact semantic adjudication policy below from the original
216-rollout TTC analysis. It is executed only by Codex collaboration subagents:
do not call OpenRouter, OpenAI/Anthropic/Google APIs, Slurm judge jobs, or any
provider-key-backed script.

Nine-seed identity extension:

- Label all agent-authored behavior for **both target and baseline agents**,
  exactly as in the original analysis.
- Every output row must add `rollout_id`, `seed`, and `source_config_id`, copied
  verbatim from its manifest row.
- Retain the original `config_id` field too; it equals `source_config_id`, but
  only `rollout_id` is globally unique across seeds.
- Each chunk contains exactly one rollout. Read its rollout view completely
  before finalizing the chunk.
- Labels are event-level and source-granular, never rollout-level summaries.
  For a public utterance, emit one row per applicable tag occurrence with the
  exact `log_index`, `round`, `discussion_turn`, `speaker_agent`,
  `speaker_model`, target/baseline role, and a verbatim quote. If the same tag
  recurs on multiple conversation turns, emit a separate row for every turn.
- For private thinking, proposal, voting, and reflection, emit one row per
  applicable tag occurrence with the exact `interaction_index`, `phase`,
  `round`, agent/model/role metadata, and a verbatim quote. Do not invent a
  `discussion_turn` when that phase has none; use null.
- `formal_outcome` is the sole exception to turn-level sourcing. Use it only
  for a structural codebook tag evidenced by the final outcome rather than an
  authored conversation or interaction.
- Public discussion evidence must use `conversation_logs`; use
  `agent_authored_interactions` for private thinking, proposal, voting, and
  reflection, avoiding duplicate discussion rows.
- Write the JSONL and audit files only to the paths specified for your chunk.

---

"""
    pipeline.atomic_write(
        pipeline.OUTPUT_ROOT / "TTC_CODEX_ADJUDICATION_INSTRUCTIONS.md",
        addendum + original,
    )


pipeline.write_instructions = write_gemini_instructions


if __name__ == "__main__":
    pipeline.main()
