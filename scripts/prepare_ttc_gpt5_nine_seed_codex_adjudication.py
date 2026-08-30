#!/usr/bin/env python3
"""Prepare 648 nine-seed GPT-5 TTC rollouts for one-rollout Codex workers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from prepare_ttc_claude_codex_adjudication import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    ORIGINAL_ROOT,
    SEED_ROOT_NAMES,
    SEED_ORDER,
    build_manifest,
    build_view,
    sha256,
    write_json,
    write_jsonl,
)


OUTPUT_ROOT = PROJECT_ROOT / "analysis/ttc_gpt5_nine_seed_codex_adjudication_20260809"
TARGET_FAMILY = "gpt-5"
SOURCE_CONFIG_IDS = tuple(range(1, 73))


WORKER_INSTRUCTIONS = """# GPT-5 TTC one-rollout semantic tagging

You are the sole semantic adjudicator for exactly one N=2 TTC bargaining
rollout. Read the assigned rollout view completely, including every public
conversation log, all private-thinking/proposal/voting/reflection interactions,
and the final outcome. Read the complete 50-tag codebook before labeling.

Tag agent-authored behavior for both the GPT-5 target and the baseline. Use
semantic judgment; the regex strings in the codebook are historical hints,
never classifiers. Include subtle real instances, but do not tag negated,
hypothetical, quoted, or merely mentioned behavior. Multiple tags on one
source are allowed. Repeated evidence in different turns/interactions gets a
separate event. Do not create duplicate events for the same tag and source.
Do not use a script, keyword scan, regex, or templated bulk generator to make
the labeling decisions. Inspect each source in context yourself. High
precision matters: a numeric mention is not automatically utility arithmetic,
a named plan is not automatically formal artifact frameworking, and a stated
preference is not automatically a threat. Rationales must explain the specific
contextual evidence rather than restating the codebook definition.

Source rules:

- Public speech: use `conversation_logs` only, `source_kind=conversation_log`,
  `evidence_type=utterance`, and copy all coordinates from that log.
- Private thinking, proposal, voting, reflection: use
  `agent_authored_interactions`, `source_kind=interaction`, and the matching
  evidence type. The exact mappings are `private_thinking` →
  `private_thinking`, `proposal` → `proposal_reasoning`, `voting` →
  `vote_reasoning`, and `reflection` → `reflection`. For every interaction,
  `speaker_order` and `total_speakers` are null. Do not tag duplicate
  discussion interactions.
- Structural tags (`scope_hint.structural=true`) may use one
  `formal_outcome` event based on `outcome`. Give it `phase=final_outcome`, all
  source coordinates null, and a concise outcome description as the quote.
- Tags whose `min_agents` exceeds 2 are impossible in these N=2 rollouts.
- Quotes for conversation/interaction evidence must be short exact verbatim
  excerpts from the cited source. Never quote prompt/setup text.

Write exactly one JSON object to the assigned output path using this shape:

```json
{
  "rollout_id": "seed_984_config_0001",
  "reviewed": true,
  "events": [
    {
      "tag_code": "conditional_veto_threat",
      "tag_title": "Conditional veto threat",
      "evidence_type": "utterance",
      "source_kind": "conversation_log",
      "phase": "discussion",
      "round": 2,
      "discussion_turn": 1,
      "log_index": 17,
      "interaction_index": null,
      "speaker_agent": "Agent_1",
      "speaker_model": "model name copied via manifest agent_model_map",
      "speaker_elo": 1337,
      "speaker_role": "target",
      "speaker_is_target": true,
      "speaker_is_baseline": false,
      "speaker_order": 1,
      "total_speakers": 2,
      "quote": "short exact excerpt",
      "rationale": "One sentence explaining why the definition applies.",
      "confidence": "high",
      "negation_checked": true
    }
  ]
}
```

Allowed confidence values are `high`, `medium`, and `low`. For a formal outcome
with no attributable speaker, use null for speaker agent/model/Elo/role and
false for both speaker booleans. An empty `events` list is allowed only after
the entire rollout and codebook were genuinely reviewed.

Use `apply_patch` to create only your assigned output file. Do not edit shared
files or call external model APIs. Spawn no agents except the single successor
explicitly required by `HANDOFF_INSTRUCTIONS.md`, if your chain has one.
"""

HANDOFF_INSTRUCTIONS = """# One-rollout worker handoff

This preserves one rollout per subagent while keeping 29 independent worker
chains moving. After you have completely tagged and written your assigned
rollout, run the rollout-scoped validator and fix every error:

`python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/validate_ttc_gpt5_nine_seed_codex_adjudication.py --require-all --no-report --rollout-id <YOUR_ROLLOUT_ID>`

Only after it reports `errors=0`, read `handoff_map.json`. If your rollout ID
maps to another rollout:

1. Spawn exactly one new subagent for that next rollout with `fork_turns=none`,
   model `gpt-5.6-sol`, and reasoning effort `high`.
2. Give it the same one-rollout task: fully read its assignment, shared worker
   instructions, complete codebook, and complete rollout view; write only its
   output with `apply_patch`; validate it until `errors=0`; then perform this
   same handoff procedure. Explicitly include these requirements in its task.
3. Name it `tag_s<seed>_c<four-digit-config>` (for example
   `tag_s526_c0004`).
4. If the spawn temporarily fails because all concurrency slots are occupied,
   wait briefly and retry until the single handoff succeeds. Never spawn more
   than that one successor.

If the map value is null, the chain is complete. Do not do another rollout in
your own agent: every rollout must remain the work of exactly one fresh worker.
"""


def main() -> None:
    views_dir = OUTPUT_ROOT / "rollout_views"
    assignments_dir = OUTPUT_ROOT / "assignments"
    outputs_dir = OUTPUT_ROOT / "subagent_outputs"
    for path in (views_dir, assignments_dir, outputs_dir):
        path.mkdir(parents=True, exist_ok=True)

    codebook_source = ORIGINAL_ROOT / "llm_tag_codebook.json"
    codebook_target = OUTPUT_ROOT / "llm_tag_codebook.json"
    shutil.copyfile(codebook_source, codebook_target)
    (OUTPUT_ROOT / "WORKER_INSTRUCTIONS.md").write_text(
        WORKER_INSTRUCTIONS, encoding="utf-8"
    )
    (OUTPUT_ROOT / "HANDOFF_INSTRUCTIONS.md").write_text(
        HANDOFF_INSTRUCTIONS, encoding="utf-8"
    )

    rows = []
    for seed in SEED_ORDER:
        source_root = RESULTS_ROOT / SEED_ROOT_NAMES[seed]
        for config_id in SOURCE_CONFIG_IDS:
            config_path = source_root / "configs" / f"config_{config_id:04d}.json"
            source_config = json.loads(config_path.read_text(encoding="utf-8"))
            if source_config.get("target_model_family") != TARGET_FAMILY:
                raise RuntimeError(f"Unexpected target family in {config_path}")
            result_dir = Path(source_config["output_dir"])
            result_path = result_dir / "run_1_experiment_results.json"
            interactions_path = result_dir / "run_1_all_interactions.json"
            if not result_path.exists() or not interactions_path.exists():
                raise FileNotFoundError(
                    f"Missing completed source for seed {seed}, config {config_id}"
                )

            rollout_id = f"seed_{seed}_config_{config_id:04d}"
            view_path = views_dir / f"{rollout_id}.json"
            output_path = outputs_dir / f"{rollout_id}.json"
            manifest = build_manifest(
                seed, source_config, result_path, interactions_path, view_path
            )
            write_json(view_path, build_view(manifest))
            assignment = {
                "rollout_id": rollout_id,
                "seed": seed,
                "source_config_id": config_id,
                "instructions_path": str(
                    (OUTPUT_ROOT / "WORKER_INSTRUCTIONS.md").resolve()
                ),
                "codebook_path": str(codebook_target.resolve()),
                "rollout_view_path": str(view_path.resolve()),
                "output_path": str(output_path.resolve()),
            }
            assignment_path = assignments_dir / f"{rollout_id}.json"
            write_json(assignment_path, assignment)
            rows.append({**manifest, "assignment_path": str(assignment_path.resolve()),
                         "output_path": str(output_path.resolve())})

    if len(rows) != 648:
        raise RuntimeError(f"Expected 648 rollouts, prepared {len(rows)}")
    write_jsonl(OUTPUT_ROOT / "all_rollouts_manifest.jsonl", rows)
    # The first 29 workers head balanced chains over the remaining rollouts.
    # Config 0030 is appended to chain zero so the running pool can stay at 29
    # workers and leave one concurrency slot available for child handoffs.
    handoff_map: dict[str, str | None] = {}
    for chain_index, head in enumerate(rows[:29]):
        chain = [head, *rows[30 + chain_index :: 29]]
        for current, following in zip(chain, chain[1:]):
            handoff_map[current["rollout_id"]] = following["rollout_id"]
        if chain_index == 0:
            handoff_map[chain[-1]["rollout_id"]] = rows[29]["rollout_id"]
            handoff_map[rows[29]["rollout_id"]] = None
        else:
            handoff_map[chain[-1]["rollout_id"]] = None
    if set(handoff_map) != {row["rollout_id"] for row in rows}:
        raise RuntimeError("Handoff map does not cover all 648 rollouts")
    write_json(OUTPUT_ROOT / "handoff_map.json", handoff_map)
    inventory = {
        "target_family": TARGET_FAMILY,
        "seeds": list(SEED_ORDER),
        "source_config_ids": [SOURCE_CONFIG_IDS[0], SOURCE_CONFIG_IDS[-1]],
        "rollout_count": len(rows),
        "codebook_tag_count": len(json.loads(codebook_target.read_text())),
        "codebook_sha256": sha256(codebook_target),
        "worker_model": "gpt-5.6-sol",
        "worker_reasoning_effort": "high",
        "one_rollout_per_worker": True,
    }
    write_json(OUTPUT_ROOT / "source_inventory.json", inventory)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
