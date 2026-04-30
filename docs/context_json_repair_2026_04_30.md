# Context Compaction and JSON Repair Notes

Date: 2026-04-30

## Scope

This patch targets two production failure modes from the full Games 1-3 homogeneous batch:

- Provider context-window failures late in long N-agent runs.
- Structured-output failures caused by lightly malformed JSON that is still semantically recoverable.

The goal is to preserve fail-fast behavior for genuinely invalid runs, while preventing avoidable hard failures from syntax artifacts or oversized public-history prompts.

## Context-Length Findings

The earlier status grid showed 22 context-length failures. The latest final failure table now contains 27 context-classified rows after later retries/backfills:

- By game: Game 1 = 4, Game 3 = 23, Game 2 = 0.
- By N: N=6 = 1, N=8 = 10, N=10 = 16.
- By model category: gpt-4o-mini-2024-07-18 = 11, amazon-nova-micro-v1.0 = 10, claude-sonnet-4-20250514 = 4, control:gpt-5-nano = 2.
- Most Game 3 failures are in later high-history runs at sigma/alpha grid points, especially N=8 and N=10.

The provider errors were not random transport failures. They were hard context-window failures:

- OpenRouter/Amazon examples: requested about 128,958-139,004 tokens against a 128,000-token limit.
- OpenAI examples: messages resulted in about 128,535-135,938 tokens against a 128,000-token limit.
- Anthropic examples: prompt about 202,288-208,232 tokens against a 200,000-token limit.
- GPT-5-nano control examples exceeded the configured 272,000-token cap by about 1,330-1,390 tokens.

## Context Fix Design

The implementation performs a prompt-local preflight before each model call:

1. Resolve the model context limit from the canonical model metadata doc, using the minimum of Arena/OpenRouter metadata and known provider caps.
2. Estimate input tokens for the exact message payload.
3. If estimated input tokens exceed 85% of the model context limit after reserving output tokens, deterministically compact older public rounds.
4. Compact earliest public rounds one at a time until the prompt fits.
5. If all compactable rounds are summarized and the prompt still does not fit, fail before calling the provider with `context_length_exceeded preflight`.

The compaction is prompt-local only. Raw interaction transcripts and run artifacts remain unchanged on disk.

The deterministic summary keeps only public information:

- One capped public discussion line per speaker.
- Formal public proposals or pledges, capped.
- Public vote/tabulation outcomes, capped.

Private thinking and private vote content are not introduced into public summaries.

## JSON Failure Findings

The malformed JSON examples showed these common recoverable causes:

- Comments inside JSON arrays or objects, especially Game 3 proposal contribution arrays.
- Markdown emphasis around numeric literals, such as `**10.0**`.
- Unescaped literal newlines or control characters inside JSON strings.
- Missing commas between JSON object fields.
- Markdown fences or prose surrounding an otherwise usable JSON object.

There are also unrecoverable or schema-invalid cases that should still fail fast:

- Wrong schema, such as Game 2 `"agreement"` vectors submitted for Game 1 allocation proposals.
- Plain text with no JSON object.
- Truncated or semantically incomplete proposal/vote objects after bounded repair attempts.

## JSON Fix Design

The deterministic parser now tries least-to-most invasive local repairs:

1. Direct JSON parse.
2. Fenced or embedded JSON-object extraction.
3. Comment and simple markdown stripping outside strings.
4. Control-character escaping inside strings.
5. Obvious missing-comma insertion.
6. Trailing-comma removal and brace/string balancing.

Schema validation remains game-specific. The repair layer does not convert one game schema into another.

For proposal and batch-vote repair retries, the prompt now includes:

- The original game-specific schema repair instruction.
- The exact invalid raw response, truncated to a bounded size.
- The parser or validator error summary.

This uses the same agent/model for repair, rather than introducing an external fixer model that could contaminate agent behavior.

## Testing So Far

Focused regression suite:

```bash
PYTHONPATH=. pytest -q tests/test_context_compaction.py tests/test_game12_parse_diagnostics.py tests/test_private_thinking_schema.py
```

Result: 22 passed.

Covered cases:

- Context limit resolution from canonical metadata and provider caps.
- Prompt-local compaction without mutating raw `conversation_history`.
- Hard preflight failure if compaction cannot fit the prompt.
- Game 1 comments in proposal and batch vote JSON.
- Game 3 markdown-wrapped numeric values and missing object-field comma.
- Private-thinking missing-comma repair and fallback provenance for unrecoverable JSON.
- Game 1 wrong-schema `"agreement"` vectors still hard-fail rather than being silently accepted.

Broader selected regression suite:

```bash
PYTHONPATH=. pytest -q tests/test_context_compaction.py tests/test_game12_parse_diagnostics.py tests/test_private_thinking_schema.py tests/test_parallel_phases.py tests/test_provider_key_rotation.py tests/test_cofunding_game.py tests/test_cofunding_phases.py tests/test_item_allocation_batch_voting.py tests/test_diplomatic_treaty_batch_voting.py
```

Result after the provider-output retry patch: 142 passed.

## Provider-Output Finding

The first Game 3 de-risk attempt exposed a separate provider-handling bug:
OpenRouter/Amazon returned HTTP success with `finish_reason=stop` but
`content=null`. The key-rotation layer incorrectly classified this as an
API-key-scoped failure, rotated from the Lewis OpenRouter key to the Joie
OpenRouter key, and finally failed with `ProviderKeyExhaustedError`.

That behavior was counterproductive. A provider-success empty message is not
evidence that a key is rate-limited, invalid, or unfunded. The implementation
now treats empty/invalid model output as a short same-key retry, still preserving
fail-fast behavior after the bounded transient retry budget is exhausted. Token
limit truncation cases such as `finish_reason=length` remain deterministic
fail-fast failures.

## Slurm De-risk Results

Results root:
`experiments/results/context_json_repair_derisk_20260430_144145`

Three selected production-shaped configs were submitted:

- `config_0633`: Game 1, homogeneous adversary, N=8, Amazon first, competition 0.75. Finished successfully.
- `config_0805`: Game 1, homogeneous adversary, N=8, GPT-4o-mini first, competition 1.0. Finished successfully in round 6.
- `config_2441`: Game 3, homogeneous adversary, N=8, Amazon last, sigma 0.2, alpha 0.8. Initial attempt hit the provider-output bug above; after patching, the rerun finished successfully in round 1.

Final batch summary for submitted configs:

- Started: 3
- Finished: 3
- Queued/in progress: 0
- Errored: 0

Health checks from the batch summary reported phase concurrency enabled,
supermajority text present, and cosine-vector errors at or near zero for all
three samples. These samples did not grow long enough to require real prompt
compaction, so the context compaction path is covered by unit tests rather than
by a live provider call in this de-risk.

The Game 3 rerun exercised targeted proposal repair: Amazon produced two
over-budget proposal attempts, both were logged as invalid diagnostics, and the
second targeted repair produced a valid proposal that passed supermajority vote.

One artifact caveat: because the failed `config_2441` attempt and the rerun used
the same run directory, old per-run malformed-JSON diagnostics from the failed
attempt can remain beside the successful rerun artifacts. The attempt-specific
Slurm logs preserve the distinction. If this becomes confusing for downstream
analysis, the batch runner should move failed-attempt diagnostics into
attempt-scoped subdirectories before a same-config rerun.
