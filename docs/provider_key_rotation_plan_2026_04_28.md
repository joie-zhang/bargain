# Provider API Key Rotation Plan

## Goal

Add one shared provider-layer mechanism for rotating API keys across Google,
Anthropic, OpenAI, and OpenRouter. The mechanism should apply to Games 1, 2,
and 3 as well as test-time-compute scaling runs, because they all route through
`run_strong_models_experiment.py` and the shared LLM agent classes.

## Key Storage

Store secrets outside the repository in:

```bash
/home/jz4391/.config/bargain/api_keys.env
```

Set restrictive permissions:

```bash
chmod 600 /home/jz4391/.config/bargain/api_keys.env
```

The file should export distinct variables rather than relying on repeated
`GOOGLE_API_KEY` / `OPENAI_API_KEY` assignments or commented-out `.bashrc`
lines.

Example:

```bash
export LLM_KEY_GROUP_ORDER="LEWIS,JOIE,POLARIS"

export LEWIS_GOOGLE_API_KEY_1="..."
export LEWIS_ANTHROPIC_API_KEY_1="..."
export LEWIS_OPENAI_API_KEY_1="..."
export LEWIS_OPENROUTER_API_KEY_1="..."

export JOIE_GOOGLE_API_KEY_1="..."
export JOIE_GOOGLE_API_KEY_2="..."
export JOIE_ANTHROPIC_API_KEY_1="..."
export JOIE_OPENAI_API_KEY_1="..."
export JOIE_OPENROUTER_API_KEY_1="..."

export POLARIS_GOOGLE_API_KEY_1="..."
```

The code should use `LLM_KEY_GROUP_ORDER` for ordering. Legacy single-key env
vars such as `GOOGLE_API_KEY` should remain supported as a fallback when no
pool variables exist.

## Runtime Behavior

- Use funded/advisor keys first according to `LLM_KEY_GROUP_ORDER`.
- Rotate immediately on quota, daily-rate-limit, provider-rate-limit, or
  insufficient-funds errors.
- If all keys for the relevant provider are exhausted for one agent/model, fail
  the current run immediately.
- Record the exact provider error code/message, the model, and the redacted key
  label that failed.
- Never log key values.
- For transient provider errors such as `500`, `502`, `503`, `504`, connection
  resets, and timeouts, retry the same key with exponential backoff for at most
  five minutes per API call.
- Deterministic bad request/configuration errors should fail fast.

## Failure Reports

Each run should write:

```text
${RUN_DIR}/monitoring/provider_failures.md
```

Also maintain convenience symlinks/copies in:

```text
experiments/results/provider_failure_reports/<run_root_name>.md
experiments/results/provider_failure_reports/latest.md
```

The report should include provider, model, failure kind, key label, count,
latest exact provider message, and the practical next step. For daily quota
exhaustion, the expected fix is to requeue the run after the daily rate limit
resets.

## Implementation Scope

- Add `negotiation/provider_key_rotation.py`.
- Wire it into:
  - `GoogleAgent`
  - `AnthropicAgent`
  - `OpenAIAgent`
  - `OpenRouterAgent`
- Source `/home/jz4391/.config/bargain/api_keys.env` in Slurm launch scripts
  when the file exists.
- Set `LLM_FAILURE_REPORT_PATH` to the run-specific monitoring report.
- Add tests for discovery order, legacy fallback, immediate rotation,
  all-keys-exhausted failure, transient retry budget, report writing, and
  Google concurrency locking.

## Google Concurrency Note

The current `google.generativeai.configure(api_key=...)` call is process-global.
Google calls should hold an async lock across configure plus generation so
phase-concurrent Gemini agents cannot use the wrong key.
