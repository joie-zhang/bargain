# OpenRouter Context-Length Failures For The Active Model Roster

Generated: 2026-04-11

This note answers a narrower question than [`openrouter_context_windows_final32_2026_04_06.md`](./openrouter_context_windows_final32_2026_04_06.md): among the models currently listed in [`strong_models_experiment/analysis/active_model_roster.py`](../../strong_models_experiment/analysis/active_model_roster.py), which ones are backed by direct repo evidence for:

1. hitting an OpenRouter context-length error during runs, and
2. being rerun via GPU/cluster backfill.

## Confirmed Models

These are the current active-roster models for which both pieces of evidence exist in-repo.

| Model | Why it is in scope | Direct OpenRouter failure evidence | GPU backfill evidence |
| --- | --- | --- | --- |
| `qwen2.5-72b-instruct` | Current active-roster model. The context audit shows OpenRouter at `32,768` tokens vs native `131K`. | `logs/cluster/cofund_api_6569989_396.out`, `..._397.out`, `..._404.out`, `..._409.out`, `..._412.out` all show `This endpoint's maximum context length is 32768 tokens`. Example: config `396` failed at about `33,280` requested tokens. | `experiments/results/cofunding_20260405_083548_cluster_backfill_pli_20260409/manifest.json` lists source model `qwen2.5-72b-instruct` with backfilled config IDs `396, 397, 404, 408, 409, 412`. The earlier fallback manifest at `experiments/results/cofunding_20260405_083548_cluster_fallback_20260406/manifest.json` lists the same model/config set. |
| `llama-3.2-1b-instruct` | Current active-roster model. The context audit shows OpenRouter at `60,000` tokens vs native `128K`. | `logs/cluster/diplo_6569449_523.out`, `..._525.out`, `..._529.out`, `..._535.out`, plus `logs/cluster/cofund_api_6569989_528.out`, all show `This endpoint's maximum context length is 60000 tokens`. Examples: config `523` failed at about `60,647` requested tokens; config `529` failed at about `64,793`; config `528` failed at about `61,397`. | `experiments/results/cofunding_20260405_083548_cluster_backfill_pli_20260409/manifest.json` lists source model `llama-3.2-1b-instruct` with backfilled config IDs `522, 523, 528`. The earlier fallback manifest at `experiments/results/cofunding_20260405_083548_cluster_fallback_20260406/manifest.json` lists the same model/config set. |

## Lower OpenRouter Context, But Not Confirmed Here As A Backfill Trigger

The broader context audit shows several other active-roster models with materially smaller OpenRouter windows than their native/full context:

- `deepseek-r1`: `64,000` on OpenRouter vs `128K` native
- `llama-3.1-8b-instruct`: `16,384` on OpenRouter vs `128K` native
- `llama-3.2-3b-instruct`: `80,000` on OpenRouter vs `128K` native

For these, this audit did **not** find the same level of direct evidence as above: a matching OpenRouter context-length failure log plus a clearly corresponding GPU backfill manifest entry for the same failure.

### Note On `llama-3.1-8b-instruct`

`llama-3.1-8b-instruct` does appear in the earlier isolated fallback manifest at `experiments/results/cofunding_20260405_083548_cluster_fallback_20260406/manifest.json`, but that script selected configs with missing results generally, not only context-limit failures. In the logs checked here, I found successful OpenRouter runs for this model as well, for example:

- `logs/cluster/diplo_6569449_498.out`
- `logs/cluster/diplo_6569449_501.out`
- `logs/cluster/cofund_api_6612746_493.out`
- `logs/cluster/cofund_api_6612746_496.out`

So I am not labeling `llama-3.1-8b-instruct` as a confirmed OpenRouter context-length backfill case from the available evidence.

## Practical Short List

If you want the short answer for the current roster, the confirmed OpenRouter context-length problem models that were redone via GPU backfill are:

- `qwen2.5-72b-instruct`
- `llama-3.2-1b-instruct`

## Sources Consulted

- `strong_models_experiment/analysis/active_model_roster.py`
- `docs/guides/openrouter_context_windows_final32_2026_04_06.md`
- `logs/cluster/cofund_api_6569989_396.out`
- `logs/cluster/cofund_api_6569989_397.out`
- `logs/cluster/cofund_api_6569989_404.out`
- `logs/cluster/cofund_api_6569989_409.out`
- `logs/cluster/cofund_api_6569989_412.out`
- `logs/cluster/cofund_api_6569989_528.out`
- `logs/cluster/diplo_6569449_523.out`
- `logs/cluster/diplo_6569449_525.out`
- `logs/cluster/diplo_6569449_529.out`
- `logs/cluster/diplo_6569449_535.out`
- `experiments/results/cofunding_20260405_083548_cluster_fallback_20260406/manifest.json`
- `experiments/results/cofunding_20260405_083548_cluster_backfill_pli_20260409/manifest.json`
