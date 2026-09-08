# E27 Gemini coalition cases

- Scope covers the three displayed cases, source quotes, votes, payoffs, population variances and their 48.5% contribution to the 25-run Gemini Game 1 mean.
- Current paper references are `fig:gemini_3p1_pro_monoculture` and `sec:appendix_gemini_coalition_case_study`.
- No experiment, figure, source or data file was changed.

## Verified results

| Run | Agents | Seed | Final round | Recomputed discounted payoffs | Population variance |
| --- | --- | --- | --- | --- | --- |
| 107 | 4 | 892954925 | 1 | 83, 72, 81, 20 | 662.5 |
| 108 | 4 | 883562058 | 1 | 49, 73, 55, 0 | 730.6875 |
| 117 | 8 | 1069032829 | 2 | 51.3, 0, 69.3, 59.4, 50.4, 63.9, 0, 52.2 | 662.87109375 |

- Independently read all 25 saved Gemini Game 1 utility vectors and recomputed population variances.
- These cases are the three highest-variance runs and account for 48.47038247997898% of their summed variance.
- Independently recalculated all three case utility vectors from allocated item values and the saved round discount.
- Each selected transcript uses `google/gemini-3.1-pro-preview`, despite the short config alias `gemini-3.1-pro`.
- All three saved vote-integrity records report no synthetic votes and no contamination.
- The seven displayed quote sources are run 107 interaction indices 12 and 32, run 108 indices 13 and 33, and run 117 indices 119, 152 and 192.
- Run 117 displays rounded discounted payoffs, so its variance must not be recomputed from the rounded figure labels.
- Run 117 joins separated source passages without an ellipsis; the raw transcript and extraction script preserve this distinction.

## Verified provenance chain

- The saved 24-model pool feeds `scripts/random_monoculture_control_batch.py`, which excludes one model and uses seed 20260628 to sample and assign models by Elo bands.
- The 25 Gemini Game 1 configurations are configs 0101 through 0125 in the June 28 random-monoculture batch.
- The original submitted array job 10367057 is recorded by the submission JSON and mapped to run 107 task 103 in its status JSON.
- The saved Slurm script calls the random-monoculture run-one adapter, which calls `full.run_config`.
- The attempt log records the exact `run_strong_models_experiment.py` invocation, including the selected model, seed, n, item count, competition and round settings.
- The Slurm script selects OpenRouter proxy transport with external queue path `/home/jz4391/openrouter_proxy`.
- The experiment entry point imports the shared experiment, model factory, phase handler, environment and provider code.
- The final PNG layout imports its base renderer, which contains editable text, vector-like drawing primitives and Source Sans 3 font paths.
- The PPTX deliverables contain the raster figure; they are original requested deliverables, not a substitute for the editable Python sources.
- The prior figure audit's reconstruction script reads raw results and transcript fields before replacing the drawing source's research content.
- Do not delete the copied renderer modules beside that reconstruction script because its local imports use them.

## Needed files and cleanup decisions

- The companion JSON lists 98 concrete existing paths with roles and evidence.
- It includes all 25 result/config pairs, the three transcripts and execution records, sampling/launch provenance, current runtime dependencies, editable figure sources, fonts and presentation deliverables.
- No file is a supported deletion candidate from this result.
- The older qualitative case builder is not required for this figure's numeric calculation, but it serves a separate tag-analysis workflow and is not approved for deletion.
- The shared runtime audit must extend the package-import dependency list before any cleanup.
- Preserve original execution records because saved configurations do not identify a historical Git revision.

## History checked

- /home/jz4391/.codex/sessions/2026/08/22/rollout-2026-08-22T20-01-58-01a02bec-8fb1-70f0-821e-2d44d38d31d2.jsonl:76 bundled search match, user supplies all three cases and 48.5% claim
- /home/jz4391/.codex/sessions/2026/08/16/rollout-2026-08-16T02-38-15-01a0094a-db86-7502-9dfa-887edc433df3.jsonl:758,784,790 original slide request, renderer creation and execution, manually inspected
- /home/jz4391/.codex/sessions/2026/08/17/rollout-2026-08-17T05-11-58-01a00efd-f2bb-7ce3-8dc1-d823598cd9b2.jsonl:1245,1252,1258 final label request, source edit and successful render, manually inspected

- Search covers only locally available Codex records.
- The bundled search completed and found the August 22 caption discussion; original August 16 creation and August 17-18 layout records were then inspected directly.

