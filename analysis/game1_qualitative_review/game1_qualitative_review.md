# Game 1 Qualitative Review

## Coverage

- Runs analyzed: 840
- Unique adversary models: 30
- No-consensus runs: 21
- LLM-labeled runs: 127

## Deterministic Signals

Lowest consensus-rate adversaries at competition 1.0:
- `llama-3.2-1b-instruct`: consensus rate 0.50
- `llama-3.2-3b-instruct`: consensus rate 0.75
- `gpt-4o-2024-05-13`: consensus rate 0.75
- `o3-mini-high`: consensus rate 0.75
- `qwen2.5-72b-instruct`: consensus rate 0.75
- `amazon-nova-micro-v1.0`: consensus rate 1.00
- `claude-3-haiku-20240307`: consensus rate 1.00
- `amazon-nova-pro-v1.0`: consensus rate 1.00

Highest adversary stubbornness proxy at competition 1.0:
- `llama-3.2-3b-instruct`: proxy 0.81
- `llama-3.2-1b-instruct`: proxy 0.50
- `gemini-2.5-pro`: proxy 0.34
- `claude-3-haiku-20240307`: proxy 0.33
- `qwen3-max-preview`: proxy 0.31
- `gpt-4o-2024-05-13`: proxy 0.28
- `gpt-5.4-high`: proxy 0.28
- `deepseek-r1`: proxy 0.25

## Exemplar Candidates

- `llama-3.2-3b-instruct` at comp `0.9` | consensus=False | rounds=10 | stubbornness=1.00 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/strong_first/comp_0.9/turns_1/run_2/experiment_results.json
- `llama-3.2-3b-instruct` at comp `0.9` | consensus=False | rounds=10 | stubbornness=1.00 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/weak_first/comp_0.9/turns_1/run_1/experiment_results.json
- `llama-3.2-3b-instruct` at comp `1.0` | consensus=False | rounds=10 | stubbornness=1.00 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/weak_first/comp_1.0/turns_1/run_1/experiment_results.json
- `llama-3.2-1b-instruct` at comp `1.0` | consensus=False | rounds=10 | stubbornness=0.72 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/comp_1.0/turns_2/run_2/experiment_results.json
- `llama-3.2-1b-instruct` at comp `0.75` | consensus=False | rounds=10 | stubbornness=0.62 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/comp_0.75/turns_1/run_2/experiment_results.json
- `llama-3.2-1b-instruct` at comp `0.95` | consensus=False | rounds=10 | stubbornness=0.54 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-1b-instruct/weak_first/comp_0.95/turns_1/run_1/experiment_results.json
- `llama-3.2-1b-instruct` at comp `0.95` | consensus=False | rounds=10 | stubbornness=0.52 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/comp_0.95/turns_1/run_2/experiment_results.json
- `llama-3.2-1b-instruct` at comp `1.0` | consensus=False | rounds=10 | stubbornness=0.52 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/comp_1.0/turns_1/run_2/experiment_results.json
- `gpt-4.1-nano-2025-04-14` at comp `0.95` | consensus=False | rounds=10 | stubbornness=0.50 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-4.1-nano-2025-04-14/weak_first/comp_0.95/turns_2/run_1/experiment_results.json
- `qwen2.5-72b-instruct` at comp `1.0` | consensus=False | rounds=10 | stubbornness=0.50 | file=/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_qwen2.5-72b-instruct/strong_first/comp_1.0/turns_1/run_2/experiment_results.json

## LLM Label Snapshot

These counts come from the labeled subset, not the full 890-run batch.

opening_style:
- `balanced_tradeoff`: 49
- `targeted_anchor`: 41
- `cooperative_exploration`: 19
- `maximalist_anchor`: 13
- `parser_or_degenerate`: 5

adaptation_style:
- `responsive_tradeoff`: 56
- `minimal_evidence`: 40
- `rigid_repetition`: 30
- `oscillating_or_incoherent`: 1

failure_mode:
- `none`: 78
- `parser_failure`: 24
- `repetitive_deadlock`: 19
- `top_item_conflict`: 3
- `late_round_brinkmanship`: 2
- `incompatible_fairness_frame`: 1

resolution_driver:
- `hybrid_compromise`: 65
- `baseline_frame_accepted`: 26
- `no_resolution`: 20
- `adversary_frame_accepted`: 16

relative_stubbornness:
- `neither`: 65
- `adversary_more_stubborn`: 43
- `both_stubborn`: 10
- `baseline_more_stubborn`: 9

## Companion Docs

- `game1_elo_paper_ready_paragraphs.md`
- `game1_label_exemplar_index.md`
- `label_exemplars.csv`
- `game1_payoff_mechanism_table.md`
- `game1_payoff_vs_elo_subsection.md`
- `why_payoff_rises_with_elo_notes.md`

