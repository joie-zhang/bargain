# Behavior annotation inter-rater review

## Purpose

- This study measures semantic agreement on one behavior label for one target-authored source record.

- The review interface shows the full compact rollout context that the machine judge received.

- The interface highlights the source record that the reviewer must classify.

- The interface hides the machine result before the reviewer saves a response.

## Current pilot sample

- The reproducible pilot has 480 review items from the TTC annotation datasets.

  - It has 230 machine-positive items and 250 machine-negative items.

  - It has 146 GPT-5 items, 174 Claude items, and 160 Gemini items.

  - It uses deterministic sampling seed `20260814`.

- The selected paper set has 23 behavior labels, but only 20 are turn-level labels.

  - `zero_value_subsidy`, `silent_free_beneficiary`, and `accepted_loss_capitulation` are structural outcome labels.

  - A separate rollout-outcome screen must review the three structural labels.

- The sampler balances by dataset, model family, effort, machine class, and label when eligible strata exist.

- The sampler collapses repeated event rows with the same rollout, source, and tag before it assigns sampling probability.

- The sample is a semantic-validation sample and is not a simple random sample of all label-turn pairs.

  - Use `sampling_weight` for candidate-population estimates.

  - Report class-specific results because positive and negative items have different sampling rates.

## Build the sample

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/build_behavior_irr_sample.py \
  --sample-size 480 \
  --label-set selected23
```

- Use `--label-set full50` to sample from every N=2-eligible turn-level label in the 50-label codebook.

- The N=2 full codebook has 37 eligible turn-level labels.

  - Four labels are structural.

  - Nine turn-level coalition labels require three or four agents.

## Start the review interface

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
PORT=8000 /scratch/gpfs/DANQIC/jz4391/bargain/ui/run_behavior_annotation_review.sh
```

- The server binds to `127.0.0.1` only.

- On a laptop, use this tunnel when the app runs on `della-vis2.princeton.edu`.

```bash
ssh -N -L 8000:127.0.0.1:8000 -J jz4391@della-pli.princeton.edu jz4391@della-vis2.princeton.edu
```

- Open `http://127.0.0.1:8000` in the laptop browser.

## Review policy

- Answer `yes` only when the target performs the named behavior in the focused record.

- Use the rest of the rollout only as context for the focused record.

- Answer `no` when the behavior does not apply to that record.

- Answer `unsure` when the evidence is semantically ambiguous.

- Use `skip` for a technical problem, missing context, or a source-integrity problem.

- Leave machine-result reveal off until the full review is complete.

- Use a stable pseudonym as the reviewer ID.

- Review no more than about 60 items in one sitting to limit fatigue.

## Files and persistence

- The sampling manifest is `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/behavior_annotation_irr_review_20260814/sampling_manifest.jsonl`.

- Sampling provenance is `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/behavior_annotation_irr_review_20260814/sampling_provenance.json`.

- The interface appends each response to `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/behavior_annotation_irr_review_20260814/human_decisions.jsonl`.

  - Each append is locked, flushed, and synced to disk.

  - A changed response adds a new journal record instead of deleting the earlier record.

- The agreement table is `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/behavior_annotation_irr_review_20260814/agreement_ready.csv`.

## Agreement analysis

- Run the analysis after the interface writes the agreement table.

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_behavior_irr.py
```

- Report the per-label confusion table with the human decision as the reference.

  - Report machine sensitivity, specificity, positive predictive value, and negative predictive value.

  - Report human and machine positive prevalence.

- Report raw agreement, Cohen's kappa, and Gwet's AC1.

  - Cohen's kappa is sensitive to low behavior prevalence.

  - Gwet's AC1 and the confusion table make the prevalence effect visible.

- Report the `unsure` and `skip` rates separately.

- Exclude `unsure` and `skip` from the primary binary estimate.

  - Add a sensitivity range that maps all `unsure` responses to `yes` and then to `no`.

- Use rollout-clustered intervals because several sampled label-turn pairs can come from one rollout.

- Keep adjudication separate from agreement measurement.

  - Resolve disagreements only after the independent decisions are frozen.

  - Do not replace independent decisions with the adjudicated result in the reliability calculation.

## More than one reviewer

- Give every reviewer the same frozen overlap sample and keep their decisions independent.

- For two human reviewers, report pairwise raw agreement, Cohen's kappa, and Gwet's AC1.

- For three or more reviewers, add nominal Krippendorff's alpha or a multi-rater agreement coefficient.

- To compare machine judges, run at least two independent judge instances on the same frozen items.

- Compare each machine judge with each human and with the other machine judge.

## Sample-size plan

- Use the 480-item sample as a pilot for prevalence, ambiguity, and reviewer time.

- The pilot is large enough for a pooled diagnostic but not for precise estimates for every label.

- A per-label target of 100 machine-positive and 100 machine-negative decisions gives a worst-case 95 percent binomial half-width of about 10 percentage points within each class.

  - The 20 turn-level selected labels would require up to 4,000 decisions at that target.

  - The 37 N=2-eligible turn-level labels in the full codebook would require up to 7,400 decisions.

- Some rare labels do not have 100 machine-positive candidates.

  - Review every available positive candidate for those labels.

  - Report the exact positive denominator and wider uncertainty.

- Use pilot estimates to simulate power for the final kappa or sensitivity target before fixing the confirmatory sample.

## Data handling

- Treat the compact rollout views as internal research data because they include model private-thinking records.

- Keep the app on loopback and use SSH forwarding.

- Do not expose the app on `0.0.0.0` or a public service.

- Do not put an email address, full name, or other personal data in the reviewer ID or notes.
