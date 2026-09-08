**Question** Can every current AI Wild figure be traced to its source and recreated?

**Short answer** All 30 figures and 32 assets were traced and independently recreated. 29 figures match every compared pixel; 19 also match file bytes.

- Each figure had its own sub-agent.
- The parent independently compared all 32 assets with frozen references.
- Figure 1 remains approximate because its original editable drawing and icons are missing.
- Exact image reproduction does not establish that the underlying analysis is correct.
- The findings below identify data and interpretation problems in the reproduced figures.

**Which paper and version were checked?**

- The source is [/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/icml_aiwild_template/icml_aiwild_2026.tex](</scratch/gpfs/DANQIC/jz4391/bargain/overleaf/icml_aiwild_template/icml_aiwild_2026.tex>).
- The user called this the ICLR AI Wild paper, but the captured entry file still uses the ICML style.
- The audit snapshot was taken at `2026-09-06T03:21:17.297437+00:00`.
- Repository commit: `444fcf9c368a60dd40a0f5f7aa8b4033552312bf`.
- Paper repository commit: `4b62b05d538b2525ba44305c90e384dd0c0a32a7`.
- [The manifest](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/manifest.json>) records each asset, source label, and reference hash.
- [The reference PDF](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/paper_reference.pdf>) preserves the captured paper.
- History searches cover locally available session files and prompt history only.
- The local inventory includes 42 session identifiers with prompt history but no full session file.
- Prior audits were used as search leads and checked against source files and fresh runs.

**How can the results be reviewed?**

- Open [the portable comparison gallery](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/gallery.html>) to compare all figures.
- The gallery includes reference and recreated previews inside the HTML file.
- Each figure below links to full-size outputs, raw-data records, history citations, scripts, and commands.
- Existing source code was reused where available, and all plot panels were redrawn from data or transcript text.
- The diagram uses newly drawn shapes and recovered font programs from the supplied PDF.
- No existing plot pixels or PDF page drawing objects were reused in a reconstruction.

**What does the paper study?**

- Agents negotiate over item allocation, diplomatic agreements, and public-project funding.
  - They discuss, think privately, propose an outcome, vote, and reflect after rejection.
  - A two-thirds supermajority accepts a proposal; delay reduces payoff.
- The experiments vary model capability, group size, competition, and reasoning effort.
- Figures 2–5 examine two-player payoff, fairness, annotated behaviour, and reasoning effort.
- Figures 6–11 examine larger groups, unequal payoffs, coalitions, and weak-agent teams.
- Figures 12–30 check alternative capability measures, metrics, game settings, and a matched coalition experiment.
- The plotted associations describe these saved experiments; they do not by themselves establish causal effects.

**Which findings need attention?**

- **Synthetic actions in 72 multi-agent runs.** The full scan found explicit synthetic proposal or vote records in 72 of 2,900 included runs.
  - These comprise 62 heterogeneous runs and 10 homogeneous runs.
  - Sixty-four runs contain 86 synthetic proposal events, and 11 runs contain 60 synthetic votes.
  - Three runs contain both types of action.
  - The 1,300 homogeneous adversary runs have no such structured records in this scan.
  - Affected records are included in the data for Figures 3, 6, 7, 23, 27, 28, and 29.
  - An affected trajectory does not necessarily mean that its final agreement was synthetic.
  - These records were preserved to reproduce the figures and must not be treated as clean observations of model actions.
  - [All 72 affected runs with raw record locations](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_23/outputs/contaminated_runs.json>).
  - [Heterogeneous raw verification](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_27/integrity_findings.json>).
  - [Homogeneous adversary integrity check](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_14/integrity_summary.json>).
- **Reversed role attribution in 11 bilateral runs.** The analysis assigns the two agents to roles that conflict with their recorded runtime model identities.
  - All 11 cases are in Game 1 and affect the bilateral data used by several figures.
  - Submitted configuration order and runtime model identity disagree.
  - The recreation preserves the historical assignment rather than correcting the plotted payoff.
  - [Eleven verified runtime role discrepancies](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_13/runtime_role_discrepancies.json>).
- **Gemini model identity differs from its plotted label.** All 50 bilateral runs labelled Gemini 3.1 Pro record gemini-3-pro-preview in their saved interactions.
  - An April 30 alias change later mapped the older configuration name to Gemini 3.1 Pro.
  - The saved execution records do not establish that Gemini 3.1 Pro ran in these experiments.
  - This finding concerns the bilateral cohort and does not imply that the separate matched Gemini experiment used the same model.
  - [Runtime verification for all 50 Gemini runs](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/data/gemini_identity_verification.json>).
  - [Model identity and alias history](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/report.md>).
- **Incomplete bilateral outcomes enter the analysis.** Fourteen included records have no saved adversary utility and receive the historical zero value.
  - Two Game 3 records instantiate only the baseline agent.
  - Five accepted Game 1 allocations omit an item and violate the full-partition rule.
  - Reproduction retains these historical inputs and does not establish that the runs completed correctly.
  - [Missing utilities](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/data/missing_utilities.json>).
  - [Invalid Game 1 allocations](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/data/invalid_game1_partitions.json>).
  - [Single-agent execution evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_22/report.md>).
- **Six fairness failures appear as zero distance.** Figure 18 treats six Game 1 failures with empty saved preferences as zero fairness distance and zero residuals.
  - The complete preferences remain available in the original setup transcripts.
  - Using those preferences gives benchmark distances from 70.72 to 141.42 for these failures.
  - A separate diagnostic measures the resulting slope changes without changing the recreated images.
  - [Preference recovery and diagnostic calculations](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_18/empty_preference_diagnostic.json>).
- **The diagram remains an approximate recreation.** Figure 1 has no local editable Google drawing or original icon source.
  - The supplied final PDF export was traced through the original user request and matching file hash.
  - The reconstruction has editable text and new geometric shapes, but small text, icon, and layout differences remain.
  - The illustrative dialogue has no verified link to an observed experiment transcript.
  - [Diagram provenance and remaining differences](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_01/report.md>).
- **The Figure 3 caption has a different Game 3 crossing.** The plotted line crosses zero at Elo 1426.32, while the captured caption states approximately 1454.
  - All five panels can still be reproduced exactly from the raw records with the original benchmark algorithm and matching numerical settings.
  - [Figure 3 numerical and caption audit](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_03/report.md>).
- **Coalition denominators and labels need correction.** Figure 9 uses 12/80, or 15%, for homogeneous Game 1 while the captured prose states 12/100, or 12%.
  - The historical adapter also subtracts 40 already-excluded Nano appearances from one Elo group.
  - The correct denominator is 675 rather than 635, but its zero numerator makes the plotted rate unchanged.
  - The original 15 audit sessions used gpt-5.6-sol with high effort, which differs from the stated xhigh setting.
  - Candidate screening and manual positive-case review do not establish a blind prevalence estimate over every transcript.
  - [Coalition counts, denominator checks, and original judge provenance](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_09/report.md>).
- **The matched coalition bars use different counting rules.** Figure 30 reproduces the saved counts, but the Gemini and GPT-5.4 counts do not use fully matching definitions.
  - Gemini proposed coalitions include one private attempt without a formal proposal.
  - Gemini selected coalitions exclude a harmful selected proposal that passed above the minimum vote threshold.
  - A common definition is needed before directly comparing the two frequencies.
  - [Matched cohort, proposal, vote, and classification audit](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_30/report.md>).
- **The Nano control line is a display convention.** The gray zero line in Figure 10 was explicitly requested for display and is not the measured optimal-allocation rate of the uncoordinated control.
  - The observed control reaches the mathematical payoff ceiling in 4 of 20 runs at each group size.
  - The blue two-agent point is also an untreated single-agent group.
  - [Team and control raw results with original display request](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_10/report.md>).
- **Qualitative labels retain judge and source limits.** Figure 4 uses saved annotations whose original worker settings differ from the paper description.
  - The original workers used gpt-5.5 with high effort, and later repair workers used gpt-5.6-sol with high effort.
  - The audit verifies saved annotation rows and source evidence but does not claim a new blind classification of every transcript.
  - The transcript figures preserve historical wording, including a model arithmetic error in Figure 11.
  - [Annotation provenance and quote checks](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_04/report.md>).
  - [Nano transcript and arithmetic verification](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_11/report.md>).
- **Reasoning experiment token limits varied.** The 2,160 Figure 5 records include 2,088 token limits of 10,500, 71 of 16,384, and one of 65,536.
  - Sixty-seven of the 71 larger-limit cases have separate recorded recovery configurations.
  - Four seed-42 cases lack a separate retained recovery configuration.
  - The recreated estimates preserve the actual saved runs.
  - [Raw reasoning experiment and recovery provenance](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_05/report.md>).

**How was an exact match checked?**

- PNG files were compared by SHA-256 and by decoded RGBA pixels at their original dimensions.
- PDFs were rendered with the same Ghostscript settings at 300 DPI, without resizing or alignment.
- Each PDF asset contains one page, and that complete page was compared.
- PDF file bytes can differ because the creation timestamp changes while the drawn content remains identical.
- A visual match has stated nonzero pixel differences; it is not counted as exact.
- [Parent comparison results](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks>) contain dimensions, hashes, differences, and side-by-side previews.

**What is the result for each figure?**

| Figure | Question | Reproduction |
|---|---|---|
| 1 | How does each negotiation game proceed? | Mismatch |
| 2 | How does model capability relate to each role's payoff? | Exact bytes and pixels |
| 3 | How do payoffs compare with the fairness benchmarks? | Exact bytes and pixels |
| 4 | Which annotated behaviours relate to payoff and capability? | Exact pixels |
| 5 | How does additional reasoning effort change payoff? | Exact pixels |
| 6 | Does capability predict payoff in larger Game 1 groups? | Exact bytes and pixels |
| 7 | How does payoff variance differ between group types? | Exact pixels |
| 8 | What do the Gemini coalition transcripts show? | Exact bytes and pixels |
| 9 | Where do exclusionary coalitions occur? | Exact pixels |
| 10 | How does a coordinated Nano team compare with an adversary model? | Exact pixels |
| 11 | What do the Nano team transcripts show? | Exact bytes and pixels |
| 12 | Does the Elo snapshot date change the bilateral trend? | Exact pixels |
| 13 | Does the trend hold for other capability measures? | Exact pixels |
| 14 | How do role payoffs and variance change in homogeneous adversary runs? | Exact bytes and pixels |
| 15 | How does inequality change when the adversary model is included? | Exact bytes and pixels |
| 16 | How does bilateral adversary payoff vary with competition? | Exact bytes and pixels |
| 17 | How does total welfare vary with competition? | Exact bytes and pixels |
| 18 | How far are outcomes from the fairness benchmarks? | Exact bytes and pixels |
| 19 | Does the bilateral trend also appear with a Llama baseline? | Exact bytes and pixels |
| 20 | How do fixed-baseline games compare with heterogeneous pairs? | Exact bytes and pixels |
| 21 | How many rounds do successful negotiations require? | Exact bytes and pixels |
| 22 | How does model order relate to negotiation outcomes? | Exact bytes and pixels |
| 23 | How do fair-share gaps vary across all multi-agent groups? | Exact pixels |
| 24 | How does adversary payoff scale at each group size? | Exact bytes and pixels |
| 25 | How does competition affect homogeneous adversary scaling? | Exact bytes and pixels |
| 26 | How does the adversary model's payoff advantage change with group size? | Exact bytes and pixels |
| 27 | How does capability predict payoff across heterogeneous games? | Exact bytes and pixels |
| 28 | How do payoff distributions differ across Elo groups? | Exact pixels |
| 29 | How does competition affect heterogeneous capability trends? | Exact bytes and pixels |
| 30 | Does the matched GPT-5.4 experiment also produce coalitions? | Exact pixels |

**Figure 1. How does each negotiation game proceed?**

- Reproduction: **Mismatch**.
- The recreated protocol diagram has editable text, new shapes, and recovered Manrope and Inter fonts.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_01/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_01/result.json>).
- Exact recreation is incomplete because the editable drawing and original icons are missing.
- Source-data locations and non-experiment inputs are listed in the detailed report.
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_01/recreated.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_01/Hero_Figure_1_stuffs.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_01/asset_01/comparison.json>).

![Figure 1, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_01/asset_01/recreated_preview.png>)


**Figure 2. How does model capability relate to each role's payoff?**

- Reproduction: **Exact bytes and pixels**.
- All four panels were regenerated from 1,500 raw result files and their configurations.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_02/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_02/result.json>).
- The underlying bilateral cohort contains 11 runtime role discrepancies and the Gemini model-label mismatch.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_02/recreated/bilateral_overview_combined.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_02/bilateral_overview_combined.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_02/asset_01/comparison.json>).

![Figure 2, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_02/asset_01/recreated_preview.png>)


**Figure 3. How do payoffs compare with the fairness benchmarks?**

- Reproduction: **Exact bytes and pixels**.
- All five panels were redrawn after checking 4,230 raw runs.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_03/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_03/result.json>).
- The primary reconstruction recalculates raw results with the original benchmark algorithms and a Haswell numerical setting that matches the rendered output.
- Small scalar differences remain below the image resolution.
- The Game 3 caption crossing differs from the plotted value.
- The included heterogeneous data contain 62 runs with explicit synthetic actions.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_03/output/raw_haswell_fast/fairshare_residual_and_role_endpoint_combined.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_03/fairshare_residual_and_role_endpoint_combined.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_03/asset_01/comparison.json>).

![Figure 3, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_03/asset_01/recreated_preview.png>)


**Figure 4. Which annotated behaviours relate to payoff and capability?**

- Reproduction: **Exact pixels**.
- Both panels were rebuilt from 5,031 saved annotation rows and 1,500 raw negotiation results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_04/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_04/result.json>).
- The exact image preserves historical annotations and their stated source-verification limits.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_04/recreated/n2_qualitative_halfwidth_stacked.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_04/n2_qualitative_halfwidth_stacked.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_04/asset_01/comparison.json>).

![Figure 4, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_04/asset_01/recreated_preview.png>)


**Figure 5. How does additional reasoning effort change payoff?**

- Reproduction: **Exact pixels**.
- All 12 means and 95% intervals were rebuilt from 2,160 raw results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_05/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_05/result.json>).
- Actual token caps and retained recovery configurations are documented for all 2,160 runs.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed1024_20260725_211500](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed1024_20260725_211500>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed128_20260727_043613](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed128_20260727_043613>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed2048_20260727_043613](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed2048_20260727_043613>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed256_20260727_043613](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed256_20260727_043613>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed4096_20260727_043613](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed4096_20260727_043613>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed423_20260725_211500](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed423_20260725_211500>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed526_20260725_181400](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed526_20260725_181400>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed612_20260727_043613](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed612_20260727_043613>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed984_20260725_025700](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed984_20260725_025700>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_05/ttc_game_averaged_target_payoff_vs_compute_halfwidth.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_05/ttc_game_averaged_target_payoff_vs_compute_halfwidth.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_05/asset_01/comparison.json>).

![Figure 5, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_05/asset_01/recreated_preview.png>)


**Figure 6. Does capability predict payoff in larger Game 1 groups?**

- Reproduction: **Exact bytes and pixels**.
- The figure was rebuilt from all 500 heterogeneous Game 1 runs.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_06/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_06/result.json>).
- Explicit synthetic actions occur in 37 of the 500 plotted Game 1 runs.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_06/generated/heterogenous_game1_payoff_singlecolumn.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_06/heterogenous_game1_payoff_singlecolumn.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_06/asset_01/comparison.json>).

![Figure 6, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_06/asset_01/recreated_preview.png>)


**Figure 7. How does payoff variance differ between group types?**

- Reproduction: **Exact pixels**.
- Both panels were rebuilt from all 1,600 included raw results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_07/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_07/result.json>).
- The included data contain 62 heterogeneous and 10 homogeneous runs with explicit synthetic actions.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_07/payoff_variance_homogeneous_runs.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_07/payoff_variance_homogeneous_runs.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_07/asset_01/comparison.json>).

![Figure 7, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_07/asset_01/recreated_preview.png>)


**Figure 8. What do the Gemini coalition transcripts show?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were redrawn from verified source transcripts, winning votes, and payoffs.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_08/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_08/result.json>).
- The panels combine public text and private thinking from source transcripts.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_08/outputs/gemini_coalition_dialogue_slide_inline_membership.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_08/gemini_coalition_dialogue_slide_inline_membership.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_08/asset_01/comparison.json>).

![Figure 8, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_08/asset_01/recreated_preview.png>)


**Figure 9. Where do exclusionary coalitions occur?**

- Reproduction: **Exact pixels**.
- All three panels were rebuilt from checked raw-result summaries and saved manual annotations.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_09/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_09/result.json>).
- The report separates plotted rates from denominator, judge-setting, and screening issues.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_09/coalition_scaling_game_family_elo.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_09/coalition_scaling_game_family_elo.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_09/asset_01/comparison.json>).

![Figure 9, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_09/asset_01/recreated_preview.png>)


**Figure 10. How does a coordinated Nano team compare with an adversary model?**

- Reproduction: **Exact pixels**.
- Both panels were rebuilt from 200 raw results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_10/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_10/result.json>).
- The gray control zero line is a requested display convention rather than the observed optimal-allocation rate.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_10/recreated/gpt54_payoff_ratio_plus_optimal_nano_team.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_10/gpt54_payoff_ratio_plus_optimal_nano_team.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_10/asset_01/comparison.json>).

![Figure 10, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_10/asset_01/recreated_preview.png>)


**Figure 11. What do the Nano team transcripts show?**

- Reproduction: **Exact bytes and pixels**.
- All nine quotes, three payoff vectors, three Nano optima, and vote thresholds were verified before redrawing.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_11/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_11/result.json>).
- The quoted +26 claim is a model arithmetic error preserved from the real transcript.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_11/gpt54_nano_coalition_dynamics_slide.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_11/gpt54_nano_coalition_dynamics_slide.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_11/asset_01/comparison.json>).

![Figure 11, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_11/asset_01/recreated_preview.png>)


**Figure 12. Does the Elo snapshot date change the bilateral trend?**

- Reproduction: **Exact pixels**.
- Both panels were rebuilt from raw outcomes and archived Elo scores.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/result.json>).
- Runtime role discrepancies, model identity mismatches, and incomplete outcomes are documented separately from exact rendering.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/outputs/bilateral_payoff_march31_vs_august2026_lmarena_elo.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_12/bilateral_payoff_march31_vs_august2026_lmarena_elo.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_12/asset_01/comparison.json>).

![Figure 12, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_12/asset_01/recreated_preview.png>)


**Figure 13. Does the trend hold for other capability measures?**

- Reproduction: **Exact pixels**.
- All 11 panels were rebuilt from 1,500 raw results and 298 saved capability scores.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_13/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_13/result.json>).
- The 11 role discrepancies remain in the common bilateral cohort.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_13/capability_benchmarks_over20.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_13/capability_benchmarks_over20.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_13/asset_01/comparison.json>).

![Figure 13, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_13/asset_01/recreated_preview.png>)


**Figure 14. How do role payoffs and variance change in homogeneous adversary runs?**

- Reproduction: **Exact bytes and pixels**.
- Both panels were recalculated from all 1,300 homogeneous adversary runs.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_14/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_14/result.json>).
- The recursive scan finds no structured synthetic actions in the 1,300 homogeneous adversary runs.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_14/homogeneous_adversary_payoff_variance_and_role_payoff.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_14/homogeneous_adversary_payoff_variance_and_role_payoff.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_14/asset_01/comparison.json>).

![Figure 14, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_14/asset_01/recreated_preview.png>)


**Figure 15. How does inequality change when the adversary model is included?**

- Reproduction: **Exact bytes and pixels**.
- All 1,300 raw runs and 7,800 agent payoffs were checked.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_15/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_15/result.json>).
- The baseline-only Gini excludes the adversary model from the inequality calculation.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_15/homogeneous_adversary_baseline_only_vs_all_payoff_gini.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_15/homogeneous_adversary_baseline_only_vs_all_payoff_gini.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_15/asset_01/comparison.json>).

![Figure 15, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_15/asset_01/recreated_preview.png>)


**Figure 16. How does bilateral adversary payoff vary with competition?**

- Reproduction: **Exact bytes and pixels**.
- All six panels were rebuilt from 1,500 raw results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_16/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_16/result.json>).
- The original bilateral cohort and its role and model identity issues are preserved.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_16/recreated_01_02_adversary_payoff_combined.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_16/01_02_adversary_payoff_combined.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_16/asset_01/comparison.json>).

![Figure 16, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_16/asset_01/recreated_preview.png>)


**Figure 17. How does total welfare vary with competition?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were rebuilt from 1,500 raw results and their configurations.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_17/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_17/result.json>).
- The primary image comes from fresh raw-data calculations rather than the separately verified retained table.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_17/10_total_welfare_by_competition_ewma.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_17/10_total_welfare_by_competition_ewma.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_17/asset_01/comparison.json>).

![Figure 17, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_17/asset_01/recreated_preview.png>)


**Figure 18. How far are outcomes from the fairness benchmarks?**

- Reproduction: **Exact bytes and pixels**.
- All three image assets were rebuilt from 1,500 raw results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_18/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_18/result.json>).
- Six empty-preference failures receive incorrect zero fairness benchmarks in the historical calculation.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_18/recreated/11_fairness_distance_overall.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_18/11_fairness_distance_overall.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_18/asset_01/comparison.json>).

![Figure 18, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_18/asset_01/recreated_preview.png>)

- Asset 2: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_18/recreated/12_fairness_distance_by_competition.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_18/12_fairness_distance_by_competition.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_18/asset_02/comparison.json>).

![Figure 18, recreated asset 2](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_18/asset_02/recreated_preview.png>)

- Asset 3: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_18/recreated/13_fairness_excess_by_role_overall.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_18/13_fairness_excess_by_role_overall.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_18/asset_03/comparison.json>).

![Figure 18, recreated asset 3](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_18/asset_03/recreated_preview.png>)


**Figure 19. Does the bilateral trend also appear with a Llama baseline?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were rebuilt from 500 raw indexed experiment results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_19/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_19/result.json>).
- All 500 raw indexed runs were verified.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_analysis_20260503](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_analysis_20260503>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game1_202605](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game1_202605>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game3_202605](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game3_202605>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_19/llama33_overall_utility_overlay_1x3.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_19/llama33_overall_utility_overlay_1x3.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_19/asset_01/comparison.json>).

![Figure 19, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_19/asset_01/recreated_preview.png>)


**Figure 20. How do fixed-baseline games compare with heterogeneous pairs?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were rebuilt from 1,760 raw negotiations.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_20/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_20/result.json>).
- All 1,760 included negotiations were traced and regenerated.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_20/n2_baseline_vs_heterogeneous_pairings.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_20/n2_baseline_vs_heterogeneous_pairings.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_20/asset_01/comparison.json>).

![Figure 20, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_20/asset_01/recreated_preview.png>)


**Figure 21. How many rounds do successful negotiations require?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were rebuilt from 1,500 raw results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_21/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_21/result.json>).
- The historical Python and Matplotlib environment matches PNG bytes; the project environment also matches pixels.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_21/render_historical/07_08_rounds_to_consensus_combined.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_21/07_08_rounds_to_consensus_combined.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_21/asset_01/comparison.json>).

![Figure 21, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_21/asset_01/recreated_preview.png>)


**Figure 22. How does model order relate to negotiation outcomes?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were rebuilt from 1,500 indexed raw records.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_22/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_22/result.json>).
- Two included Game 3 records contain only the baseline agent.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_22/generated/bilateral_order_diagnostics_gpt5_nano_order_1x3.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_22/bilateral_order_diagnostics_gpt5_nano_order_1x3.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_22/asset_01/comparison.json>).

![Figure 22, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_22/asset_01/recreated_preview.png>)


**Figure 23. How do fair-share gaps vary across all multi-agent groups?**

- Reproduction: **Exact pixels**.
- All five panels were rebuilt from 2,900 raw records with the original benchmark algorithm.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_23/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_23/result.json>).
- The primary image uses all 2,900 raw records and the original benchmark algorithm.
- Seventy-two included records have explicit synthetic proposal or vote evidence.
- A different benchmark algorithm was retained only as a separate sensitivity calculation.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_23/outputs/historical_raw/multiagent_fairshare_full.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_23/multiagent_fairshare_full.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_23/asset_01/comparison.json>).

![Figure 23, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_23/asset_01/recreated_preview.png>)


**Figure 24. How does adversary payoff scale at each group size?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were rebuilt from 1,300 homogeneous adversary results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_24/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_24/result.json>).
- The exact image uses the frozen current style and all 1,300 homogeneous adversary runs.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_24/hom_adversary_payoff_vs_elo_by_n.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_24/hom_adversary_payoff_vs_elo_by_n.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_24/asset_01/comparison.json>).

![Figure 24, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_24/asset_01/recreated_preview.png>)


**Figure 25. How does competition affect homogeneous adversary scaling?**

- Reproduction: **Exact bytes and pixels**.
- All 15 panels were rebuilt from 1,300 homogeneous adversary results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_25/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_25/result.json>).
- All 15 panels use the current lowercase group-size labels.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_25/hom_adversary_payoff_vs_elo_by_competition_3x5.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_25/hom_adversary_payoff_vs_elo_by_competition_3x5.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_25/asset_01/comparison.json>).

![Figure 25, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_25/asset_01/recreated_preview.png>)


**Figure 26. How does the adversary model's payoff advantage change with group size?**

- Reproduction: **Exact bytes and pixels**.
- All 1,300 run-level payoff gaps and 75 plotted means were recalculated.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_26/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_26/result.json>).
- All 1,300 run-level gaps and 75 plotted means were recomputed.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_26/hom_adversary_dilution_advantage_vs_n_top_only_clean.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_26/hom_adversary_dilution_advantage_vs_n_top_only_clean.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_26/asset_01/comparison.json>).

![Figure 26, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_26/asset_01/recreated_preview.png>)


**Figure 27. How does capability predict payoff across heterogeneous games?**

- Reproduction: **Exact bytes and pixels**.
- All three panels were rebuilt from 1,300 heterogeneous results.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_27/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_27/result.json>).
- The cohort contains 62 runs with explicit synthetic actions.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_27/heterogeneous_payoff_vs_arena_elo_by_n.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_27/heterogeneous_payoff_vs_arena_elo_by_n.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_27/asset_01/comparison.json>).

![Figure 27, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_27/asset_01/recreated_preview.png>)


**Figure 28. How do payoff distributions differ across Elo groups?**

- Reproduction: **Exact pixels**.
- All 50 bars were rebuilt after checking 1,300 runs and 7,800 agent payoffs.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_28/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_28/result.json>).
- The 62 affected runs contribute to 36 of the 50 plotted bars.
- Only PNG software metadata differs from the reference.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_plus_multiagent_comparison_analysis_20260505>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_28/output/heterogeneous_utility_by_elo_bucket_by_n.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_28/heterogeneous_utility_by_elo_bucket_by_n.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_28/asset_01/comparison.json>).

![Figure 28, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_28/asset_01/recreated_preview.png>)


**Figure 29. How does competition affect heterogeneous capability trends?**

- Reproduction: **Exact bytes and pixels**.
- All 15 panels were rebuilt from 1,300 runs and 7,800 agent appearances.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_29/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_29/result.json>).
- The 62 affected runs enter 318 plotted model-cell means and 27 fitted strata.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_29/heterogeneous_payoff_vs_arena_elo_by_competition_5x3.png>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_29/heterogeneous_payoff_vs_arena_elo_by_competition_5x3.png>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_29/asset_01/comparison.json>).

![Figure 29, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_29/asset_01/recreated_preview.png>)


**Figure 30. Does the matched GPT-5.4 experiment also produce coalitions?**

- Reproduction: **Exact pixels**.
- All four bars were rebuilt after checking 50 matched raw runs, 436 proposals, and 3,424 votes.
- [Full source, data, history, and reproduction report](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_30/report.md>).
- [Machine-readable evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_30/result.json>).
- All 50 raw results, 436 proposals, and 3,424 votes were checked.
- The historical Gemini counting rules differ from the GPT counting rules.
- Experiment and analysis input locations:
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357>).
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_high_matched_coalition_pilot_direct_openai_20260829](</scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_high_matched_coalition_pilot_direct_openai_20260829>).
- Asset 1: [recreated full-size file](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_30/gpt54_gemini_matched_coalition_rates.pdf>); [frozen reference](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/references/figure_30/gpt54_gemini_matched_coalition_rates.pdf>); [comparison evidence](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_30/asset_01/comparison.json>).

![Figure 30, recreated asset 1](</scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/parent_checks/figure_30/asset_01/recreated_preview.png>)


**Did the working paper change during this audit?**

- At final verification, all 32 live figure assets still match their frozen reference hashes.
- At final verification, all captured TeX files match the live TeX files.
- The only new entry in repository Git status is the overview report at /scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/aiwild_figure_recreation_20260905.md.
- Recreations, diagnostics, logs, and comparisons are stored in /scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905.
