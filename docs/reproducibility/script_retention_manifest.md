# Public-release script scope

Updated September 9, 2026.

The release retains the game implementation, public launcher, experiment batch
generation, result validation, and the calculations needed to reproduce final
paper results. Intermediate reports and display experiments are outside this
scope.

## Retained code

- The public command-line launcher and game runtime remain in their existing packages.
- Experiment generators, workers, preference locks, and recorded recovery tools remain available.
- Final figure renderers and their data producers remain available.
- Final TTC analyses share calculations through `scripts/ttc_analysis_common.py`.
- Annotation preparation and validation remain available in `scripts/reproduction_scripts/`.
- The old Claude preparation script remains because human annotation review uses its 23-tag subset file.
- Coalition analysis that supplies paper figures and matched experiment inputs remains available.
- The public transcript explorer, dataset downloader, and human annotation review tools remain available.
- The detailed game viewers, batch viewers, shared UI components, and launch command have been restored.
- Seven rebuttal experiment and diagnostic scripts are grouped in `scripts/rebuttal_ablations/`.

The former retained-analysis directory is split into two directories:

- `scripts/reproduction_scripts/` contains 14 annotation and coalition scripts retained for reproduction.
- `scripts/historical_support/` contains eight older preparation, repair, screening, and report scripts for review before deletion.

Each directory has a README that explains every script's role. Historical
support is not needed as a Python import by the reproduction scripts, but its
saved codebooks, inventories, and corrected annotations can still be inputs.
Keep those inputs if the historical source scripts are removed.

The final TTC commands remain separate because they produce different outputs:

- `scripts/analyze_ttc_ten_seeds.py` produces the ten-seed comparison tables, plots, and report.
- `scripts/analyze_ttc_complete_seed_panels.py` supplies the figure pipeline with protocol checks, artifact hashes, and adjusted endpoint tests.

Both currently include all ten seeds and the declared recovery exception.
The obsolete two-, three-, and five-seed commands have been removed after their
shared functions were extracted. The final ten-seed report now describes nine
degrees of freedom and 2,160 runs; its inherited five-seed text was stale.
Keep the earlier seed result directories: those runs are also part of the
final ten-seed dataset.

The saved figure manifests and the September paper-code audit are evidence for
dependencies. They are not complete lists of disposable files.

## Removed code

The initial cleanup removed 43 files from the scripts directory:

- Twelve diagnostic, monitoring, cost-estimate, and display-preview scripts.
- Twenty-one temporary research reports, exploratory analyses, and review tools from the retained-analysis directory.
- Six files from the superseded Claude annotation workflow that used the OpenRouter judge.
- The standalone Game 3 utility exporter used by the retired capability report.
- The ASCII figure-navigation document generator.
- The unused bitmap reconstruction input for the fair-share figure.
- The cluster-specific identifier-check submission wrapper.

The TTC consolidation additionally removed `analyze_ttc_seed_replication.py`,
`analyze_ttc_three_seeds.py`, and `analyze_ttc_five_seeds.py` from the scripts
directory. The experiment generator `generate_ttc_seed_replication_jobs.py`
remains available to reproduce runs.

The identifier check remains available as a direct command:

```bash
export BARGAIN_ROOT="/absolute/path/to/bargain"
"$BARGAIN_ROOT/.venv/bin/python" "$BARGAIN_ROOT/scripts/build_paper_experiment_data_manifest.py" --check --check-identifiers
```

The cleanup also removed seven unused generic assistant helpers, 15 internal UI
files, and three test files specific to the removed internal viewers.
The game viewers and their existing sample-viewer tests have been restored.

These removals retire the old preview and internal review workflows.
They do not remove raw experiment data, annotations, paper assets, or the game
runtime. Historical source versions remain in Git.

## Validation

Use the repository test suite and check that retained imports and figure
producer paths still resolve. The paper figure verifier can regenerate declared
outputs when the corresponding result bundles are present.

Do not apply the old July or August cleanup lists as current deletion commands.
In particular, old-looking filenames can still contain shared calculations or
record the recovery of a published run.
