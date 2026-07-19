# Paper Figure Manifest

This directory covers graphics in all three current paper roots:

- `paper_figure_manifest.csv`: `overleaf/NExT_Game_2026_style_new/sample.tex`.
- `icml_figure_manifest.csv`: `overleaf/icml_aiwild_template/icml_aiwild_2026.tex`.
- `neurips_figure_manifest.csv`: `overleaf/neurips/neurips_2026.tex`.

Each machine-readable manifest contains one row for each active
`\includegraphics` command in that root's TeX input graph.

## Current Inventory

- Main text graphics: 3.
- Appendix graphics: 36.
- Total active graphics: 39.
- ICML active graphics: 40.
- NeurIPS active graphics: 40.
- Appendix retention rule: keep each graphic and its dependencies until the paper
  authors make a final appendix decision.

## Provenance Status

- `verified`: The named script writes the graphic or a named staging copy.
- `inferred`: The script recreates the graphic, but the paper copy has a different
  name or a later visual edit.
- `manual-composite`: The scripts for the component plots exist, but code for the
  final combined image was not found.
- `manual-asset`: The author made the final asset outside this repository. No
  generation script is expected.
- `nonportable`: The producer exists, but it reads a temporary or deleted path.
- `unresolved`: No producer or source was found.

## Open Problems

1. `F005`, `F025`, `F026`, and `F027` have no final composition script.
2. Several scripts write to older Overleaf staging trees. The final release must use
   one explicit export destination.
3. Seven graphic paths are active only in an older root. Keep them and their
   dependencies until the paper-root consolidation is complete.

The repository contains the fixed inputs for `F008` and `F013` under
`scripts/paper_figures/assets/`.

Run the validator after a paper or figure change:

```bash
PYTHONPATH=. python scripts/validate_paper_figure_manifest.py
```

The validator checks active graphic coverage, file hashes, producer paths, and input
paths. It reports known nonportable and unresolved entries as warnings.
