# Audit review and UI handoff

## Result

- The read-only audit UI is running on `della-vis2.princeton.edu`, port 8003.
- Its persistent tmux session is `bargain-code-audit-8003`.
- It binds to `127.0.0.1` only.
- It has no file deletion, move, commit or approval controls.
- It serves audit evidence and allowlisted report downloads, not arbitrary source-file contents.

## Connect from your laptop

Run this command and leave its terminal open:

```bash
ssh -N -L 8003:127.0.0.1:8003 -J jz4391@della-pli.princeton.edu jz4391@della-vis2.princeton.edu
```

Then open [http://127.0.0.1:8003](http://127.0.0.1:8003).

- Complete Duo authentication if prompted.
- A quiet terminal after authentication is normal for this tunnel command.
- If laptop port 8003 is already occupied, change only the local port to 18003 and open the corresponding URL:

```bash
ssh -N -L 18003:127.0.0.1:8003 -J jz4391@della-pli.princeton.edu jz4391@della-vis2.princeton.edu
```

Open [http://127.0.0.1:18003](http://127.0.0.1:18003) for that alternative.

## What the UI shows

- Overview shows the 1,204-file code inventory and its four status groups.
- Experiment audits links all 45 assignments and the two root cross-checks to their reports.
- File explorer searches 12,078 keep entries or the code inventory, with status and audit filters.
- File detail gives the source report, retention role, reason and evidence location.
- Cleanup candidates separates nine conditional source candidates from 25 bytecode candidates.
- Paper coverage maps all 30 figures and 10 tables to audit IDs and source locations.
- Review checks shows 15 fresh consistency checks and their limits.
- Full report includes a downloadable Markdown report.

## What the review corrected

- The order-diagnostics metadata used `E34` while the coverage map used `e34`.
- That inconsistency broke one link and caused the initial viewer to count 44 agent audits instead of 45.
- Corrected the task ID and its references in six audit artifacts.
- No keep/delete classification or experimental result changed.
- The fresh checks verify report coverage, path existence, keep-list union, candidate conflicts, source counterparts, mapped paper labels and totals.
- They do not repeat every scientific calculation or establish complete dependency coverage.
- The 664 unresolved code paths and 43 directory-selection paths remain explicit limitations.
- The new UI and tests are outside the original September 7 inventory.

## Checks performed

- All 15 fresh audit consistency checks passed.
- All 30 Python tests passed against the real audit records and a real loopback server.
- Real Chromium checks passed for seven views, E34 navigation, path search, candidate-type filtering, paper coverage, report download and mobile width.
- Browser checks reported no JavaScript page errors.
- Desktop and mobile screenshots were inspected.
- Remote HTTP health check returned 200 at `/api/status`.
- The UI has no image endpoints; screenshots below are handoff artifacts rather than UI dependencies.
- Unknown file paths, traversal attempts, non-loopback Host values and write methods were rejected in tests.

## Deliverables

- [Reviewed audit report](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/paper_code_audit_20260907/report.md).
- [Fresh consistency-check results](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/paper_code_audit_20260907/ui_review.json).
- [Desktop screenshot](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/paper_code_audit_20260907/ui_overview.png).
- [Mobile screenshot](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/paper_code_audit_20260907/ui_mobile.png).
- [Python server](/scratch/gpfs/DANQIC/jz4391/bargain/ui/paper_code_audit_viewer.py).
- [HTML and styles](/scratch/gpfs/DANQIC/jz4391/bargain/ui/paper_code_audit_viewer.html).
- [Browser code](/scratch/gpfs/DANQIC/jz4391/bargain/ui/paper_code_audit_viewer.js).
- [Python tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_paper_code_audit_viewer.py).
- [Browser test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/browser_paper_code_audit_viewer.cjs).

## Server operation

The exact persistent launch command used was:

```bash
tmux new-session -d -s bargain-code-audit-8003 -c /scratch/gpfs/DANQIC/jz4391/bargain 'exec env PYTHONDONTWRITEBYTECODE=1 /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python -u /scratch/gpfs/DANQIC/jz4391/bargain/ui/paper_code_audit_viewer.py --host 127.0.0.1 --port 8003 > /scratch/gpfs/DANQIC/jz4391/bargain/logs/ui/paper_code_audit_8003.log 2>&1'
```

- Do not run that command while this session or port already exists.
- The log is `/scratch/gpfs/DANQIC/jz4391/bargain/logs/ui/paper_code_audit_8003.log`.
- The server loads audit records once at startup and records a digest.
- If indexed report files change, the overview displays a stale-input warning.
- Restart only this viewer after changes to review the new records.
- To attach to its process, run `tmux attach -t bargain-code-audit-8003` on `della-vis2`.
- Detach without stopping it with `Ctrl-b`, then `d`.

Check the remote service with:

```bash
curl -fsS http://127.0.0.1:8003/api/status
```

## History and skills used

- Used codex-search on locally available history, then manually checked the original messages.
- Session `019fea02-fcba-7760-9da8-de5c4b050fbe`, lines 8 and 71, showed the failed tunnel and the persistent tmux fix on `della-vis2`.
  - Source: `/home/jz4391/.codex/sessions/2026/08/10/rollout-2026-08-10T00-51-32-019fea02-fcba-7760-9da8-de5c4b050fbe.jsonl`.
- Session `019f108f-6eba-7471-a17f-2836e9b58005`, lines 6103 and 6168, documented the move from a crashing Streamlit viewer to plain HTTP.
  - Source: `/home/jz4391/.codex/sessions/2026/06/28/rollout-2026-06-28T19-27-43-019f108f-6eba-7471-a17f-2836e9b58005.jsonl`.
- Used the cluster-ui-ssh-tunnel skill for loopback binding, persistent launch, remote health verification and exact jump-host command generation.
- The viewer uses the Python standard library and local static assets, with no CDN or browser-side external service.
- Browser test tools were installed only under `/tmp/bargain-audit-browser.EK6fmz`, not into the project environment.

No experiment runtime code, paper source or research data was changed, and nothing was deleted or committed.
