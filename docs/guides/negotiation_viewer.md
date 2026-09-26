# Negotiation transcript viewer

The public transcript explorer covers all three games in one interface.
It reads the released paper results downloaded with the
[README instructions](../../README.md#inspect-released-results).
For new public-launcher outputs, use `bargain status` and `bargain summarize`
as described in the [launcher guide](public_quickstart.md#inspect-a-plan-and-results).

## Install and start

Set the checkout location and install the viewer dependencies:

```bash
export BARGAIN_ROOT="/absolute/path/to/bargain"
python3 -m venv "$BARGAIN_ROOT/.venv-review"
"$BARGAIN_ROOT/.venv-review/bin/python" -m pip install -r "$BARGAIN_ROOT/ui/requirements.txt"
```

Download the published transcripts if needed:

```bash
"$BARGAIN_ROOT/.venv-review/bin/python" "$BARGAIN_ROOT/scripts/download_review_transcripts.py" --all --download
```

Start the viewer on the machine that holds the data:

```bash
"$BARGAIN_ROOT/.venv-review/bin/python" -m streamlit run \
  "$BARGAIN_ROOT/ui/transcript_review.py" \
  --server.address 127.0.0.1 --server.port 8010 \
  --server.headless true --browser.gatherUsageStats false
```

Open http://127.0.0.1:8010 on that machine. Keep the terminal open.

## Remote access

Check the viewer on the remote machine:

```bash
curl --fail http://127.0.0.1:8010/_stcore/health
```

The response should be `ok`. On your laptop, start an SSH tunnel:

```bash
ssh -N -L 8010:127.0.0.1:8010 YOUR_USERNAME@VIEWER_HOST
```

For a host reached through a login node:

```bash
ssh -N -L 8010:127.0.0.1:8010 -J YOUR_USERNAME@LOGIN_HOST YOUR_USERNAME@VIEWER_HOST
```

Keep the tunnel open and visit http://127.0.0.1:8010 in the laptop browser.

## Inspect results

Select a batch and run, then read the messages, saved prompts, configuration, and
outcome. Filters cover game, model, seed, and agreement. The viewer can export
transcript JSON and the visible run index as CSV.

The separate human annotation review interface remains available for the
inter-rater agreement workflow.

## Game-specific viewers

These viewers provide allocation, voting, and funding displays for paper-format
results. Activate the environment with the viewer dependencies installed.

For Game 2 and Game 3 trajectories:

```bash
python -m streamlit run "$BARGAIN_ROOT/ui/experiment_viewer.py" \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

For coordinated-team transcripts:

```bash
bash "$BARGAIN_ROOT/ui/run_binding_team_viewer.sh"
```

The Game 2 and Game 3 command uses port `8501`.
The team launcher uses the checkout's `.venv`, the saved team batch, and port
`8002` by default; set `RESULTS_ROOT` or `PORT` to select another collection or port.
Use the selected port in the browser address and SSH tunnel.
Run one viewer per port.

Other viewers are available for specific collections:

- [Multi-game samples](../../ui/multi_game_sample_viewer.py)
- [Game 1 samples](../../ui/game1_sample_viewer.py)
- [Game 2 batches](../../ui/game2_batch_viewer.py)
- [Game 3 batches](../../ui/game3_batch_viewer.py)
- [Random-monoculture samples](../../ui/random_monoculture_sample_viewer.py)
