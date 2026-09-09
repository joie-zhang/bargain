# Negotiation transcript viewer

The public transcript explorer covers all three games in one interface.
The game-specific viewers also provide allocation, voting, and funding displays;
see the UI commands in [the README](../../README2.md#ui-viewers).

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
