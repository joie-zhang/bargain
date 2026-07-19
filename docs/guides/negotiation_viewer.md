# Negotiation Viewer

## Scope

The Streamlit UI shows completed runs and active runs. It reads JSON files from `experiments/results/`.

## Install The UI Requirements

Activate the project environment. Then, install the UI requirements.

```bash
source .venv/bin/activate
pip install -r ui/requirements.txt
```

## Start The Main Viewer

Use the launcher for local work:

```bash
bash ui/run_viewer.sh --port 8501
```

The launcher starts `ui/experiment_viewer.py`. It supports Game 2 and Game 3.
Open `http://localhost:8501` on the same computer.

The main viewer supports Games 1, 2, and 3. It lets you select a result root and inspect run state.

## Start A Specialized Viewer

Use a specialized viewer when you need a game-specific layout:

```bash
streamlit run ui/multi_game_sample_viewer.py --server.address 127.0.0.1 --server.port 8501
streamlit run ui/game1_sample_viewer.py --server.address 127.0.0.1 --server.port 8501
streamlit run ui/game2_batch_viewer.py --server.address 127.0.0.1 --server.port 8501
streamlit run ui/game3_batch_viewer.py --server.address 127.0.0.1 --server.port 8501
```

Use `multi_game_sample_viewer.py` to inspect a set of samples from all three
games. Use a game-specific viewer when you need batch filters or exports.

## Use The Viewer On A Remote Cluster

Bind Streamlit to the loopback interface on the remote node:

```bash
streamlit run ui/experiment_viewer.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

Create an SSH tunnel from your local computer:

```bash
ssh -N -L 8501:127.0.0.1:8501 USER@REMOTE_HOST
```

Open `http://localhost:8501` on your local computer.

## Required Result Files

The viewer can use these files:

- `experiment_results.json` contains the final utilities and outcome.
- `all_interactions.json` contains the complete transcript.
- `progress.json` contains active-run progress.
- `agent_interactions/` contains per-agent transcript files.

Some old runs use names such as `run_1_experiment_results.json`. The viewers include compatibility code for these files.

## Troubleshooting

If the UI shows no runs, select the correct result root. Confirm that the root contains result JSON files.

If the port is in use, select a different port:

```bash
bash ui/run_viewer.sh --port 8502
```

If a remote browser cannot connect, confirm the loopback bind and the SSH tunnel. Do not expose the Streamlit port to a public network.
