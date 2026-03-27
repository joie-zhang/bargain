# Run UI Viewer

Help the user launch the correct Streamlit UI and set up port forwarding on their local machine.

## Step 1: Ask which game

Ask the user which game they want to view results for:
- **Game 1**: Item Allocation → `ui/negotiation_viewer.py`
- **Game 2**: Diplomatic Treaty → `ui/experiment_viewer.py`
- **Game 3**: Co-Funding / Participatory Budgeting → `ui/experiment_viewer.py`

Also ask if they want a specific port, defaulting to **8501** if not specified.

## Step 2: Check if a viewer is already running on that port

Run:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/healthz
```

If it returns `200`, tell the user a viewer is already running on that port and skip to Step 4. If not, proceed.

## Step 3: Launch the correct viewer

Activate the venv and launch in the background:

```bash
source .venv/bin/activate && streamlit run ui/<VIEWER>.py --server.port <PORT> --server.headless true &
```

Wait a couple seconds, then confirm it's up with the same `curl` health check.

## Step 4: Tell the user the port forwarding command

**This is the critical step.** First, check the current hostname:

```bash
hostname
```

Use whatever hostname is returned (e.g. `della-pli.princeton.edu`, `della.princeton.edu`, `della-vis1.princeton.edu`, etc.). Then tell the user to run this command **in a terminal on their local laptop** (not on the cluster):

```
ssh -L <PORT>:localhost:<PORT> jz4391@<HOSTNAME>
```

Explain:
- This creates an SSH tunnel forwarding their local port `<PORT>` to port `<PORT>` on the cluster node
- They must **keep that terminal open** while using the viewer
- Once the tunnel is running, open `http://localhost:<PORT>` in their browser

## Notes
- Game 2 and Game 3 both use `experiment_viewer.py`, which has a sidebar to switch between game types
- `diplomacy_latest` in the results directory is a symlink and is intentionally skipped — use the dated directory (e.g., `diplomacy_20260304_015517`) instead
- If the port is already in use by another process, suggest trying port 8502 or 8503
