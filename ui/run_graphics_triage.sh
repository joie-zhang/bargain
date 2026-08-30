#!/usr/bin/env bash
# =============================================================================
# Launch the Unreferenced Graphics Triage Viewer
# =============================================================================
#
# Streamlit UI for deciding keep-or-delete on the image files in
# overleaf/icml_aiwild_template/graphics/ that the compiled ICML AIWILD paper
# does not reference.
#
# Usage:
#   ./ui/run_graphics_triage.sh              # default port 8502
#   ./ui/run_graphics_triage.sh --port 8080
#
# On a cluster, forward the port from your laptop first:
#   ssh -N -L 8502:localhost:8502 <user>@della.princeton.edu
# then open http://localhost:8502
#
# What it creates:
#   docs/reproducibility/graphics_triage_decisions.csv    # written as you click
#   docs/reproducibility/stage_unreferenced_graphics.sh   # via the export button
#
# The viewer never deletes or moves files. The export button writes a staging
# script that you review and run yourself.
#
# Dependencies:
#   .venv with streamlit
#   docs/reproducibility/unreferenced_graphics_manifest.json
#     (built by scripts/build_unreferenced_graphics_manifest.py; this script
#      builds it automatically if missing)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PORT=8502

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            if [[ $# -lt 2 ]]; then
                echo "Error: --port requires a value." >&2
                exit 2
            fi
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--port PORT]"
            echo ""
            echo "Launch the unreferenced-graphics triage viewer."
            echo ""
            echo "Options:"
            echo "  --port PORT    Port to run on (default: 8502)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

cd "$PROJECT_ROOT"

MANIFEST="docs/reproducibility/unreferenced_graphics_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest missing; building it..."
    .venv/bin/python scripts/build_unreferenced_graphics_manifest.py
fi

NODE="$(hostname -f)"
cat <<BANNER

  Triage viewer starting on ${NODE}:${PORT}

  From your laptop, tunnel to THIS node by name:

      ssh -N -L ${PORT}:localhost:${PORT} ${USER}@${NODE}

  then open  http://localhost:${PORT}

  Do NOT tunnel to della.princeton.edu -- it round-robins to a different
  login node (currently $(getent hosts della.princeton.edu 2>/dev/null | awk '{print $2}' || echo della9.princeton.edu)),
  where nothing is listening. Leave this process running while you triage.

  No tunnel? Use the offline page instead, which needs no server:
      python scripts/build_graphics_triage_html.py
      scp ${USER}@${NODE}:${PROJECT_ROOT}/docs/reproducibility/graphics_triage.html .

BANNER

exec .venv/bin/streamlit run ui/graphics_triage_viewer.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
