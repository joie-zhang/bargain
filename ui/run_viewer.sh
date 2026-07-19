#!/usr/bin/env bash
# Launch the Negotiation Viewer UI
#
# Usage:
#   ./ui/run_viewer.sh              # Default port 8501
#   ./ui/run_viewer.sh --port 8080  # Custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default port
PORT=8501

# Parse arguments
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
            echo "Launch the Multi-Agent Negotiation Viewer"
            echo ""
            echo "Options:"
            echo "  --port PORT    Port to run on (default: 8501)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "Error: port must be an integer from 1 to 65535." >&2
    exit 2
fi

# Activate the project venv if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Activating project virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed in the active environment." >&2
    echo "Install the UI dependencies with: pip install -r ui/requirements.txt" >&2
    exit 1
fi

echo "Launching Negotiation Viewer on 127.0.0.1:$PORT..."
echo "   URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PROJECT_ROOT"
exec streamlit run ui/experiment_viewer.py \
    --server.address 127.0.0.1 \
    --server.port "$PORT" \
    --server.headless true
