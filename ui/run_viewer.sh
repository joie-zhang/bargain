#!/bin/bash
# Launch the Negotiation Viewer UI
#
# Usage:
#   ./ui/run_viewer.sh              # Default port 8501
#   ./ui/run_viewer.sh --port 8080  # Custom port

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default port
PORT=8501

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
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
            shift
            ;;
    esac
done

# Activate the project venv if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Activating project virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

echo "🤝 Launching Negotiation Viewer on port $PORT..."
echo "   URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PROJECT_ROOT"
streamlit run ui/negotiation_viewer.py --server.port "$PORT" --server.headless true
