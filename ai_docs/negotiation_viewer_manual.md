# Negotiation Viewer UI - User Manual

*Multi-Agent Negotiation Experiment Visualization Tool*

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Interface Guide](#interface-guide)
5. [Features](#features)
6. [Viewing Modes](#viewing-modes)
7. [Data Formats](#data-formats)
8. [Troubleshooting](#troubleshooting)
9. [Customization](#customization)

---

## Overview

The Negotiation Viewer is a Streamlit-based web application for visualizing multi-agent negotiation experiments. It provides:

- **Real-time streaming** of ongoing experiments
- **Post-hoc analysis** of completed experiment results
- **Color-coded phase visualization** for easy tracking
- **Side-by-side agent comparison** showing private vs public interactions
- **Analytics dashboard** with utility tracking and voting patterns

### Use Cases

- Watching experiments unfold in real-time during development
- Reviewing and analyzing completed negotiation transcripts
- Identifying strategic behaviors (manipulation, gaslighting, etc.)
- Comparing agent performance across different model pairings
- Generating insights for research papers and presentations

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Navigate to project root
cd /scratch/gpfs/DANQIC/jz4391/bargain

# Install required packages
pip install -r ui/requirements.txt

# Or install manually
pip install streamlit>=1.28.0 pandas>=1.5.0 plotly>=5.15.0
```

### Verify Installation

```bash
streamlit --version
# Should output: Streamlit, version X.XX.X
```

---

## Quick Start

### Option 1: Using the Launch Script

```bash
# Make script executable (first time only)
chmod +x ui/run_viewer.sh

# Launch with default port (8501)
./ui/run_viewer.sh

# Launch with custom port
./ui/run_viewer.sh --port 8080
```

### Option 2: Direct Streamlit Command

```bash
# From project root
streamlit run ui/negotiation_viewer.py

# With custom options
streamlit run ui/negotiation_viewer.py --server.port 8080 --server.headless true
```

### Option 3: From Any Directory

```bash
streamlit run /scratch/gpfs/DANQIC/jz4391/bargain/ui/negotiation_viewer.py
```

### Accessing the UI

Once launched, open your browser to:
- **Local**: http://localhost:8501
- **Remote/Cluster**: http://[node-hostname]:8501

For Princeton cluster access, you may need SSH tunneling:
```bash
ssh -L 8501:localhost:8501 username@della.princeton.edu
```

---

## Interface Guide

### Layout Overview

```
┌─────────────────────────────────────────────────────────────┐
│  🤝 Multi-Agent Negotiation Viewer                          │
├──────────────┬──────────────────────────────────────────────┤
│              │  📈 Experiment Overview                       │
│  📁 Sidebar  │  ┌─────┬─────┬─────┬─────┐                   │
│              │  │Cons.│Round│Util │Rate │                   │
│  - Mode      │  └─────┴─────┴─────┴─────┘                   │
│  - Folder    ├──────────────────────────────────────────────┤
│  - Run       │  [Timeline] [Comparison] [Analytics] [Raw]   │
│  - Options   │                                               │
│  - Stats     │  ┌─────────────────────────────────────────┐ │
│              │  │                                         │ │
│              │  │           Main Content Area             │ │
│              │  │                                         │ │
│              │  └─────────────────────────────────────────┘ │
└──────────────┴──────────────────────────────────────────────┘
```

### Sidebar Controls

| Control | Description |
|---------|-------------|
| **Mode** | Switch between Post-hoc Analysis and Live Streaming |
| **Experiment** | Select experiment folder from results directory |
| **Run** | Choose specific run number (1, 2, 3, etc.) |
| **Show Prompts** | Toggle visibility of system prompts |
| **Show Private Thinking** | Toggle visibility of private agent reasoning |
| **Agent Comparison Mode** | Enable side-by-side view |

### Main Tabs

1. **📜 Timeline View** - Chronological display of all interactions
2. **👥 Agent Comparison** - Side-by-side agent view per round
3. **📊 Analytics** - Charts and metrics dashboard
4. **🔍 Raw Data** - JSON explorer for debugging

---

## Features

### Phase Visualization

Each negotiation phase is color-coded for easy identification:

| Phase | Color | Icon | Description |
|-------|-------|------|-------------|
| Game Setup | Indigo | 🎮 | Initial rules explanation |
| Preference Assignment | Purple | 🔒 | Private preference reveal |
| Discussion | Blue | 💬 | Public agent conversation |
| Private Thinking | Slate | 🧠 | Internal agent reasoning |
| Proposal | Emerald | 📝 | Allocation proposals |
| Voting | Amber | 🗳️ | Accept/reject votes |
| Reflection | Pink | 🔄 | Post-round analysis |

### Agent Colors

- **Agent_Alpha**: Blue (#3b82f6)
- **Agent_Beta**: Red (#ef4444)
- **System**: Gray (#64748b)

### Interaction Cards

Each interaction is displayed as a card showing:
- Agent name with color indicator
- Phase badge
- Response content
- Expandable prompt (if enabled)
- Parsed JSON data (for proposals/votes)

### Utility Tracking

The analytics tab shows:
- **Line chart**: Proposed utilities over rounds (with discount factor applied)
- **Heatmap**: Voting patterns (green=accept, red=reject)
- **Metrics**: Strategic behavior counts (manipulation, anger, gaslighting, cooperation)

### Agent Comparison

Side-by-side view displays:
- Each agent in a separate column
- Private thinking sections (collapsible)
- Public discussion content
- Clear visual separation between agents

---

## Viewing Modes

### Post-hoc Analysis Mode

For reviewing completed experiments:

1. Select "📂 Post-hoc Analysis" in sidebar
2. Choose an experiment folder from dropdown
3. Select specific run number
4. Navigate through tabs to explore data

**Best for:**
- Detailed analysis of completed experiments
- Comparing outcomes across runs
- Generating figures for papers
- Debugging experiment issues

### Live Streaming Mode

For watching experiments in real-time:

1. Select "🔴 Live Streaming" in sidebar
2. Enter the watch folder path
3. Set file pattern (default: `*all_interactions*.json`)
4. Adjust refresh rate (1-10 seconds)
5. Click "▶️ Start Streaming"

**Best for:**
- Monitoring ongoing experiments
- Development and debugging
- Live demonstrations

**Note:** Live mode automatically detects and displays the most recently modified file matching the pattern.

---

## Data Formats

### Expected Input Files

The viewer expects experiment results in this structure:

```
experiments/results/
└── [experiment_folder]/
    ├── _summary.json                    # Batch summary
    ├── run_1_experiment_results.json    # Run results
    ├── run_1_all_interactions.json      # Full interaction log
    ├── agent_Agent_Alpha_interactions.json
    └── agent_Agent_Beta_interactions.json
```

### Interaction Entry Schema

```json
{
  "timestamp": 1767576458.238585,
  "experiment_id": "strong_models_20260104_202724",
  "agent_id": "Agent_Alpha",
  "phase": "discussion_round_1",
  "round": 1,
  "prompt": "System prompt text...",
  "response": "Agent response text..."
}
```

### Proposal Format

```json
{
  "allocation": {
    "Agent_Alpha": [0, 2],
    "Agent_Beta": [1, 3, 4]
  },
  "reasoning": "Explanation of the proposal...",
  "proposed_by": "Agent_Alpha",
  "round": 1
}
```

### Vote Format

```json
{
  "voter": "Agent_Alpha",
  "proposal_number": 1,
  "vote_decision": "accept",
  "reasoning": "Explanation of the vote..."
}
```

---

## Troubleshooting

### Common Issues

#### "No experiments found"

**Cause:** Results directory doesn't exist or is empty.

**Solution:**
1. Run some experiments first
2. Check the path in `negotiation_viewer.py` line ~45 (`RESULTS_DIR`)

#### Charts not displaying

**Cause:** Plotly not installed.

**Solution:**
```bash
pip install plotly
```

#### Live streaming not updating

**Cause:** File pattern doesn't match or permissions issue.

**Solutions:**
1. Check file pattern matches your output files
2. Ensure read permissions on results directory
3. Try clicking "🔄 Refresh Now" manually

#### Port already in use

**Cause:** Another Streamlit instance or process using port 8501.

**Solution:**
```bash
# Use a different port
./ui/run_viewer.sh --port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9
```

#### SSH tunnel issues on cluster

**Solution:**
```bash
# On your local machine
ssh -L 8501:localhost:8501 -N username@della.princeton.edu

# Then access http://localhost:8501 in browser
```

---

## Customization

### Changing Colors

Edit `ui/negotiation_viewer.py` to modify color schemes:

```python
# Phase colors (line ~25)
PHASE_COLORS = {
    "discussion": "#3b82f6",  # Change to your preferred color
    ...
}

# Agent colors (line ~40)
AGENT_COLORS = {
    "Agent_Alpha": "#3b82f6",
    "Agent_Beta": "#ef4444",
}
```

### Adding Custom Phases

If your experiments use additional phases:

```python
PHASE_COLORS["my_custom_phase"] = "#your_color"
PHASE_ICONS["my_custom_phase"] = "🆕"
```

### Modifying Results Directory

Change the default results path:

```python
# Line ~45 in negotiation_viewer.py
RESULTS_DIR = Path("/your/custom/path/to/results")
```

### Adding Export Formats

Use the component functions in `ui/components.py`:

```python
from ui.components import export_to_markdown, export_to_csv

# Generate markdown transcript
md_content = export_to_markdown(interactions, "my_experiment")

# Generate CSV
csv_content = export_to_csv(interactions)
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `R` | Refresh/Rerun the app |
| `C` | Clear cache |
| `Ctrl+F` | Browser search within page |

---

## Support

For issues or feature requests:

1. Check this manual first
2. Review the code in `ui/negotiation_viewer.py`
3. Check Streamlit documentation: https://docs.streamlit.io
4. Contact the research team

---

*Manual last updated: 2026-01-05*
