# UI Development Conversation Log

*Saved from Claude Code session on 2026-01-05*

This document captures the key conversation exchanges during the development of the Streamlit Negotiation Viewer UI.

---

## Exchange 1: Initial Request and Requirements Gathering

### User Prompt
```
could you help me spin up a UI to make the streaming input more pretty and user-friendly for this experiment setup of [JSON data showing negotiation interactions between Agent_Alpha and Agent_Beta]
```

### Assistant Response
The assistant asked clarifying questions about:

1. **UI Framework Preference:**
   - Streamlit (Recommended) - Python-based, quick to build, great for research
   - React/TypeScript - More polished, better for complex interactions
   - Simple HTML/JS - Lightweight, no dependencies

2. **Viewing Mode:**
   - Live streaming (Recommended) - Watch experiments unfold in real-time
   - Post-hoc analysis - Load and explore completed experiment JSON files
   - Both modes

3. **Key Features:**
   - Clear phase visualization - Color-coded phases with distinct styling
   - Agent comparison view - Side-by-side showing private thinking vs public discussion
   - Utility tracking - Charts showing utility over rounds, proposal acceptance rates
   - All of the above

**User Selected:** Streamlit, Both modes, All features

---

## Exchange 2: Codebase Exploration Results

### Assistant Action
Launched an exploration agent to understand the experiment structure before building the UI.

### Key Findings from Exploration

#### Negotiation Phases (6-Phase Protocol)
1. **Discussion** - Agents share information about preferences
2. **Proposal** - Each agent submits an allocation proposal
3. **Proposal Enumeration** - System displays all proposals numerically
4. **Voting** - Each agent votes accept/reject on each proposal
5. **Vote Tabulation** - System checks for unanimous consensus
6. **Reflection** - Agents reflect on outcomes (if no consensus)

#### Data Schema
```python
# Main Results File Structure
{
    "experiment_id": "strong_models_20251123_175648_856861",
    "timestamp": float,
    "config": {...},
    "consensus_reached": bool,
    "final_round": int,
    "final_utilities": {"Agent_Alpha": float, "Agent_Beta": float},
    "final_allocation": {"Agent_Alpha": [0, 2], "Agent_Beta": [1, 3, 4]},
    "agent_preferences": {"Agent_Alpha": [8.0, 2.0, ...], ...},
    "strategic_behaviors": {
        "manipulation_attempts": int,
        "anger_expressions": int,
        "gaslighting_attempts": int,
        "cooperation_signals": int
    },
    "conversation_logs": [...]
}
```

#### Results Directory Structure
```
experiments/results/
├── [Model_A]_vs_[Model_B]_runs[N]_comp[0-1]/
│   ├── _summary.json
│   ├── run_1_experiment_results.json
│   ├── run_1_all_interactions.json
│   ├── agent_Agent_Alpha_interactions.json
│   └── agent_Agent_Beta_interactions.json
```

---

## Exchange 3: UI Implementation Complete

### Assistant Response

Created a comprehensive Streamlit-based UI with the following files:

```
ui/
├── negotiation_viewer.py   # Main Streamlit app (29KB)
├── components.py           # Reusable UI components (15KB)
├── requirements.txt        # Dependencies
├── run_viewer.sh          # Launch script
└── __init__.py            # Package init
```

#### Features Implemented

**1. Phase Visualization (Color-coded)**
- 🎮 Game Setup (Indigo)
- 🔒 Preference Assignment (Purple)
- 💬 Discussion (Blue)
- 🧠 Private Thinking (Slate - muted)
- 📝 Proposals (Emerald)
- 🗳️ Voting (Amber)
- 🔄 Reflection (Pink)

**2. Agent Comparison View**
- Side-by-side layout showing both agents
- Distinguishes private thinking from public discussion
- Shows reasoning and strategy for each agent

**3. Utility Tracking Charts**
- Line chart of proposed utilities over rounds
- Voting heatmap showing accept/reject patterns
- Strategic behavior metrics dashboard

**4. Dual Mode Support**
- Post-hoc Analysis: Browse and explore completed experiments
- Live Streaming: Watch experiments unfold in real-time

#### Quick Start
```bash
# Install dependencies
pip install streamlit pandas plotly

# Run the viewer
streamlit run ui/negotiation_viewer.py

# Or use the launch script
./ui/run_viewer.sh --port 8501
```

#### UI Layout
- **Sidebar**: Experiment folder selection, run picker, view options
- **Tab 1 (Timeline)**: Round-by-round interaction cards
- **Tab 2 (Comparison)**: Side-by-side agent view per round
- **Tab 3 (Analytics)**: Utility charts, voting patterns, strategic behaviors
- **Tab 4 (Raw Data)**: JSON explorer for debugging

---

## Technical Notes

### Phase Color Mapping
```python
PHASE_COLORS = {
    "game_setup": "#6366f1",           # Indigo
    "preference_assignment": "#8b5cf6", # Purple
    "discussion": "#3b82f6",            # Blue
    "private_thinking": "#64748b",      # Slate
    "proposal": "#10b981",              # Emerald
    "voting": "#f59e0b",                # Amber
    "reflection": "#ec4899",            # Pink
}
```

### Agent Color Mapping
```python
AGENT_COLORS = {
    "Agent_Alpha": "#3b82f6",  # Blue
    "Agent_Beta": "#ef4444",   # Red
    "system": "#64748b",       # Slate
}
```

### Key Data Loading Functions
- `load_experiment_folders()` - Find all experiment result folders
- `load_all_interactions()` - Load interaction JSON for an experiment
- `load_experiment_results()` - Load results for a specific run
- `extract_preferences()` - Parse agent preferences from prompts
