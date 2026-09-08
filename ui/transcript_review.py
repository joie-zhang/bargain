#!/usr/bin/env python3
"""Local transcript explorer for downloaded historical negotiation runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui.transcript_review_data import discover, prompt_text, read_events, result_path, summary

parser = argparse.ArgumentParser()
parser.add_argument('--results-root', type=Path, default=ROOT / 'experiments' / 'results')
args, _ = parser.parse_known_args()
root = args.results_root.resolve()
st.set_page_config(page_title='Negotiation atlas', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp {background:#f7f8fc;color:#17233b}
.block-container {max-width:1550px;padding-top:2rem}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:white;border:1px solid #e2e7ef;border-radius:14px;padding:16px}
[data-testid="stSidebar"] {background:#edf1f8}
[data-testid="stChatMessage"] {background:white;border:1px solid #e2e7ef;border-radius:12px}
</style>''', unsafe_allow_html=True)
st.title('Negotiation atlas')
st.caption('Explore the conversations behind the experiments. Filter a collection and inspect a run.')

if not root.is_dir():
    st.info(f'No downloaded results found at {root}. Follow the download guide, then refresh.')
    st.stop()

@st.cache_data(show_spinner=False)
def cached_summary(relative, transcript_mtime, result_mtime):
    # Schema v2: historical discussion_round_N_turn_N phase names are public discussion.
    return summary(root, relative)

with st.sidebar:
    st.header('Your collection')
    st.caption(str(root))
    batches = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(('.', 'TO_DELETE', 'excluded_', 'superseded_')) and not p.is_symlink())
    selected_batches = st.multiselect('Experiment batches', batches, default=([b for b in batches if b.startswith('appendix_llama33_baseline_game1')][:1] or batches[:1]))
    st.caption('Only the batches you select are scanned.')
    include_history = st.checkbox('Include archive and recovery folders', value=False)
    query = st.text_input('Filter run paths', placeholder='Model, seed, competition, or config ID')
    limit = st.number_input('Maximum runs to index', min_value=10, max_value=10000, value=300, step=50)
    st.caption('The first matching paths are indexed in sorted order. This is a browsing subset, not a random sample.')
    if st.button('Refresh downloaded files'):
        st.cache_data.clear()
        st.rerun()

paths = [p for p in discover(root, selected_batches, include_history) if query.lower() in p.lower()]
if not paths:
    st.info('No transcripts match. Select a downloaded batch or clear the path filter.')
    st.stop()
rows, errors = [], []
with st.spinner('Reading the selected transcript metadata…'):
    for relative in paths[:limit]:
        try:
            path = root / relative
            result = result_path(path)
            rows.append(cached_summary(relative, path.stat().st_mtime_ns,
                                       result.stat().st_mtime_ns if result.exists() else 0))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f'{relative}: {exc}')
if errors:
    st.warning(f'{len(errors)} files could not be indexed. They are excluded from this view.')
    with st.expander('Read errors'):
        st.text('\n'.join(errors))
if not rows:
    st.stop()
df = pd.DataFrame(rows)
with st.sidebar:
    games = st.multiselect('Games', sorted(df.game.unique()), default=sorted(df.game.unique()))
    model_query = st.text_input('Filter model names')
    agreement = st.selectbox('Agreement', ['All', 'Reached', 'Not reached', 'Unknown'])
    cue_groups = st.multiselect('Discussion cue groups', sorted(df.cue_group.unique()))
    st.caption('Cue groups use keyword matches in public discussion, not validated behavior labels.')
visible = df[df.game.isin(games) & df.models.str.contains(model_query, case=False, regex=False)]
if agreement != 'All':
    visible = visible[visible.agreement == {'Reached': True, 'Not reached': False, 'Unknown': 'Unknown'}[agreement]]
if cue_groups:
    visible = visible[visible.cue_group.isin(cue_groups)]
cols = st.columns(4)
cols[0].metric('Matching files', f'{len(paths):,}')
cols[1].metric('Indexed runs', f'{len(df):,}')
cols[2].metric('Visible runs', f'{len(visible):,}')
cols[3].metric('Experiment batches', visible.batch.nunique())
st.caption('Folder contents can include extra seeds, older attempts, and controls outside the paper selection. These counts are file counts, not the paper’s sample sizes.')
if visible.empty:
    st.info('No indexed runs match these filters.')
    st.stop()

left, right = st.columns([1, 2], gap='large')
with left:
    st.subheader('Run browser')
    table = visible[['game', 'models', 'agents', 'agreement', 'rounds', 'cue_group']].copy()
    table['agreement'] = table.agreement.map({True: 'Reached', False: 'Not reached', 'Unknown': 'Unknown'})
    st.dataframe(table, hide_index=True, height=250)
    st.download_button('Export visible run index', visible.to_csv(index=False), 'transcript_index.csv', 'text/csv')
    options = visible.path.tolist()
    if st.session_state.get('chosen_run') not in options:
        st.session_state['chosen_run'] = options[0]
    chosen = st.selectbox('Open transcript', options, key='chosen_run')
    path = root / chosen
    st.caption(str(path))
    events = read_events(path)
    result = result_path(path)
    result_data = json.loads(result.read_text()) if result.exists() else {}
    st.download_button('Download original transcript JSON', path.read_bytes(), path.name, 'application/json')
    if result_data:
        with st.expander('Outcome and run configuration'):
            st.json({k:v for k,v in result_data.items() if k in ['config','final_utilities','final_allocation','consensus_reached','final_round','agent_preferences']})
    else:
        st.warning('Companion result file is missing. Outcome and game metadata are unknown.')
with right:
    st.subheader('Conversation')
    a, b = st.columns(2)
    agents = a.multiselect('Speakers', sorted({str(e.get('agent_id', 'Unknown')) for e in events}))
    phases = b.multiselect('Phases', sorted({str(e.get('phase', 'Unknown')) for e in events}))
    text_query = st.text_input('Search response text', placeholder='Search this conversation')
    show_prompts = st.checkbox('Show prompts, including externalized prompts')
    filtered = [(i,e) for i,e in enumerate(events) if
                (not agents or str(e.get('agent_id', 'Unknown')) in agents) and
                (not phases or str(e.get('phase', 'Unknown')) in phases) and
                text_query.lower() in str(e.get('response', '')).lower()]
    pages = max(1, (len(filtered)+29)//30)
    page = st.number_input('Message page', 1, pages, 1, key=f'page_{chosen}')
    st.caption(f'{len(filtered)} matching messages · original file order · 30 messages per page')
    for index, e in filtered[(page-1)*30:page*30]:
        with st.chat_message('assistant'):
            st.caption(f"#{index+1} · Round {e.get('round', '?')} · {e.get('agent_id', 'Unknown')} · {e.get('phase', 'Unknown')} · {e.get('model_name', '')}")
            # Render model text literally so generated HTML/links are never executed.
            st.text(str(e.get('response', '')))
            if show_prompts:
                with st.expander('Prompt'):
                    try:
                        st.text(prompt_text(e, path.parent))
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
            with st.expander('Full event record'):
                st.json(e)
