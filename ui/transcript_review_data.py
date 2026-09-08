"""Read-only transcript helpers. No model calls and no downloads."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path


CUES = {
    'Trade / compromise': r'\b(trade|swap|compromise|concession|concede|meet halfway)\b',
    'Coalition / coordination': r'\b(coalition|team|alliance|coordinate|coordination|majority|outsider)\b',
    'Fairness / equality': r'\b(fair|fairness|equal|equally|equitable)\b',
    'Pressure / threats': r'\b(threat|threaten|reject|refuse|ultimatum|last chance)\b',
    'Self-interest': r'\b(my utility|my payoff|my benefit|my share|maximize my)\b',
}
SKIP = ('archive', 'backfill', 'superseded', 'failed_attempt', 'monitoring', 'recovery')


def discover(root: Path, batches: list[str], include_history=False):
    paths = []
    for batch in batches:
        for path in (root / batch).rglob('*all_interactions.json'):
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            rel = path.relative_to(root).as_posix()
            if include_history or not any(s in rel.lower() for s in SKIP):
                paths.append(rel)
    return sorted(set(paths))


def read_events(path: Path):
    events = json.loads(path.read_text())
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        raise ValueError(f'Expected a list of interaction objects: {path}')
    return events


def result_path(path: Path):
    paired = path.with_name(path.name.replace('all_interactions.json', 'experiment_results.json'))
    if paired.exists():
        return paired
    # Historical Game 2/3 use run_1 interactions with an unprefixed terminal result.
    other = path.with_name('experiment_results.json')
    if path.name == 'run_1_all_interactions.json' and other.exists():
        return other
    return paired


def summary(root: Path, relative: str):
    path = root / relative
    events = read_events(path)
    result = result_path(path)
    data = json.loads(result.read_text()) if result.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f'Expected a result object: {result}')
    ids = {e.get('experiment_id') for e in events if e.get('experiment_id')}
    if data and ids and data.get('experiment_id') not in ids:
        raise ValueError(f'Result/transcript experiment ID mismatch: {path}')
    config = data.get('config', {})
    # Only public discussion responses enter the keyword cue counts.
    texts = [str(e.get('response', '')) for e in events if re.fullmatch(r'discussion(?:_round_\d+(?:_turn_\d+)?)?', str(e.get('phase', '')))]
    text = '\n'.join(texts)
    words = len(re.findall(r'\b\w+\b', text))
    counts = {k: len(re.findall(v, text, re.I)) for k, v in CUES.items()}
    best = max(counts, key=counts.get)
    return {
        'path': relative, 'result_path': result.relative_to(root).as_posix(),
        'batch': relative.split('/')[0], 'game': config.get('game_type', 'Unknown'),
        'models': ', '.join(sorted({str(e['model_name']) for e in events if e.get('model_name')})),
        'agents': len({e['agent_id'] for e in events if e.get('agent_id')}),
        'agreement': data.get('consensus_reached', 'Unknown'),
        'rounds': data.get('final_round'), 'messages': len(events),
        'discussion_words': words, 'cue_group': best if counts[best] else 'No cue matches',
        **{k: v / max(words, 1) * 1000 for k, v in counts.items()},
    }


def prompt_text(event, folder: Path):
    stored = event.get('prompt_storage_path')
    if not stored:
        return str(event.get('prompt', ''))
    path = (folder / stored).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError('External prompt path leaves the transcript folder.')
    if not path.exists():
        raise FileNotFoundError(f'Missing external prompt: {path}. Download the full run folder.')
    text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
    expected = event.get('prompt_sha256')
    if expected and hashlib.sha256(text.encode()).hexdigest() != expected:
        raise ValueError(f'External prompt checksum mismatch: {path}')
    return text
