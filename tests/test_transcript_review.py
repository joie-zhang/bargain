import gzip
import hashlib
import json

import pytest

from scripts.download_review_transcripts import safe_destination, file_matches
from ui.transcript_review_data import prompt_text, summary


def test_download_paths_and_content_checks(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        safe_destination(tmp_path, '../outside')
    with pytest.raises(ValueError, match='Unsafe'):
        safe_destination(tmp_path, '/absolute')
    target = safe_destination(tmp_path, 'batch/run/file.json')
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{}')
    entry = {'path':'batch/run/file.json', 'size':2,
             'oid':hashlib.sha1(b'blob 2\0{}').hexdigest()}
    assert file_matches(target, entry)
    target.write_bytes(b'[]')
    assert not file_matches(target, entry)


def test_external_prompts_require_safe_path_and_correct_hash(tmp_path):
    text = 'Exact original prompt\n'
    with gzip.open(tmp_path / 'prompt.gz', 'wt') as f:
        f.write(text)
    event = {'prompt_storage_path': 'prompt.gz', 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}
    assert prompt_text(event, tmp_path) == text
    with pytest.raises(ValueError, match='leaves'):
        prompt_text({'prompt_storage_path': '../outside'}, tmp_path)
    with pytest.raises(ValueError, match='checksum'):
        prompt_text({**event, 'prompt_sha256': 'wrong'}, tmp_path)
    with pytest.raises(FileNotFoundError):
        prompt_text({'prompt_storage_path': 'missing'}, tmp_path)


def test_summary_excludes_prompt_and_private_text_from_cues(tmp_path):
    run = tmp_path / 'batch' / 'run'
    run.mkdir(parents=True)
    events = [
        {'experiment_id': 'id', 'agent_id': 'A', 'phase': 'discussion_round_1_turn_1', 'response': 'We can trade.', 'prompt': 'coalition coalition'},
        {'experiment_id': 'id', 'agent_id': 'A', 'phase': 'private_thinking', 'response': 'coalition coalition'},
    ]
    (run / 'run_1_all_interactions.json').write_text(json.dumps(events))
    (run / 'experiment_results.json').write_text(json.dumps({'experiment_id':'id', 'config':{'game_type':'diplomacy'},'consensus_reached':False}))
    row = summary(tmp_path, 'batch/run/run_1_all_interactions.json')
    assert row['game'] == 'diplomacy'
    assert row['agreement'] is False
    assert row['Coalition / coordination'] == 0
    assert row['Trade / compromise'] > 0
    (run / 'experiment_results.json').write_text('{"experiment_id":"different"}')
    with pytest.raises(ValueError, match='mismatch'):
        summary(tmp_path, 'batch/run/run_1_all_interactions.json')
