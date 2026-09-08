#!/usr/bin/env python3
"""Preview and download transcript folders through the anonymous dataset link."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import time
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://anonymous-hf.com/a/xgxza3yfsgll/'
API = 'https://anonymous-hf.com/api/a/xgxza3yfsgll/'
SKIP = ('archive', 'backfill', 'superseded', 'failed_attempt', 'monitoring', 'recovery')


def safe_destination(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError(f'Unsafe dataset path: {name}')
    dest = root.joinpath(*p.parts)
    if not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Destination leaves results root: {name}')
    return dest


def open_url(url, timeout):
    return urlopen(Request(url, headers={'User-Agent': 'bargain-transcript-review/1.0'}), timeout=timeout)


def get_json(url):
    with open_url(url, timeout=60) as response:
        return json.load(response)


def tree(folder=''):
    url = API + 'tree/' + quote(folder, safe='/')
    entries, seen = [], set()
    while url:
        if url in seen or urlparse(url).netloc != 'anonymous-hf.com':
            raise ValueError('Unexpected pagination URL from anonymous dataset service.')
        seen.add(url)
        with open_url(url, timeout=60) as response:
            page = json.load(response)
            link = response.headers.get('Link', '')
        if not isinstance(page, list):
            raise ValueError('Unexpected tree response; expected a list.')
        import re
        match = re.search(r'<([^>]+)>;\s*rel="?next"?', link)
        if len(page) >= 1000 and not match:
            raise ValueError('Directory listing may be truncated; refusing an incomplete download plan.')
        entries.extend(page)
        url = urljoin(url, match.group(1)) if match else None
    return entries


def walk(folder):
    for entry in sorted(tree(folder), key=lambda x: x['path']):
        if entry['type'] == 'directory':
            yield from walk(entry['path'])
        elif entry['type'] == 'file':
            yield entry
        else:
            raise ValueError(f"Unknown tree entry type: {entry['type']}")


def run_folders(folder, limit):
    """Stop metadata traversal after N active transcript folders."""
    found = []
    def visit(current):
        entries = tree(current)
        if any(e['type'] == 'file' and e['path'].endswith('all_interactions.json') for e in entries):
            found.append(current)
            return
        for entry in sorted(entries, key=lambda x: x['path']):
            if len(found) >= limit:
                break
            if entry['type'] == 'directory' and not any(s in entry['path'].lower() for s in SKIP):
                visit(entry['path'])
    visit(folder)
    return found


def file_matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    lfs = entry.get('lfs')
    if lfs:
        expected = lfs.get('oid', lfs.get('sha256', '')).removeprefix('sha256:')
        digest = hashlib.sha256()
    else:
        expected = entry.get('oid', '')
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
    if not expected:
        raise ValueError(f"No content hash supplied for {entry['path']}")
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == expected


def download_file(entry, destination):
    target = safe_destination(destination, entry['path'])
    if target.exists():
        if file_matches(target, entry):
            return 'already verified'
        raise FileExistsError(f'Existing file differs from the dataset: {target}. Move it aside before downloading this version.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.review-', delete=False) as output:
            temp = Path(output.name)
            with open_url(API + 'resolve/' + quote(entry['path'], safe='/'), timeout=120) as response:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
        if not file_matches(temp, entry):
            raise ValueError(f"Downloaded content does not match the listing: {entry['path']}. The dataset may have changed; rerun the preview.")
        # Exclusive creation prevents overwriting an existing experiment file.
        import os
        os.link(temp, target)
        return 'downloaded'
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Select the entire dataset, including all batches and archive files')
    parser.add_argument('--batch', action='append', default=[], help='Exact top-level batch; repeat as needed')
    parser.add_argument('--folder', action='append', default=[], help='Exact dataset folder from the anonymous browser; repeat as needed')
    parser.add_argument('--list', action='store_true', help='List batches without downloading transcripts')
    parser.add_argument('--limit-runs', type=int, help='First N active transcript folders across selected batches')
    parser.add_argument('--download', action='store_true', help='Without this option, preview only')
    args = parser.parse_args()
    if args.all and (args.batch or args.folder or args.limit_runs is not None or args.list):
        parser.error('--all cannot be combined with subset or list options')
    if args.limit_runs is not None and args.limit_runs < 1:
        parser.error('--limit-runs must be positive')
    info = get_json(API + 'info/')
    print('Source:', SOURCE)
    print('Dataset reference:', info['branch'])
    root_entries = tree()
    roots = [e['path'] for e in root_entries if e['type'] == 'directory']
    if args.all:
        args.batch = roots
    if args.list:
        print('\n'.join(roots))
        return
    if not args.batch and not args.folder:
        parser.error('Select --all, --batch, or --folder, or use --list')
    if set(args.batch) - set(roots):
        parser.error('Batch absent from this dataset: ' + ', '.join(sorted(set(args.batch)-set(roots))))
    if args.folder and args.limit_runs:
        parser.error('--limit-runs applies to --batch; --folder already selects an exact folder')
    destination = ROOT / 'experiments' / 'results'
    folders = list(args.folder)
    for batch in dict.fromkeys(args.batch):
        remaining = args.limit_runs - len(folders) if args.limit_runs else None
        if remaining is not None and remaining <= 0:
            break
        folders.extend(run_folders(batch, remaining) if remaining else [batch])
    if not folders:
        parser.error('No active transcript folders found')
    for folder in folders:
        safe_destination(destination, folder)
    print('Reading file metadata for', len(folders), 'selected folders…', flush=True)
    selected = {e['path']: e for e in root_entries if args.all and e['type'] == 'file'}
    for folder in folders:
        print('Listing', folder, flush=True)
        selected.update({entry['path']: entry for entry in walk(folder)})
    if not selected:
        parser.error('The selected folders contain no files')
    entries = list(selected.values())
    total_bytes = sum(e['size'] for e in entries)
    print(f"{len(entries):,} files; {total_bytes:,} bytes ({total_bytes/1e9:.3f} GB) total selected")
    print('Destination:', destination)
    for entry in entries[:8]:
        print(' ', safe_destination(destination, entry['path']))
    if not args.download:
        print('Preview only. Add --download to fetch this selection.')
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        for status in pool.map(lambda e: download_file(e, destination), entries):
            pass
    receipt = destination / '.review_downloads'
    receipt.mkdir(exist_ok=True)
    (receipt / f'{time.time_ns()}.json').write_text(json.dumps({
        'source': SOURCE, 'dataset_info': info, 'files': entries,
        'destination': str(destination),
    }, indent=2) + '\n')
    print('Download complete. Source metadata and verified file hashes saved in', receipt)


if __name__ == '__main__':
    main()
