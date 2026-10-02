#!/usr/bin/env python3
"""Index original P6 assets and hashes without duplicating experiment data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def sha(file):
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(repository, destination):
    if destination.exists():
        raise FileExistsError(destination)
    text = subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=repository, text=True)
    sources = []
    for block in text.strip().split('\n\n'):
        item = dict(line.split(' ', 1) for line in block.splitlines() if ' ' in line)
        name = Path(item['worktree']).name
        if name.startswith('Ada-MGAD-e2e-v2-') and not name.endswith(('integration', 'thesis-freeze')):
            sources.append(item)
    destination.mkdir(parents=True)
    count = 0
    with (destination / 'asset_catalog.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_tree', 'source_head', 'source_path', 'bytes', 'sha256'], lineterminator='\n')
        writer.writeheader()
        for item in sources:
            tree = Path(item['worktree'])
            for file in sorted((tree / 'experiments/p6').rglob('*')):
                if file.is_symlink():
                    raise ValueError('do not follow shared input links: ' + str(file))
                if not file.is_file() or '__pycache__' in file.parts or file.suffix == '.pyc':
                    continue
                writer.writerow(dict(source_tree=tree.name, source_head=item['HEAD'],
                    source_path=str(file.relative_to(tree)), bytes=file.stat().st_size, sha256=sha(file)))
                count += 1
    (destination / 'asset_inventory.json').write_text(json.dumps(dict(sources=sources,
        file_count=count, catalog_sha256=sha(destination / 'asset_catalog.csv'),
        status='INVENTORY_ONLY', data_copied=False, original_assets_removed=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    inventory(args.repository.resolve(), args.output_dir.resolve())
