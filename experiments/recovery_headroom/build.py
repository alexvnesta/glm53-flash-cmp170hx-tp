"""Reuse attested row32 objects; rebuild only the changed KDA C++ unit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/concurrent_throughput'))
from build_row16 import build


def files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()
            and not any(s in ('.git', '__pycache__', '.pytest_cache') for s in p.relative_to(root).parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-profile', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    base, candidate = args.base_profile.resolve(), args.candidate.resolve()
    before, after = files(base / 'engine'), files(candidate / 'engine')
    relative = 'exllamav3/exllamav3_ext/libtorch/gated_delta_net.cpp'
    changes = {k: {'before': before.get(k), 'after': after.get(k)}
               for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    prepared = json.loads((candidate / 'receipts/candidate.json').read_text())
    allowed = {relative}
    if prepared.get('python_patches'):
        expected_patches = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ROOT / 'experiments/tokenizer').glob('*.patch')}
        if prepared['python_patches'] != expected_patches:
            raise ValueError('Candidate tokenizer patch identities changed')
        allowed.add('exllamav3/tokenizer/tokenizer.py')
    if set(changes) != allowed:
        raise ValueError('Only the reviewed KDA unit and optional tokenizer may differ from the attested base')
    if changes != prepared['changed_sources']:
        raise ValueError('Candidate source bytes changed after preparation')
    if prepared['native_patch_sha256'] != hashlib.sha256((HERE / '0001-kda-graph-headroom.patch').read_bytes()).hexdigest():
        raise ValueError('Candidate patch identity changed')
    (candidate / 'engine.sha256').write_text((candidate / 'receipts/engine.sha256').read_text())
    (candidate / 'preparation.json').write_text(json.dumps({
        'decode_max_rows': 32, 'changed_sources': changes,
    }, indent=2) + '\n')
    build(base, candidate, args.python.absolute(),
          'f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046',
          translation_units=[Path(relative)])


if __name__ == '__main__':
    main()
