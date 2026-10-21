"""Prepare an isolated recovery/headroom candidate; never changes a live unit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--with-tokenizer', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    subprocess.run([sys.executable, str(repo / 'profiles/community-four-active/prepare.py'), str(output)], check=True)
    patch = here / '0001-kda-graph-headroom.patch'
    engine = output / 'engine'
    pin_files = {'exllamav3/exllamav3_ext/libtorch/gated_delta_net.cpp': 'glm_native_concurrency.py'}
    if args.with_tokenizer:
        pin_files['exllamav3/tokenizer/tokenizer.py'] = 'glm_vision.py'
    before = {relative: hashlib.sha256((engine / relative).read_bytes()).hexdigest()
              for relative in pin_files}
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=engine, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=engine, check=True)
    cache_patch = here / '0002-disk-cache-budget-progress.patch'
    cache_relative = 'exllamav3/generator/disk_cache.py'
    cache_before = hashlib.sha256((engine / cache_relative).read_bytes()).hexdigest()
    subprocess.run(['git', 'apply', '--check', str(cache_patch)], cwd=engine, check=True)
    subprocess.run(['git', 'apply', str(cache_patch)], cwd=engine, check=True)
    python_patches = {cache_patch.name: hashlib.sha256(cache_patch.read_bytes()).hexdigest()}
    if args.with_tokenizer:
        for source in sorted((repo / 'experiments/tokenizer').glob('*.patch')):
            subprocess.run(['git', 'apply', '--check', str(source)], cwd=engine, check=True)
            subprocess.run(['git', 'apply', str(source)], cwd=engine, check=True)
            python_patches[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    changes = {cache_relative: {'before': cache_before,
                               'after': hashlib.sha256((engine / cache_relative).read_bytes()).hexdigest()}}
    for relative, filename in pin_files.items():
        digest = hashlib.sha256((engine / relative).read_bytes()).hexdigest()
        pins = output / 'service/glm' / filename
        text = pins.read_text()
        expected = repr(relative) + ': ' + repr(before[relative])
        if text.count(expected) != 1:
            raise ValueError('Expected exact old source pin: ' + relative)
        pins.write_text(text.replace(expected, repr(relative) + ': ' + repr(digest)))
        changes[relative] = {'before': before[relative], 'after': digest}
    subprocess.run([sys.executable, str(repo / 'engine/check_engine_pins.py'), str(engine),
                    str(output / 'service/glm')], check=True)
    manifest = []
    for path in sorted(engine.rglob('*')):
        if path.is_file() and not any(p in ('.git', '__pycache__') for p in path.parts):
            manifest.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' +
                            path.relative_to(engine).as_posix() + '\n')
    (output / 'receipts/engine.sha256').write_text(''.join(manifest))
    (output / 'receipts/candidate.json').write_text(json.dumps({
        'status': 'prepared, not GPU-qualified or deployed',
        'base': 'profiles/community-four-active',
        'native_patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
        'python_patches': python_patches,
        'changed_sources': changes,
        'native_changed_files': ['exllamav3/exllamav3_ext/libtorch/gated_delta_net.cpp'],
        'required_environment': {'EXL3_KDA_GRAPH_MIN_FREE_MB': '1024'},
        'native_rebuild_required': True,
    }, indent=2) + '\n')
    print('Prepared isolated candidate; rebuild native before qualification:', output)


if __name__ == '__main__':
    main()
