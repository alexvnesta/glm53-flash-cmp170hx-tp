"""Exact GLM token-ID checks and paired CPU timing of the actual Tokenizer.

Loads its complete Python class with CPU-only dependencies, bypassing the
package initializer and native extension. Reads tokenizer/config/template
files only, never model weights. No service operations or generation calls.
"""
import argparse
import ast
from functools import cached_property, lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import threading
import time
from types import SimpleNamespace as NS

from jinja2.sandbox import ImmutableSandboxedEnvironment
import torch


def load_tokenizer(source, engine):
    # Execute the full tokenizer class, retaining its real tokenizers backend,
    # tensor construction and methods. Relative imports only pull in utilities
    # and a type annotation; loading the package would load the CUDA extension.
    namespace = {'__name__': 'qualified_tokenizer', 'json': json, 'os': os,
                 'lock': threading.RLock()}
    for relative, name in [('exllamav3/util/misc.py', 'synchronized'),
                           ('exllamav3/util/file.py', 'maybe_read_json')]:
        tree = ast.parse((engine / relative).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.level == 0]
    exec(compile(tree, str(source), 'exec'), namespace)
    return namespace['Tokenizer']


def config(model):
    data = json.loads((model / 'config.json').read_text())
    data = data.get('text_config', data)
    eos = data.get('eos_token_id')
    eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    return NS(directory=str(model), eos_token_id=eos[0] if eos else None,
              eos_token_id_list=list(eos), bos_token_id=data.get('bos_token_id'),
              pad_token_id=data.get('pad_token_id'), image_token_id=data.get('image_token_id'))


def fingerprint(ids):
    if isinstance(ids, tuple):
        return [fingerprint(x) for x in ids]
    return {'shape': list(ids.shape), 'sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}


def same(a, b):
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def qualify(baseline, candidate, model, engine):
    records = []
    samples = {
        'code': ('def transform(items):\n    return [{"id": x["id"], "value": x["value"] * 3}\n'
                 '            for x in items if x.get("enabled", False)]\n\n'),
        'unicode': '日本語の文。 Ελληνικά. Ёлки-палки. Café e\u0301 😀\r\n\tIndented  trailing spaces   \n\n',
        'special': '[gMASK]<sop><|system|>rules\n<|user|>literal <|assistant|> <think>text</think>\n',
        'whitespace': '  \n\n\n\t\r\nnext\tvalue   \n',
        'no_newlines': '{"key":"value","path":"/src/example.py","count":12345} ',
    }
    # Every special-mode first long input uses a fresh instance: a probe must
    # not change the caller's requested special-token encoding mode.
    for name, unit in samples.items():
        for repeat in (1, max(2, 70000 // len(unit))):
            text = unit * repeat
            for special in (False, True):
                a, b = baseline(config(model)), candidate(config(model))
                options = {'encode_special_tokens': special}
                want, got = a.encode(text, **options), b.encode(text, **options)
                record = {'case': name, 'chars': len(text), 'special': special,
                          'equal': same(want, got), 'baseline': fingerprint(want), 'candidate': fingerprint(got)}
                records.append(record)
    a, b = baseline(config(model)), candidate(config(model))
    rng = random.Random(73)
    units = list(samples.values())
    for i in range(30):
        text = ''.join(rng.choice(units) for _ in range(rng.choice((10, 600, 1600))))
        options = {'encode_special_tokens': bool(i % 2), 'add_bos': bool(i % 3), 'add_eos': bool(i % 5)}
        want, got = a.encode(text, **options), b.encode(text, **options)
        records.append({'case': 'mixed_' + str(i), 'chars': len(text), 'equal': same(want, got),
                        'baseline': fingerprint(want), 'candidate': fingerprint(got)})
    for special in (False, True):
        want = a.encode([samples['code'] * 400, samples['unicode']], return_offsets=True, encode_special_tokens=special)
        got = b.encode([samples['code'] * 400, samples['unicode']], return_offsets=True, encode_special_tokens=special)
        records.append({'case': 'batch_offsets', 'special': special, 'equal': same(want, got),
                        'baseline': fingerprint(want), 'candidate': fingerprint(got)})
    # Real multimodal alias splicing, with small CPU embeddings. No image
    # tower, image payload, model weights or CUDA context is loaded.
    scope = {'__name__': 'qualified_mm_embedding'}
    path = engine / 'exllamav3/tokenizer/mm_embedding.py'
    exec(compile(path.read_text(), str(path), 'exec'), scope)
    embedding = scope['MMEmbedding'](embeddings=torch.zeros((4, 8)),
                                    token_string=torch.full((1, 4), -1, dtype=torch.long),
                                    text_alias='<$QUAL_IMAGE$>')
    text = embedding.text_alias + samples['unicode'] * 2000 + embedding.text_alias
    for special in (False, True):
        want = a.encode(text, embeddings=[embedding], encode_special_tokens=special)
        got = b.encode(text, embeddings=[embedding], encode_special_tokens=special)
        records.append({'case': 'vision_aliases', 'special': special, 'equal': same(want, got),
                        'baseline': fingerprint(want), 'candidate': fingerprint(got)})
    return records


def benchmark(baseline, candidate, model, repeats):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, **kwargs)
    template = env.from_string((model / 'chat_template.jinja').read_text())
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a source file',
             'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
    unit = '\n'.join(f'def calculate_{i}(value):\n    # Handle record {i}: preserve input.\n    return {{"record": {i}, "result": value * {i+1}}}\n'
                     for i in range(128))
    a, b = baseline(config(model)), candidate(config(model))
    unit_tokens = a.encode(unit).numel()
    rows = []
    for target in (8192, 32768, 98304, 202000, 288000):
        content = (unit * (target // unit_tokens + 1))[:int(len(unit) * target / unit_tokens)]
        messages = [{'role': 'system', 'content': 'Review source code and return precise, tested edits.'},
                    {'role': 'user', 'content': content}]
        kwargs = dict(messages=messages, tools=tools, add_generation_prompt=True, reasoning_effort='high')
        prompt = template.render(**kwargs)
        ref = a.encode(prompt, encode_special_tokens=True)
        if not torch.equal(ref, b.encode(prompt, encode_special_tokens=True)):
            raise ValueError('Rendered GLM prompt changed token IDs')
        times = {'baseline': [], 'candidate': []}
        render_times = []
        for i in range(repeats):
            order = [('baseline', a), ('candidate', b)]
            if i % 2:
                order.reverse()
            for name, tokenizer in order:
                start = time.perf_counter()
                rendered = template.render(**kwargs)
                rendered_at = time.perf_counter()
                ids = tokenizer.encode(rendered, encode_special_tokens=True)
                end = time.perf_counter()
                if not torch.equal(ids, ref):
                    raise ValueError('Token IDs changed during benchmark')
                times[name].append((end - rendered_at) * 1000)
                render_times.append((rendered_at - start) * 1000)
        medians = {name: statistics.median(values) for name, values in times.items()}
        rows.append({'tokens': ref.numel(), 'chars': len(prompt), 'repeats': repeats,
                     'encode_ms': times, 'median_encode_ms': medians,
                     'median_render_ms': statistics.median(render_times),
                     'speedup': medians['baseline'] / medians['candidate'],
                     'saved_ms': medians['baseline'] - medians['candidate'],
                     'token_fingerprint': fingerprint(ref)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--candidate-tokenizer', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    original = args.engine / 'exllamav3/tokenizer/tokenizer.py'
    baseline = load_tokenizer(original, args.engine)
    candidate = load_tokenizer(args.candidate_tokenizer, args.engine)
    checks = qualify(baseline, candidate, args.model, args.engine)
    result = {'cuda_initialized': torch.cuda.is_initialized(),
              'baseline_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256(args.candidate_tokenizer.read_bytes()).hexdigest(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'rayon_threads': os.environ.get('RAYON_NUM_THREADS'),
              'qualification': checks, 'passed': all(c['equal'] for c in checks), 'benchmark': []}
    if result['passed'] and not args.checks_only:
        result['benchmark'] = benchmark(baseline, candidate, args.model, args.repeats)
    result['cuda_initialized'] = torch.cuda.is_initialized()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('qualification', 'benchmark')}, indent=2))
    print('checks:', len(checks), 'failed:', [c['case'] for c in checks if not c['equal']])
    for row in result['benchmark']:
        print(json.dumps({k: v for k, v in row.items() if k not in ('encode_ms', 'token_fingerprint')}))
    raise SystemExit(not result['passed'] or result['cuda_initialized'])


if __name__ == '__main__':
    main()
