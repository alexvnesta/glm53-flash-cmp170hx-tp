"""Startup custody for the optional isolated Weightless engine experiment."""
import json
import os
from pathlib import Path


def runtime_status(expected_devices, environ=None, proc_root=Path('/proc')):
    env = os.environ if environ is None else environ
    if not env.get('GLM_WEIGHTLESS_VECTOR'):
        return {'enabled': False}
    from exllamav3.modules import transformer
    projector = getattr(transformer, '_weightless_projector', None)
    if projector is None:
        raise ValueError('Weightless requested but the experimental engine hook is absent')
    valid = set()
    root = env.get('GLM_WEIGHTLESS_RECEIPTS')
    if root:
        for path in Path(root).glob('numerics-*.json'):
            data = json.loads(path.read_text())
            pid = int(data['pid'])
            if pid != os.getpid():
                try:
                    status = (proc_root / str(pid) / 'status').read_text().splitlines()
                    parent = next(int(s.split()[1]) for s in status if s.startswith('PPid:'))
                    if parent != os.getpid():
                        continue
                except (OSError, StopIteration):
                    continue
            if data.get('passed') is True and len(data.get('cases', [])) == 18:
                valid.add(data['device'])
    expected = {f'cuda:{int(d)}' for d in expected_devices}
    if env.get('GLM_WEIGHTLESS_VALIDATE') == '1' and valid != expected:
        raise ValueError(f'Weightless GPU numerical checks incomplete: {sorted(valid)} vs {sorted(expected)}')
    return {'enabled': True, 'mode': 'project', 'alpha': projector.alpha,
            'layers': len(projector.vectors), 'width': 16384,
            'vector_sha256': env['GLM_WEIGHTLESS_SHA256'],
            'base_revision': env['GLM_WEIGHTLESS_BASE_REVISION'],
            'numerics_verified_devices': sorted(valid)}
