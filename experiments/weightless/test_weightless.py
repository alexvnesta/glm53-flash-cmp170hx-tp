"""CPU format and actual engine-hook tests; GPU qualification is separate."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest

from glm_weightless import load_vector, make_projector, WIDTH
from prepare_engine import prepare, INITIALIZE, HOOK, IMPORT, RETURN

REVISION = "a" * 40
ROOT = Path(__file__).resolve().parents[2]


def write_vector(tmp_path, changes=None, tensors=None):
    path = tmp_path / "vector.gguf"
    metadata = {"glp.mode": "project", "glp.spec_version": 1,
                "glp.hook_point": "residual_stream_post_layer",
                "general.base_model.0.version": REVISION,
                "glp.layer_ids_zero_based": "3"}
    metadata.update(changes or {})
    writer = gguf.GGUFWriter(str(path), "controlvector")
    for k, v in metadata.items():
        if v is None:
            continue
        if type(v) is int:
            writer.add_uint32(k, v)
        else:
            writer.add_string(k, v)
    if tensors is None:
        tensors = {"direction.3": np.arange(1, WIDTH + 1, dtype=np.float32)}
    for k, v in tensors.items():
        writer.add_tensor(k, v)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_gguf_round_trip_and_layer_mapping(tmp_path):
    path, digest = write_vector(tmp_path)
    vectors = load_vector(path, digest, REVISION)
    assert set(vectors) == {3}  # direction.3 applies to layer 3, not layer 2.
    assert abs(np.linalg.norm(vectors[3]) - 1) < 1e-6
    env = {"GLM_WEIGHTLESS_VECTOR": str(path), "GLM_WEIGHTLESS_SHA256": digest,
           "GLM_WEIGHTLESS_BASE_REVISION": REVISION, "GLM_WEIGHTLESS_ALPHA": "0"}
    projector = make_projector(env)
    x, params = object(), {"hc_pending": object()}
    assert projector(SimpleNamespace(attn_hc=True, layer_idx=3), x, params) is x
    assert "hc_pending" in params  # alpha zero preserves deferred state too.


@pytest.mark.parametrize("changes", [
    {"glp.mode": None}, {"glp.mode": "add"}, {"glp.mode": "unknown"},
    {"glp.spec_version": 2}, {"glp.spec_version": None},
    {"glp.hook_point": "ffn_write"}, {"glp.hook_point": None},
    {"general.base_model.0.version": "b" * 40},
    {"glp.layer_ids_zero_based": "2"}, {"glp.layer_ids_zero_based": None},
    {"glp.rank": 2}, {"glp.dir_scales": "1"}, {"glp.layer_scales": "3:1"},
    {"dspark.mode": "project"},
])
def test_incompatible_metadata_fails_closed(tmp_path, changes):
    path, digest = write_vector(tmp_path, changes)
    with pytest.raises(ValueError):
        load_vector(path, digest, REVISION)


@pytest.mark.parametrize("name,value", [
    ("direction.0", np.ones(WIDTH, dtype=np.float32)),
    ("direction.45", np.ones(WIDTH, dtype=np.float32)),
    ("direction.3.1", np.ones(WIDTH, dtype=np.float32)),
    ("direction.3", np.ones((4, 4096), dtype=np.float32)),
    ("direction.3", np.ones(4096, dtype=np.float32)),
    ("direction.3", np.ones(WIDTH, dtype=np.float16)),
    ("direction.3", np.zeros(WIDTH, dtype=np.float32)),
    ("direction.3", np.full(WIDTH, np.nan, dtype=np.float32)),
    ("direction.3", np.full(WIDTH, np.inf, dtype=np.float32)),
])
def test_bad_tensors_fail_closed(tmp_path, name, value):
    path, digest = write_vector(tmp_path, tensors={name: value})
    with pytest.raises(ValueError):
        load_vector(path, digest, REVISION)


def test_integrity_and_disabled_dependency_boundary(tmp_path):
    assert make_projector({}) is None
    path, digest = write_vector(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_vector(path, "0" * 64, REVISION)
    with pytest.raises(ValueError, match="pin the base"):
        load_vector(path, digest, "unknown")
    for alpha in ["nan", "inf", "-1", "2.1"]:
        with pytest.raises(ValueError, match="ALPHA"):
            make_projector({"GLM_WEIGHTLESS_VECTOR": str(path), "GLM_WEIGHTLESS_ALPHA": alpha})


class Tensor(np.ndarray):
    def half(self):
        return self.astype(np.float16)

    def mean(self, dim):
        return np.asarray(self).mean(axis=dim).view(Tensor)

    def clamp_(self, lower, upper):
        np.clip(self, lower, upper, out=self)
        return self


def engine_forward(projector):
    """Execute the actual pinned TransformerBlock.forward with CPU collaborators.

    This checks the integration seam rather than copying its order into a mock.
    """
    text = (ROOT / "engine-src/exllamav3/modules/transformer.py").read_text()
    patched = text.replace(IMPORT, IMPORT + INITIALIZE).replace(RETURN, HOOK + RETURN, 1)
    tree = ast.parse(patched)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TransformerBlock")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    fn.decorator_list = []
    namespace = {"torch": SimpleNamespace(Tensor=Tensor, half=np.float16),
                 "to2": lambda x, *args: x, "_weightless_projector": projector}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "actual-transformer-forward", "exec"), namespace)
    return namespace["forward"]


def block():
    hc = SimpleNamespace(mix_norm=lambda x, *args: (None, None, x.mean(dim=2), True),
                         apply_=lambda x, *args: x + 1)
    return SimpleNamespace(layer_idx=3, attn_hc=hc, mlp_hc=hc, attn_norm=None,
                           mlp_norm=None, attn=SimpleNamespace(forward=lambda x, p: x),
                           mlp=SimpleNamespace(forward=lambda x, p: x),
                           attn_resid_scalar=None, mlp_resid_scalar=None,
                           layer_scalar_f=None, out_dtype=None)


def test_actual_hook_preserves_pre_steering_dflash_capture():
    observed = []
    def project(b, x, params):
        observed.append(params["export_states"][0].copy())
        x += 10
        return x
    x = np.zeros((1, 2, 4, 4096), dtype=np.float32).view(Tensor)
    params = {"export_state_layers": [3]}
    result = engine_forward(project)(block(), x, params)
    assert np.all(result == 12)
    assert np.all(observed[0] == 2)
    assert np.all(params["export_states"][0] == 2)


def test_disabled_forward_matches_and_prefill_early_exit_is_not_steered():
    x = np.zeros((1, 2, 4, 4096), dtype=np.float32).view(Tensor)
    assert np.all(engine_forward(None)(block(), x, {}) == 2)
    def unexpected(*args):
        pytest.fail("early-exit prefill must not invoke a post-layer hook")
    assert engine_forward(unexpected)(block(), x, {"prefill": True}) is x


def test_prepare_is_separate_and_rejects_unknown_source(tmp_path):
    source = ROOT / "engine-src"
    destination = tmp_path / "experimental"
    before = hashlib.sha256((source / "exllamav3/modules/transformer.py").read_bytes()).hexdigest()
    result = prepare(source, destination)
    assert result != before
    assert hashlib.sha256((source / "exllamav3/modules/transformer.py").read_bytes()).hexdigest() == before
    ast.parse((destination / "exllamav3/modules/transformer.py").read_text())
    with pytest.raises(ValueError, match="new tree"):
        prepare(source, destination)
    with pytest.raises(ValueError, match="qualified source"):
        prepare(destination, tmp_path / "another")
