"""Check actual GLM token masks and per-job parser state without CUDA."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        p.error("Run with CUDA_VISIBLE_DEVICES= to preserve live GPU consumers")
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "glm"), str(args.engine)]
    import torch
    assert not torch.cuda.is_initialized()
    from exllamav3 import Config, Tokenizer
    import glm_json_output as jo
    tokenizer = Tokenizer.from_config(Config.from_directory(str(args.model_dir)))
    assert tokenizer.single_id("</think>") >= 0
    results = []

    def ids(text):
        return tokenizer.encode(text, encode_special_tokens=True).view(-1).tolist()

    def allowed(f, token):
        words = f.get_next_logit_mask().view(-1).numpy().view("uint32")
        return bool((int(words[token >> 5]) >> (token & 31)) & 1)

    def feed_text(f, text):
        for token in ids(text):
            if f.is_active:
                assert allowed(f, token), (text, token)
            f.feed(token)

    samples = ["{}", '{"code":"def f(x):\\n    return x + 1"}',
               '{"nested":[null,true,false,-12.5,{"unicode":"β雪","quote":"\\\""}]}']
    for text in samples:
        f = jo.build_filter(tokenizer)
        assert not f.is_active
        feed_text(f, 'Reasoning can contain ```json and [anything].')
        f.feed(tokenizer.single_id("</think>"))
        assert f.is_active
        for bad in ["```", "[", "NaN", "hello"]:
            assert not allowed(f, ids(bad)[0]), bad
        feed_text(f, text)
        for bad in ["```", "hello"]:
            assert not allowed(f, ids(bad)[0]), bad
        jo.validate_complete(text)
        results.append({"kind": "object_and_reasoning_trigger", "text": text, "passed": True})

    a, b = jo.build_filter(tokenizer), jo.build_filter(tokenizer)
    a.feed(tokenizer.single_id("</think>")); b.feed(tokenizer.single_id("</think>"))
    feed_text(a, '{"left":')
    feed_text(b, '{"right":[')
    feed_text(a, '1}')
    feed_text(b, '2,3]}')
    assert a._matcher is not b._matcher
    results.append({"kind": "interleaved_independent_parser_states", "passed": True})
    a.reset()
    a.is_active = True
    feed_text(a, '{"fresh":true}')
    results.append({"kind": "reset_clears_previous_state", "passed": True})
    assert not torch.cuda.is_initialized()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "cuda_initialized": False,
        "llguidance_version": __import__('importlib.metadata', fromlist=['version']).version('llguidance'),
        "cases": results}, indent=2) + "\n")
    print("PASS", len(results), "actual-tokenizer grammar cases; CUDA remains uninitialized")


if __name__ == "__main__":
    main()
