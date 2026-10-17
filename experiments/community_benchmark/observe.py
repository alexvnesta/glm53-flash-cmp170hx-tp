"""Run the unmodified pinned community tool with per-agent counter observation.

Usage: python observe.py /path/to/llm_decode_bench.py -- <upstream arguments>
The source stays unmodified. Counter proxies forward the same shared list
writes and retain each agent's contribution to the upstream measurement window.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import hashlib
import sys
import time

SOURCE_SHA256 = "41a0bda35282f45a85b6eccc95858fdf876ce74f1964609fc591b08a4bc4a684"


class Counter:
    def __init__(self, original, agent, events):
        self.original, self.agent, self.events = original, agent, events

    def __getitem__(self, index):
        return self.original[index]

    def __setitem__(self, index, value):
        before = self.original[index]
        self.original[index] = value
        self.events.append((time.monotonic(), self.agent, value - before))


def summarize(events, start, duration, agents, expected):
    counts = [0] * agents
    for at, agent, delta in events:
        if start < at <= start + duration:
            counts[agent] += delta
    if sum(counts) != expected:
        raise ValueError(f"Per-agent counters do not reconcile: {counts} vs {expected}")
    return {"completion_tokens": counts, "tokens_per_second": [n / duration for n in counts],
            "measurement_start_monotonic": start, "measurement_seconds": duration,
            "source": "upstream shared_usage_token_count increments, common upstream window"}


def main():
    source = Path(sys.argv[1]).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Community benchmark source does not match pinned v0.8.0")
    arguments = sys.argv[2:]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    spec = importlib.util.spec_from_file_location("community_bench", source)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    original_stream, original_cell = bench.stream_one_request, bench.run_one_cell
    stream_signature = inspect.signature(original_stream)
    # Upstream wraps run_one_cell in an event-emission *args/**kwargs shim.
    # Bind against its measured implementation while still calling the shim.
    cell_signature = inspect.signature(bench._run_one_cell_measured)
    current = {}

    async def stream(*args, **kwargs):
        bound = stream_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        counter = bound.arguments["shared_usage_token_count"]
        if counter is not None and current:
            bound.arguments["shared_usage_token_count"] = Counter(counter, bound.arguments["index"], current["events"])
        began = time.monotonic()
        result = await original_stream(*bound.args, **bound.kwargs)
        if current:
            current["completed_streams"].append((began, bound.arguments["index"], result.total_tokens))
        return result

    async def cell(*args, **kwargs):
        bound = cell_signature.bind(*args, **kwargs)
        current.update(events=[], completed_streams=[])
        try:
            result = await original_cell(*args, **kwargs)
            state = bound.arguments["state"]
            audit = {"concurrency": result.concurrency, "context_tokens": result.context_tokens,
                     "benchmark_mode": result.benchmark_mode, "aggregate_source": result.aggregate_source,
                     "requested_duration": bound.arguments["duration"],
                     "request_count_target": result.request_count_target}
            if result.aggregate_source in ("openai_continuous_usage", "openai_completed_usage"):
                events = current["events"]
                if result.aggregate_source == "openai_completed_usage":
                    events = [(at, agent, count) for at, agent, count in current["completed_streams"]
                              if at >= state.cell_measurement_start]
                # The JSON field rounds seconds to 6 decimals. Recover the
                # original denominator from the untouched aggregate formula;
                # rounding it can exclude the final increment by <0.5 us.
                exact_seconds = result.client_output_tokens / result.aggregate_tps
                if abs(exact_seconds - result.measurement_seconds) > .00000051:
                    raise ValueError("Unexpected upstream throughput denominator")
                audit["reported_measurement_seconds_rounded"] = result.measurement_seconds
                audit.update(summarize(events, state.cell_measurement_start,
                                       exact_seconds, result.concurrency, result.client_output_tokens))
                audit["aggregate_tokens_per_second"] = result.aggregate_tps
            else:
                audit["invalid_reason"] = "Exact OpenAI usage unavailable"
            with Path("agent-audit.jsonl").open("a") as out:
                out.write(json.dumps(audit) + "\n")
            return result
        finally:
            current.clear()

    bench.stream_one_request, bench.run_one_cell = stream, cell
    sys.argv = [str(source), *arguments]
    bench.main()


if __name__ == "__main__":
    main()
