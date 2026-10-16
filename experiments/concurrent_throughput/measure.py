"""Measure serial versus genuinely batched GLM requests on an idle service.

No settings or services are changed. Other traffic or identity drift cancels
this client's streams. This is a bounded load probe, not a quality benchmark.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import time

import httpx


class OwnedRequests(set):
    def __init__(self):
        super().__init__()
        self.pending = {}


class ProbeResponseFailure(ValueError):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


def failure_details(error):
    details = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, BaseExceptionGroup):
        details["exceptions"] = [failure_details(child) for child in error.exceptions]
    if isinstance(error, ProbeResponseFailure):
        details["response"] = error.response
    return details


def cases(size=64, workload="structured"):
    pair = [{
        "id": f"array-{start}",
        "prompt": (f"Return only a JSON array of all integers from {start} "
                   f"through {start + size - 1}, inclusive, in ascending order. "
                   "No markdown or explanation."),
        "expected": list(range(start, start + size)),
    } for start in (100, 400)]
    if workload == "mixed":
        pair[0] = {
            "id": "compress-runs-code", "validator": "code",
            "prompt": (
                "Implement Python function compress_runs(values). It returns the consecutive "
                "runs of an integer list as a list of (value, count) pairs. Preserve order "
                "and do not mutate the input. Empty input returns []. Use only ordinary "
                "Python loops, lists, indexing and append; no imports. Return only a JSON "
                "object with one key, code, containing the source as a string."
            ),
        }
    return pair


def identity(health):
    return {
        "native_sha256": health.get("native_extension", {}).get("sha256"),
        "profile_digest": health.get("profile_digest"),
        "weightless": health.get("weightless"),
        "cache_tokens": health.get("cache_tokens"),
        "max_active_sequences": health.get("max_active_sequences"),
        "draft_num_tokens": health.get("draft_num_tokens"),
        "hybrid_draft_tokens": health.get("hybrid_drafting", {}).get("hybrid_draft_tokens"),
    }


async def health(client):
    response = await client.get("/health", timeout=3)
    response.raise_for_status()
    return response.json()


def compact(snapshot, owned):
    metrics = snapshot.get("request_metrics", {})
    return {
        "at": time.time(),
        "busy": snapshot.get("busy"),
        "active_sequences": snapshot.get("active_sequences"),
        "pending_sequences": snapshot.get("pending_sequences"),
        "request_queue": snapshot.get("request_queue"),
        "concurrent_ring": snapshot.get("concurrent_ring"),
        "own_metrics": [r for r in metrics.get("recent", []) if r["id"] in owned],
        "own_active": [r for r in metrics.get("active", []) if r["id"] in owned],
    }


def require_idle(snapshot):
    if not all(identity(snapshot)[key] for key in ("native_sha256", "profile_digest")):
        raise RuntimeError("Serving identity unavailable; probe refused")
    if snapshot.get("status") != "ok" or snapshot.get("busy") is not False:
        raise RuntimeError("Service unhealthy or busy; probe yields")
    if (snapshot.get("active_sequences", 0) or snapshot.get("pending_sequences", 0)
            or snapshot.get("concurrent_ring", {}).get("active_ring_leases", 0)):
        raise RuntimeError("Native jobs or leases remain; probe yields")
    if snapshot.get("request_metrics", {}).get("active"):
        raise RuntimeError("Another request is active; probe yields")
    if snapshot.get("request_queue", {}).get("waiting", 0):
        raise RuntimeError("Other requests are waiting; probe yields")
    if snapshot.get("switcher", {}).get("inflight", 0):
        raise RuntimeError("Router has inflight requests; probe yields")
    if snapshot.get("max_active_sequences") != 2:
        raise RuntimeError("This bounded probe requires the qualified two-slot profile")


async def stream(client, case, effort, owned):
    started = time.perf_counter()
    content, reasoning = [], []
    result = {"case": case["id"], "ttft_s": None, "first_visible_s": None}
    payload = {
        "model": "GLM-5.3-Flash", "reasoning_effort": effort,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0, "max_tokens": 768, "stream": True,
        "stream_options": {"include_usage": True},
    }
    usage, finished, timings = None, None, None
    if isinstance(owned, OwnedRequests):
        owned.pending[case["id"]] = time.monotonic()
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("id"):
                owned.add(event["id"])
                if isinstance(owned, OwnedRequests):
                    owned.pending.pop(case["id"], None)
                result["request_id"] = event["id"]
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text, thought = delta.get("content") or "", delta.get("reasoning_content") or ""
                elapsed = time.perf_counter() - started
                if (text or thought) and result["ttft_s"] is None:
                    result["ttft_s"] = elapsed
                if text and result["first_visible_s"] is None:
                    result["first_visible_s"] = elapsed
                content.append(text)
                reasoning.append(thought)
                finished = choice.get("finish_reason") or finished
    result.update(
        finish_s=time.perf_counter() - started, usage=usage, timings=timings,
        finish_reason=finished, answer="".join(content),
        reasoning_characters=sum(map(len, reasoning)),
    )
    try:
        if case.get("validator") == "code":
            from code_oracle import validate
            result["correct"] = finished == "stop" and await asyncio.to_thread(validate, result["answer"])
        else:
            result["correct"] = finished == "stop" and json.loads(result["answer"]) == case["expected"]
    except (ValueError, SyntaxError, OSError, TimeoutError):
        result["correct"] = False
    if not result["correct"]:
        raise ProbeResponseFailure("Known-answer failure: " + case["id"], result)
    if not usage or not isinstance(usage.get("completion_tokens"), int):
        raise ProbeResponseFailure("Final token usage missing; cannot compare throughput", result)
    return result


async def monitor(client, before, owned, snapshots):
    known = {r["id"] for r in before["request_metrics"]["recent"]}
    unknown_since = {}
    while True:
        snapshot = await health(client)
        if identity(snapshot) != identity(before):
            raise RuntimeError("Serving identity changed; probe yields")
        if snapshot.get("status") != "ok":
            raise RuntimeError("Service health problem; probe yields")
        metrics = snapshot.get("request_metrics", {})
        if any(r["id"] not in known | owned for r in metrics.get("recent", [])):
            raise RuntimeError("Another client's request completed; probe yields")
        unresolved = []
        for record in metrics.get("active", []):
            ident = record["id"]
            if ident in owned:
                continue
            # Our ID becomes known with the initial SSE role event, before
            # model output. Allow only a short HTTP registration race.
            first = unknown_since.setdefault(ident, time.monotonic())
            unresolved.append(ident)
            if time.monotonic() - first > .5:
                raise RuntimeError("Request ownership unresolved; cancelling probe streams")
        queue = snapshot.get("request_queue", {})
        waiting = queue.get("waiting", 0) or snapshot.get("switcher", {}).get("waiting", 0)
        pending = getattr(owned, "pending", {})
        registration_race = (
            unresolved and len(unresolved) <= len(pending)
            and all(time.monotonic() - stamp <= .5 for stamp in pending.values())
        )
        if waiting and not registration_race:
            raise RuntimeError("Another client may be waiting; cancelling probe streams")
        snapshots.append(compact(snapshot, owned))
        await asyncio.sleep(.15)


async def wave(client, before, pair, effort, concurrent):
    owned, snapshots = OwnedRequests(), []

    async def work():
        if concurrent:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(stream(client, c, effort, owned)) for c in pair]
            return [task.result() for task in tasks]
        return [await stream(client, c, effort, owned) for c in pair]

    worker = asyncio.create_task(work())
    watcher = asyncio.create_task(monitor(client, before, owned, snapshots))
    started = time.perf_counter()
    try:
        done, _ = await asyncio.wait((worker, watcher), timeout=90,
                                     return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            await watcher  # Raises with the verified reason for yielding.
        if worker not in done:
            raise TimeoutError("Probe exceeded its 90-second wave budget")
        rows = await worker
        wall = time.perf_counter() - started
    except BaseException as exc:
        exc.probe_context = {"owned_ids": sorted(owned), "snapshots": snapshots}
        raise
    finally:
        worker.cancel()
        watcher.cancel()
        await asyncio.gather(worker, watcher, return_exceptions=True)
    deadline = time.monotonic() + 2
    while True:
        after = await health(client)
        new = [r for r in after["request_metrics"]["recent"] if r["id"] in owned]
        if len(new) == len(rows):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Final per-request metrics missing; comparison excluded")
        await asyncio.sleep(.05)
    if any(r["id"] not in {r["id"] for r in before["request_metrics"]["recent"]} | owned
           for r in after["request_metrics"]["recent"]):
        raise RuntimeError("Other traffic arrived at completion; comparison excluded")
    return {
        "mode": "concurrent" if concurrent else "serial", "effort": effort,
        "batch_finish_s": wall,
        "aggregate_completion_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in rows) / wall,
        "median_request_finish_s": statistics.median(r["finish_s"] for r in rows),
        "median_request_ttft_s": statistics.median(r["ttft_s"] for r in rows),
        "requests": rows, "server_metrics": new,
        "ring_before": before.get("concurrent_ring"),
        "ring_after": after.get("concurrent_ring"), "snapshots": snapshots,
    }


def compare(serial, parallel):
    b, a = parallel["ring_before"], parallel["ring_after"]
    truly_batched = (a["batched_verifier_rounds"] > b["batched_verifier_rounds"]
                     and a["batched_draft_rounds"] > b["batched_draft_rounds"])
    return {
        "serial_batch_finish_s": serial["batch_finish_s"],
        "concurrent_batch_finish_s": parallel["batch_finish_s"],
        "batch_speedup": serial["batch_finish_s"] / parallel["batch_finish_s"],
        "real_draft_and_target_batch_observed": truly_batched,
        "all_correct": all(r["correct"] for w in (serial, parallel) for r in w["requests"]),
        "serial_completion_tokens": sum(r["usage"]["completion_tokens"] for r in serial["requests"]),
        "concurrent_completion_tokens": sum(r["usage"]["completion_tokens"] for r in parallel["requests"]),
        "cached_tokens": sum(r.get("cached_tokens", 0) for w in (serial, parallel) for r in w["server_metrics"]),
    }


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    pair = cases(args.size, args.workload)
    report = {
        "scope": "Two independent bounded synthetic tasks; not a general multi-agent or coding benchmark",
        "fixture_sha256": hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest(),
        "settings": vars(args) | {"output": str(args.output)}, "waves": [], "comparisons": [],
    }
    path = args.output / "results.json"
    async with httpx.AsyncClient(base_url=args.base_url, timeout=httpx.Timeout(30, connect=3)) as client:
        try:
            initial = await health(client)
            require_idle(initial)
            report["identity"] = identity(initial)
            # Stable idle check; later monitor checks also yield on arrivals.
            await asyncio.sleep(2)
            require_idle(await health(client))
            for effort in args.effort:
                for repeat in range(args.repeats):
                    waves = {}
                    for concurrent in ([False, True] if repeat % 2 == 0 else [True, False]):
                        before = await health(client)
                        require_idle(before)
                        item = await wave(client, before, pair, effort, concurrent)
                        item["repeat"] = repeat
                        report["waves"].append(item)
                        waves[concurrent] = item
                        path.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps({k: item[k] for k in (
                            "mode", "effort", "repeat", "batch_finish_s", "aggregate_completion_tokens_per_s"
                        )}), flush=True)
                        await asyncio.sleep(.3)
                    comparison = {"effort": effort, "repeat": repeat,
                                  **compare(waves[False], waves[True])}
                    report["comparisons"].append(comparison)
                    print(json.dumps(comparison), flush=True)
            report["completed"] = True
        except Exception as exc:
            report.update(completed=False, error=f"{type(exc).__name__}: {exc}")
            report["failure_details"] = failure_details(exc)
            report["interrupted_wave"] = getattr(exc, "probe_context", None)
            print(report["error"], flush=True)
        finally:
            path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["completed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18012")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effort", nargs="+", choices=("low", "high"), default=["low", "high"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--workload", choices=("structured", "mixed"), default="structured")
    args = parser.parse_args()
    if args.repeats not in (1, 2, 3) or not 16 <= args.size <= 96:
        parser.error("Bounded repeats 1..3 and size 16..96 required")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
