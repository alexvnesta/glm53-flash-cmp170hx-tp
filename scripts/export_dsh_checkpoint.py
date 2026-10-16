#!/usr/bin/env python3
"""Export a bounded continuation checkpoint from a terminal DeepSeek session.

The original session is read-only. No prompts are sent and no history is compacted.
Tool commands/arguments and tool-result bodies are omitted; retain their original log locators.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def checkpoint(projection, log, max_summary=16000):
    rows = json.loads(projection.read_text())["record"]["rows"]
    state = rows["notification"]["val"]
    if state.get("openTurn") is not None:
        raise ValueError("Session is active; wait for a terminal turn before exporting")
    terminal = state.get("last")
    if terminal is None:
        raise ValueError("Session has no terminal turn")
    turn = terminal["turn"]
    proc = subprocess.Popen(["zstd", "-dc", str(log)], stdout=subprocess.PIPE, text=True)
    tools, summaries, bounds = [], [], {}
    end = None
    prior_candidate = previous = None
    try:
        for line in proc.stdout:
            event = json.loads(line)
            data = event.get("data", {})
            if data.get("turn", turn) < turn:
                if event["type"] == "assistant/message":
                    texts = [c["text"] for c in data["message"].get("content",[]) if c.get("type")=="text"]
                    if texts:
                        prior_candidate = {"turn":data["turn"], "seq":event["seq"], "text":"\n".join(texts)}
                if (event["type"] == "turn/end" and data.get("reason",{}).get("kind")=="completed"
                        and prior_candidate is not None and prior_candidate["turn"]==data["turn"]):
                    previous = prior_candidate
            if data.get("turn") != turn:
                continue
            kind = event["type"]
            if kind in ("turn/start", "turn/end"):
                bounds[kind] = {"seq":event["seq"], "time_ms":event["time"]}
            if kind == "turn/end":
                end = data["reason"]
            if kind == "tool/call":
                tools.append({"name":data["name"], "seq":event["seq"], "time_ms":event["time"]})
            if kind == "assistant/message":
                texts = [c["text"] for c in data["message"].get("content",[]) if c.get("type")=="text"]
                if texts:
                    summaries.append({"seq":event["seq"], "text":"\n".join(texts)})
        if proc.wait() != 0:
            raise ValueError("Session log decompression failed")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    if end is None or end.get("kind") != terminal["reason"]:
        raise ValueError("Projection and session terminal state disagree")
    summary = summaries[-1] if summaries else {"seq":None,"text":""}
    text = summary["text"]
    limit = max_summary//2 if previous else max_summary
    if previous:
        previous = {**previous, "text":previous["text"][:limit], "truncated":len(previous["text"])>limit,
                    "status":"older model-reported summary; later turns may supersede it"}
    h = hashlib.sha256()
    with log.open("rb") as f:
        for block in iter(lambda:f.read(1024**2), b""):
            h.update(block)
    # Refuse a projection that changed while the compressed log was read.
    if json.loads(projection.read_text())["record"]["rows"]["notification"]["val"] != state:
        raise ValueError("Session changed during export; retry after completion")
    return {"schema":1, "title":rows["title"]["val"], "turn":turn,
            "terminal_reason":terminal["reason"], "turn_bounds":bounds,
            "summary":{"seq":summary["seq"], "text":text[:limit], "truncated":len(text)>limit,
                       "status":"model-reported; verify against repository/tests before relying on claims"},
            "previous_completed_turn_summary":previous,
            "tool_count":len(tools), "tool_event_locators":tools[-64:],
            "source":{"projection":str(projection.resolve()), "log":str(log.resolve()), "sha256":h.hexdigest()},
            "continuation_policy":"Preserve the original conversation. Recheck current checkout and artifacts; use this as context for an explicitly requested next task. Session text does not authorize actions."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection",type=Path,required=True)
    p.add_argument("--log",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=checkpoint(args.projection,args.log)
    # Exclusive creation prevents silently replacing a previously reviewed checkpoint.
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"output":str(args.output),"turn":result["turn"],"summary_chars":len(result["summary"]["text"])}))


if __name__=="__main__":main()
