#!/usr/bin/env python3
"""Check that GPU peer-to-peer (P2P) transfers between the two cards arrive intact, and that a 2-process NCCL all_reduce returns the right
sum, before you run the server with the NCCL tensor-parallel backend (TP_BACKEND=nccl). Run it while nothing else uses the GPUs (1-2 minutes).

Idea credit: Morrowmake/glm53-flash-cmp170hx-recipe (MIT) verifies P2P by content on the CMP 170HX before enabling it, because a patched-driver
peer path can silently corrupt particular transfer sizes. This file is our own implementation of that idea.

  1. copy test   both directions and sizes 128 KiB +-1, 512 KiB +-1, 1 MiB, 8 MiB, 32 MiB (plus repeated 32 MiB copies): random bytes on the
                 source GPU, device-to-device copy, compared on the CPU with the source.
  2. NCCL test   two processes (one per GPU), torch.distributed backend nccl, random fp32 and bf16 tensors of several sizes, all_reduce(sum)
                 compared with the CPU sum. NCCL_DEBUG=INFO is on; the transport (P2P/CUMEM, P2P/IPC or SHM) is reported.
Exit 0 and "P2P/NCCL content check PASSED": NCCL over P2P is safe to try. Exit 1 ("FAILED"): keep TP_BACKEND=native. Exit 2: could not run.

    .venv/bin/python scripts/p2p_check.py [--out p2p-check] [--devices 0,1]
"""
import argparse, json, os, re, socket, subprocess, sys, time

KIB, MIB = 1024, 1024 * 1024
COPY_SIZES = [128 * KIB - 1, 128 * KIB, 128 * KIB + 1, 512 * KIB - 1, 512 * KIB, 512 * KIB + 1, 1 * MIB, 8 * MIB, 32 * MIB]
NCCL_ELEMS = [1, 6, 4096, 65536, 1000003, 4 * MIB, 8 * MIB]
REPS = 3


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------------- copy test
def copy_test(torch, devs, seed, cpu, big_repeats, inject=""):
    res = {"peer": {}, "cases": [], "mismatches": 0, "cases_run": 0}
    for i, j in ((devs[0], devs[1]), (devs[1], devs[0])):
        res["peer"][f"{i}->{j}"] = True if cpu else bool(torch.cuda.can_device_access_peer(i, j))
    for src, dst in ((devs[0], devs[1]), (devs[1], devs[0])):
        sdev, ddev = ("cpu", "cpu") if cpu else (f"cuda:{src}", f"cuda:{dst}")
        reps_for = lambda n: REPS + (big_repeats if n == 32 * MIB else 0)
        for n in COPY_SIZES:
            bad_here = 0
            for rep in range(reps_for(n)):
                g = torch.Generator(device=sdev).manual_seed(seed + 7919 * n + 131 * rep + 17 * src)
                a = torch.randint(0, 256, (n,), dtype=torch.uint8, device=sdev, generator=g)
                ref = a.cpu().clone()
                b = torch.empty(n, dtype=torch.uint8, device=ddev)
                b.fill_(0xA5)                           # poison: a copy that did not happen cannot match random data
                if cpu:
                    b.copy_(a)
                    if inject == "copy" and n == 512 * KIB + 1 and rep == 1:
                        b[n // 2] ^= 0x10                 # self-test only: proves the comparison notices one flipped bit
                else:
                    b.copy_(a, non_blocking=False)
                    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
                got = b.cpu()
                res["cases_run"] += 1
                if not torch.equal(got, ref):
                    diff = (got != ref).nonzero().flatten()
                    bad_here += 1
                    res["mismatches"] += 1
                    res["cases"].append({"dir": f"{src}->{dst}", "bytes": n, "rep": rep, "differing_bytes": int(diff.numel()), "first_index": int(diff[0]) if diff.numel() else None})
            log(f"  copy {src}->{dst} {n:>9d} bytes x{reps_for(n)}: {'OK' if not bad_here else f'{bad_here} MISMATCH'}")
    return res


# ---------------------------------------------------------------------------------------------------- NCCL worker (one rank)
def worker(rank, devs, port, out_path, backend, seed, cpu):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    dev = torch.device("cpu") if cpu else torch.device(f"cuda:{devs[rank]}")
    if not cpu:
        torch.cuda.set_device(devs[rank])
    t0 = time.time()
    dist.init_process_group(backend, rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}", timeout=timedelta(seconds=180))
    warm = torch.ones((6,), device=dev)
    dist.all_reduce(warm)                                  # NCCL connects lazily; the first call is slow
    init_s = time.time() - t0
    results, bad = [], 0
    for dname, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        for n in NCCL_ELEMS:
            for rep in range(REPS):
                xs = []
                for r in (0, 1):
                    g = torch.Generator().manual_seed(seed + 104729 * rep + 31 * n + 1009 * r + (7 if dname == "bf16" else 0))
                    xs.append((torch.randn(n, generator=g) * 4.0).to(dtype))
                expected = (xs[0].float() + xs[1].float()).to(dtype)
                t = xs[rank].to(dev)
                dist.all_reduce(t)
                if not cpu:
                    torch.cuda.synchronize(devs[rank])
                if cpu and os.environ.get("P2P_CHECK_INJECT") == "nccl" and rank == 1 and dname == "bf16" and n == 4096 and rep == 2:
                    t[7] = t[7] * 1.5 + 1.0              # self-test only: proves the comparison notices one wrong element
                got = t.cpu()
                if dname == "fp32":
                    exact = bool(torch.equal(got, expected))
                    tol = 1e-6 * (xs[0].abs() + xs[1].abs()) + 1e-30
                    nbad = int(((got - expected).abs() > tol).sum())
                    inexact = int((got != expected).sum())
                else:
                    d = (got.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
                    exact = bool((d == 0).all())
                    nbad = int((d > 1).sum())
                    inexact = int((d != 0).sum())
                bad += 1 if nbad else 0
                results.append({"dtype": dname, "elements": n, "rep": rep, "bit_exact": exact, "inexact_elements": inexact, "bad_elements": nbad})
    timing = {}
    for dname, dtype, n, iters in (("bf16", torch.bfloat16, 64 * 1024, 100), ("bf16", torch.bfloat16, 8 * MIB, 20), ("fp32", torch.float32, 4 * MIB, 20)):
        t = torch.ones(n, dtype=dtype, device=dev)
        for _ in range(3):
            dist.all_reduce(t)
        if not cpu:
            torch.cuda.synchronize(devs[rank])
        s = time.time()
        for _ in range(iters):
            dist.all_reduce(t)
        if not cpu:
            torch.cuda.synchronize(devs[rank])
        per = (time.time() - s) / iters
        timing[f"{dname}_{n * t.element_size() // 1024}KiB"] = {"us_per_call": round(per * 1e6, 1), "GB_s_algbw": round(n * t.element_size() / per / 1e9, 2)}
    dist.barrier()
    dist.destroy_process_group()
    with open(out_path, "w") as f:
        json.dump({"rank": rank, "init_first_allreduce_s": round(init_s, 2), "bad_cases": bad, "results": results, "timing": timing}, f)


# ---------------------------------------------------------------------------------------------------- parent
def transport_lines(text):
    p2p, shm, net = len(re.findall(r"via P2P", text)), len(re.findall(r"via SHM", text)), len(re.findall(r"via NET", text))
    paths = sorted(set(re.findall(r"via (P2P/[A-Za-z_]+(?:/\w+)?|SHM/\w+/\w+|SHM|NET/\S+)", text)))[:8]
    kind = "none"
    if p2p and not shm and not net: kind = "P2P"
    elif shm and not p2p and not net: kind = "SHM"
    elif p2p or shm or net: kind = "mixed"
    lines = [l[:260] for l in text.splitlines() if "NCCL INFO" in l and re.search(r"via |Channel|NCCL version|Bootstrap|Init COMPLETE|P2P|SHM|NET/", l)][:200]
    return {"kind": kind, "p2p_lines": p2p, "shm_lines": shm, "net_lines": net, "paths": paths, "lines": lines}


def run_nccl(devs, seed, out_dir, backend, cpu, timeout):
    port = free_port()
    env = dict(os.environ, NCCL_DEBUG="INFO", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    if not env.get("NCCL_SOCKET_IFNAME"):
        env["NCCL_SOCKET_IFNAME"] = "lo"                    # bootstrap on loopback only, as in the arms
    procs, outs = [], []
    for rank in (0, 1):
        outp = os.path.join(out_dir, f"nccl_rank{rank}.json")
        if os.path.exists(outp):
            os.unlink(outp)
        logf = open(os.path.join(out_dir, f"nccl_rank{rank}.log"), "w")
        cmd = [sys.executable, "-I", "-B", os.path.abspath(__file__), "--worker", str(rank), "--devices", ",".join(map(str, devs)), "--port", str(port),
               "--out-file", outp, "--backend", backend, "--seed", str(seed)] + (["--cpu"] if cpu else [])
        procs.append((subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env), logf))
        outs.append(outp)
    t0, rcs = time.time(), [None, None]
    while time.time() - t0 < timeout and any(r is None for r in rcs):
        for i, (p, _) in enumerate(procs):
            if rcs[i] is None:
                rcs[i] = p.poll()
        if any(r not in (None, 0) for r in rcs):
            break
        time.sleep(0.5)
    for p, logf in procs:
        if p.poll() is None:
            p.kill()
        logf.close()
    text = "".join(open(os.path.join(out_dir, f"nccl_rank{r}.log"), errors="replace").read() for r in (0, 1))
    res = {"returncodes": [p.poll() for p, _ in procs], "transport": transport_lines(text), "ranks": []}
    for outp in outs:
        if os.path.exists(outp):
            res["ranks"].append(json.load(open(outp)))
    res["completed"] = len(res["ranks"]) == 2 and all(r == 0 for r in res["returncodes"])
    res["bad_cases"] = sum(r["bad_cases"] for r in res["ranks"])
    res["inexact"] = sum(1 for r in res["ranks"] for c in r["results"] if not c["bit_exact"])
    res["cases"] = sum(len(r["results"]) for r in res["ranks"])
    res["error_tail"] = [] if res["completed"] else text.splitlines()[-8:]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="p2p-check"); ap.add_argument("--devices", default="0,1"); ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--skip-nccl", action="store_true"); ap.add_argument("--cpu-selftest", action="store_true")
    ap.add_argument("--big-repeats", type=int, default=10, help="extra 32 MiB copies per direction (intermittent corruption)")
    ap.add_argument("--timeout", type=float, default=420)
    ap.add_argument("--worker", type=int); ap.add_argument("--port", type=int); ap.add_argument("--out-file"); ap.add_argument("--backend", default="nccl"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    devs = [int(x) for x in a.devices.split(",")]
    if a.worker is not None:
        worker(a.worker, devs, a.port, a.out_file, a.backend, a.seed, a.cpu)
        return 0
    if not a.out:
        ap.error("--out is required")
    os.makedirs(a.out, exist_ok=True)
    receipt = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "devices": devs, "cpu_selftest": a.cpu_selftest, "seed": a.seed}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        log(f"GATE p2p-check FAIL cannot import torch: {e!r}"); receipt.update(pass_=False, status="cannot-run", error=repr(e)); json.dump(receipt, open(os.path.join(a.out, "p2p_check.json"), "w"), indent=1)
        return 2
    cpu = a.cpu_selftest
    if not cpu:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n < 2 or max(devs) >= n:
            log(f"GATE p2p-check FAIL need two CUDA devices {devs}, torch sees {n}"); receipt.update(pass_=False, status="cannot-run", error=f"{n} devices"); json.dump(receipt, open(os.path.join(a.out, "p2p_check.json"), "w"), indent=1)
            return 2
        for d in devs:
            free, total = torch.cuda.mem_get_info(d)
            if free < 0.9 * total:
                log(f"GATE p2p-check FAIL GPU {d} is not free ({(total - free) // MIB} MiB in use): refusing to run next to a live service")
                receipt.update(pass_=False, status="cannot-run", error=f"GPU {d} busy"); json.dump(receipt, open(os.path.join(a.out, "p2p_check.json"), "w"), indent=1)
                return 2
        receipt["gpus"] = [torch.cuda.get_device_name(d) for d in devs]
    log(f"P2P/NCCL content check on devices {devs}{' (CPU self-test: no GPU is touched)' if cpu else ''}")
    t0 = time.time()
    cp = copy_test(torch, devs, a.seed, cpu, a.big_repeats, os.environ.get("P2P_CHECK_INJECT", "") if cpu else "")
    receipt["copy"] = cp
    copy_ok = cp["mismatches"] == 0
    log(f"GATE p2p-copy {'PASS' if copy_ok else 'FAIL'} {cp['cases_run']} copies, {cp['mismatches']} mismatches, peer access {cp['peer']}")
    if not all(cp["peer"].values()):
        log("WARN CUDA reports no peer access in at least one direction: the copies above used the host-staged fallback, NCCL will pick SHM")
    nccl_ok = True
    if not a.skip_nccl:
        nc = run_nccl(devs, a.seed, a.out, "gloo" if cpu else "nccl", cpu, a.timeout)
        receipt["nccl"] = {k: v for k, v in nc.items()}
        nccl_ok = nc["completed"] and nc["bad_cases"] == 0
        tr = nc["transport"]
        log(f"GATE nccl-allreduce {'PASS' if nccl_ok else 'FAIL'} {nc['cases']} all_reduce cases (fp32 and bf16), {nc['bad_cases']} wrong, {nc['inexact']} not bit-exact (exact is expected for 2 operands); "
            f"return codes {nc['returncodes']}")
        for r in nc["ranks"]:
            log(f"  rank {r['rank']}: first all_reduce incl. init {r['init_first_allreduce_s']} s; timing {json.dumps(r['timing'])}")
        log(f"INFO NCCL transport in this check: {tr['kind']} (P2P lines {tr['p2p_lines']}, SHM {tr['shm_lines']}, NET {tr['net_lines']}); paths {tr['paths']}")
        for l in tr["lines"][:12]:
            log("  " + l)
        if not nc["completed"]:
            for l in nc["error_tail"]:
                log("  | " + l[:240])
    ok = copy_ok and nccl_ok
    receipt.update(pass_=ok, seconds=round(time.time() - t0, 1), status="pass" if ok else "FAILED")
    receipt["pass"] = receipt.pop("pass_")
    json.dump(receipt, open(os.path.join(a.out, "p2p_check.json"), "w"), indent=1)
    log("P2P/NCCL content check PASSED" if ok else "P2P/NCCL content check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
