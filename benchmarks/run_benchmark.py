#!/usr/bin/env python3
"""
UFM benchmark harness — one-command, reproducible.

Compares three ways to run a routed Mixture-of-Experts whose expert bank grows
past VRAM, as the number of experts is swept upward:

  baseline       all experts resident on GPU            (fastest, but OOMs)
  naive_offload  experts on CPU, .cuda()/.cpu() per use (fits, slow, no caching)
  ufm            experts on CPU(+pinned), UFM pages them (fits, LRU keep-hot)

For each (mode, num_experts) it records: success / OOM, peak VRAM, peak process
RAM, throughput (tokens/sec), per-expert footprint, and a wall-clock telemetry
trace. Results are written to results/ as JSON + JSONL + per-run CSV.

Usage:
    python run_benchmark.py                      # default sweep, all modes
    python run_benchmark.py --experts 32 64 128  # custom sweep
    python run_benchmark.py --modes baseline ufm
    python run_benchmark.py --quick              # tiny smoke run

Honest by design: if a mode OOMs, that is recorded as the result, not hidden.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

# Run from a checkout without installing: make ../src importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ufm import UFM  # noqa: E402
from model import ExpertBank  # noqa: E402
from telemetry import TelemetrySampler  # noqa: E402

_GB = 1024 ** 3
RESULTS = HERE / "results"


def _is_oom(err: Exception) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def make_batch(batch: int, seq: int, vocab: int, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=g).to(device)


def place_experts(model: ExpertBank, mode: str, pin: bool) -> None:
    """Sticky parts (embed/router/head) always on GPU. Experts placed per mode."""
    model.embed.cuda(); model.router.cuda(); model.head.cuda()
    if mode == "baseline":
        for e in model.experts:
            e.cuda()
    else:  # naive_offload + ufm keep experts on CPU initially
        for e in model.experts:
            e.cpu()
            if pin and mode == "ufm":
                for p in e.parameters():
                    p.data = p.data.pin_memory()


def expert_ctx_factory(mode: str, ufm: UFM | None):
    if mode == "ufm":
        return lambda m: ufm.activate(m)
    if mode == "naive_offload":
        @contextlib.contextmanager
        def _naive(m):
            m.cuda()
            try:
                yield
            finally:
                m.cpu()
        return _naive
    return None  # baseline


def run_one(mode: str, n_experts: int, cfg: dict) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {
        "mode": mode,
        "num_experts": n_experts,
        "status": "ok",
        "error": None,
        "peak_vram_gb": None,
        "peak_ram_gb": None,
        "tokens_per_s": None,
        "step_ms_mean": None,
        "expert_bank_gb": None,
        "total_params_m": None,
        "steps_measured": 0,
    }
    tele_csv = RESULTS / f"telemetry_{mode}_{n_experts:04d}.csv"
    ufm = None
    model = None
    try:
        model = ExpertBank(
            vocab=cfg["vocab"], d_model=cfg["d_model"], d_hidden=cfg["d_hidden"],
            num_experts=n_experts, top_k=cfg["top_k"],
        )
        model.eval()
        rec["expert_bank_gb"] = round(model.expert_bytes(4) / _GB, 3)
        rec["total_params_m"] = round(model.total_params() / 1e6, 1)

        if mode == "ufm":
            ufm = UFM(vram_target_gb=cfg["ufm_target_gb"], gpu_headroom_gb=cfg["ufm_headroom_gb"],
                      offload_dir=str(RESULTS / "ufm_store"))

        place_experts(model, mode, pin=cfg["pin"])
        if mode == "ufm":
            for e in model.experts:
                ufm.register_module(e)

        ctx = expert_ctx_factory(mode, ufm)
        batch = make_batch(cfg["batch"], cfg["seq"], cfg["vocab"], "cuda", cfg["seed"])
        tokens = cfg["batch"] * cfg["seq"]

        step_times = []
        with torch.no_grad(), TelemetrySampler(str(tele_csv), offload_dir=str(RESULTS / "ufm_store")) as tele:
            for step in range(cfg["steps"]):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(batch, expert_ctx=ctx)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if step >= cfg["warmup"]:
                    step_times.append(dt)
            rec["peak_ram_gb"] = round(tele.peak_ram_gb, 3)

        rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / _GB, 3)
        if step_times:
            mean_dt = sum(step_times) / len(step_times)
            rec["step_ms_mean"] = round(mean_dt * 1000, 1)
            rec["tokens_per_s"] = round(tokens / mean_dt, 1)
            rec["steps_measured"] = len(step_times)
    except Exception as err:  # noqa: BLE001
        if _is_oom(err):
            rec["status"] = "oom"
        else:
            rec["status"] = "error"
        rec["error"] = f"{type(err).__name__}: {str(err)[:200]}"
    finally:
        del model, ufm
        torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser(description="UFM benchmark")
    ap.add_argument("--modes", nargs="+", default=["baseline", "naive_offload", "ufm"])
    ap.add_argument("--experts", nargs="+", type=int, default=[48, 96, 144, 192])
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--d-hidden", type=int, default=8192)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ufm-target-gb", type=float, default=16.0)
    ap.add_argument("--ufm-headroom-gb", type=float, default=1.0)
    ap.add_argument("--no-pin", action="store_true", help="disable pinned memory for UFM")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    if args.quick:
        args.experts = [8, 16]
        args.d_hidden = 2048
        args.steps = 2

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — this benchmark needs a GPU.", file=sys.stderr)
        sys.exit(1)

    cfg = dict(
        d_model=args.d_model, d_hidden=args.d_hidden, vocab=args.vocab, top_k=args.top_k,
        batch=args.batch, seq=args.seq, steps=args.steps, warmup=args.warmup,
        ufm_target_gb=args.ufm_target_gb, ufm_headroom_gb=args.ufm_headroom_gb,
        pin=not args.no_pin, seed=args.seed,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    dev = torch.cuda.get_device_properties(0)
    env = {
        "gpu": dev.name,
        "vram_total_gb": round(dev.total_memory / _GB, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": _driver_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "sweep": args.experts,
        "modes": args.modes,
    }

    jsonl_path = RESULTS / "runs.jsonl"
    summary = {"env": env, "results": []}
    print(f"# UFM benchmark on {env['gpu']} ({env['vram_total_gb']} GB) | torch {env['torch']}")
    print(f"# expert MLP: d_model={cfg['d_model']} d_hidden={cfg['d_hidden']} "
          f"-> {2*cfg['d_model']*cfg['d_hidden']/1e6:.1f}M params/expert ({2*cfg['d_model']*cfg['d_hidden']*4/_GB:.3f} GB fp32)\n")
    hdr = f"{'mode':<14}{'N':>5}{'bank_GB':>9}{'status':>9}{'peakVRAM':>10}{'peakRAM':>9}{'tok/s':>10}{'step_ms':>9}"
    print(hdr); print("-" * len(hdr))

    with open(jsonl_path, "w") as jf:
        for mode in args.modes:
            for n in args.experts:
                rec = run_one(mode, n, cfg)
                summary["results"].append(rec)
                jf.write(json.dumps(rec) + "\n"); jf.flush()
                vram = f"{rec['peak_vram_gb']}" if rec["peak_vram_gb"] is not None else "-"
                ram = f"{rec['peak_ram_gb']}" if rec["peak_ram_gb"] is not None else "-"
                tps = f"{rec['tokens_per_s']}" if rec["tokens_per_s"] is not None else "-"
                sms = f"{rec['step_ms_mean']}" if rec["step_ms_mean"] is not None else "-"
                bank = f"{rec['expert_bank_gb']}" if rec["expert_bank_gb"] is not None else "-"
                print(f"{mode:<14}{n:>5}{bank:>9}{rec['status']:>9}{vram:>10}{ram:>9}{tps:>10}{sms:>9}")

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n# wrote {RESULTS/'summary.json'} and {jsonl_path}")
    print("# next: python plot_results.py")


def _driver_version() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
