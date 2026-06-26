# UFM Benchmark — fitting an expert bank larger than VRAM

This benchmark tests one claim, honestly:

> **Can UFM run a routed model whose expert bank exceeds GPU memory — and at what cost?**

It sweeps the number of experts upward and runs three strategies for placing them:

| mode | where experts live | expectation |
|---|---|---|
| `baseline` | all experts resident on GPU | fastest, until it **OOMs** |
| `naive_offload` | on CPU; `.cuda()`/`.cpu()` per call | fits in tiny VRAM, but slow every step |
| `ufm` | on CPU (+pinned); UFM pages them, LRU keep-hot | fits, and caches the hot set |

## Quick start

```bash
cd ufm
pip install -e .            # installs ufm + torch
pip install psutil matplotlib

cd benchmarks
python run_benchmark.py     # full sweep, all three modes
python plot_results.py      # writes figures to results/

# or:
./run.sh                    # run + plot in one command
python run_benchmark.py --quick   # ~30s smoke test
```

Results land in `results/`: `summary.json`, `runs.jsonl`, per-run telemetry CSVs,
and three PNG figures.

## Reproduced result (the numbers in the figures)

**Hardware:** NVIDIA RTX 4090 (23.51 GiB usable), driver 580.126.18.
**Software:** torch 2.10.0+cu128, CUDA 12.8, Python 3.x, Linux.
**Model:** token-choice MoE, `d_model=2048`, `d_hidden=8192` → **33.6M params/expert
(0.125 GB fp32)**, `top_k=2`, batch 4 × seq 128 (512 tokens/step), 3 steps (1 warmup).
**UFM:** `vram_target_gb=16`, `headroom=1`, pinned CPU memory.

| mode | experts | bank (GB) | status | peak VRAM (GB) | tokens/s |
|---|---:|---:|:--:|---:|---:|
| baseline | 96 | 12.0 | ok | 12.1 | 21,017 |
| baseline | 144 | 18.0 | ok | 18.1 | 14,374 |
| **baseline** | **192** | **24.0** | **OOM** | — | — |
| naive_offload | 96 | 12.0 | ok | 0.22 | 87 |
| naive_offload | 192 | 24.0 | ok | 0.22 | 43 |
| **ufm** | **96** | **12.0** | **ok** | **12.1** | **21,174** |
| **ufm** | **192** | **24.0** | **ok** | **19.6** | **37** |

### What this shows (and doesn't)

1. **Fit-the-unfittable — confirmed.** At 192 experts the bank is **24.0 GB > 23.5 GB
   VRAM**; `baseline` OOMs. UFM runs the same model, holding VRAM at **19.6 GB**.
2. **When the working set fits the budget, paging is nearly free.** At 96 experts
   (12 GB bank ≤ 16 GB target), UFM hits **21,174 tok/s — within ~1% of baseline**
   (21,017) and **~240× faster than naive offload** (87). The LRU keep-hot cache means
   experts are paged in once and reused.
3. **Honest limitation — no locality, no win.** This synthetic workload routes 512
   tokens across up to 192 experts, so at the largest size *almost every expert fires
   every step*. With a working set (24 GB) far above the budget (16 GB) and no reuse,
   UFM (37 tok/s) is **on par with — even slightly behind — naive streaming** (43).
   When you genuinely touch everything every step, you are transfer-bound and caching
   cannot help. The regime where UFM wins is the realistic one: **routing locality**, so
   the active working set fits the VRAM budget.

## Honest tradeoffs / limitations

- **UFM v0.1 paging is forward/inference-safe.** It swaps tensor *storages*; it is not
  yet wired for training-time autograd across evicted params. Optimizer-state relief for
  training is a separate lever (`OffloadedAdam`), not exercised here. This benchmark is
  **inference** (`torch.no_grad`).
- **Eviction is plain LRU** (cost-aware eviction and an NVMe Tier-2 are roadmap). The
  `nvme_offload_gb` telemetry column is therefore ~0 here, and reported as such.
- **fp32 experts** are used to make the memory story legible; bf16 would roughly double
  the bank you can hold. Numbers are single-run on one machine — treat as indicative, not
  a leaderboard. Re-run on your hardware; the script prints your exact environment.
- **Throughput is small-batch.** This measures the paging mechanism, not a tuned serving
  setup.

## Files

```
model.py           ExpertBank — the routed MoE under test
telemetry.py       background VRAM/RAM/NVMe sampler -> CSV
run_benchmark.py   the harness (one command, OOM-capturing)
plot_results.py    figure generator
run.sh             run + plot
results/           summary.json, runs.jsonl, telemetry_*.csv, *.png
```
