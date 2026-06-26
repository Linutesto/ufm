# UFM — Unified Fractal Memory

![status](https://img.shields.io/badge/status-v0.1.0-ff5fa2)
![python](https://img.shields.io/badge/python-3.9%2B-3776ab)
![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-22c55e)

> Train bigger models on the GPU you already own.

> 🧬 Extracted from the [**Fractal Neurons**](https://yandesbiens.com/projects/fractal-neurons/)
> research framework by [Yan Desbiens](https://yandesbiens.com). Part of a wider body of
> local-first AI work — see also [`fmm`](https://github.com/Linutesto/fmm).

UFM treats **GPU VRAM + CPU pinned RAM** as one elastic memory pool. It keeps the
hot parts of your model on-device, asynchronously prefetches what you're about to
use, and evicts least-recently-used sub-modules when VRAM gets tight — so a single
high-end GPU (e.g. an RTX 4090) can train or serve models whose parameter +
optimizer footprint would otherwise OOM.

It exploits the fact that routed/sparse models (MoE, fractal backbones) only fire
a *subset* of their parameters per step. Routers and attention stay sticky on the
GPU; experts and large subtrees swap in and out around them.

Extracted from the [Fractal Neurons](https://yandesbiens.com/projects/fractal-neurons/)
research framework.

## Install

```bash
pip install -e .
```

## Usage

```python
import torch
from ufm import UFM, OffloadedAdam

model = MyRoutedModel().cuda()

# 1) Elastic residency for swappable sub-modules
ufm = UFM(vram_target_gb=22.0, gpu_headroom_gb=1.0)
for expert in model.experts:
    ufm.register_module(expert)

# Only the active expert is guaranteed resident; others may live on CPU.
with ufm.activate(model.experts[idx]):
    y = model.experts[idx](x)

print(ufm.report())   # {'gpu_alloc_gb': ..., 'evict': ..., 'prefetch': ...}

# 2) CPU-offloaded optimizer state (removes the 2-3x Adam VRAM cost)
opt = OffloadedAdam(model.parameters(), lr=3e-4)
loss.backward()
opt.step()
```

## What's in here

| Component | Purpose |
|---|---|
| `UFM` | Residency manager: register modules, `activate()` regions, LRU evict + async prefetch on a dedicated CUDA stream. |
| `OffloadedAdam` | AdamW-like optimizer keeping FP32 masters + moments in pinned CPU memory. |

## Status

v0.1.0 — extracted, packaged, and reproducible. The eviction policy is plain LRU;
a cost-aware variant (`bytes × age`) and Tier-2 (NVMe mmap) backing are on the
roadmap. See the
[UFM whitepaper](https://yandesbiens.com/projects/fractal-neurons/) for the full
design.

MIT licensed.
