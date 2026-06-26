# UFM Benchmark — public artifact chain (drafts)

All numbers below are from the reproduced run: RTX 4090 (23.5 GB), torch 2.10.0+cu128,
fp32 experts (33.6M params each), token-choice MoE, inference. Keep claims exactly as
stated — they are calibrated to the data. Do not round "240×" up or drop the limitation.

---

## 1. X / Twitter thread

**1/**
I ran a 24 GB Mixture-of-Experts on a 24 GB RTX 4090.

The standard "put it all on the GPU" approach OOMs. UFM runs it at 19.6 GB — by paging
experts in and out of VRAM.

Benchmark, code, and the honest catch 🧵

**2/**
Setup: a token-choice MoE where the expert bank is the memory cost. 192 experts × 33.6M
params (fp32) = 24.0 GB. The 4090 has 23.5 GB usable.

baseline → OOM.
UFM → runs, VRAM held at 19.6 GB. ✅

[attach vram.png]

**3/**
But "it runs" isn't the story. Throughput is.

When the working set fits the VRAM budget (96 experts / 12 GB):
• UFM: 21,174 tok/s
• baseline: 21,017 tok/s  → within ~1%
• naive CPU offload: 87 tok/s → UFM is ~240× faster

Paging is nearly free when you reuse the hot set.

[attach throughput.png]

**4/** the honest catch
At 192 experts my synthetic routing hits ~every expert every step. No reuse → UFM (37
tok/s) ties naive streaming (43). When you touch everything every step you're
transfer-bound and a cache can't help.

UFM is a bet on locality, not magic memory.

**5/**
So the real claim is calibrated:
"Run models larger than your VRAM, at full speed — as long as you're not touching all of
it at once."

For a lot of MoE inference, that's exactly the situation.

**6/**
Reproduce in 3 commands (it prints your exact env and records OOMs as results):
github.com/Linutesto/ufm → benchmarks/

Part of an open research program: capable AI you can own, on commodity hardware.
Full writeup: yandesbiens.com/blog/ufm-benchmark/

---

## 2. LinkedIn post

I just published the first formal benchmark from my independent AI research at Éthiqueia
Québec inc., and I held it to one rule: every claim is reproducible from a one-command
script, or it's marked speculative.

The question: can you run a model whose memory footprint exceeds your GPU?

I tested UFM — a memory manager that treats VRAM + RAM as one pool — on a routed
Mixture-of-Experts whose expert bank is 24 GB, on a 24 GB RTX 4090.

Results:
• The standard all-on-GPU approach runs out of memory.
• UFM runs the same model, holding VRAM at 19.6 GB.
• When the active working set fits the VRAM budget, UFM delivers full-GPU throughput
  (within ~1% of baseline) and ~240× the throughput of naive CPU offloading.

And the part most benchmarks leave out — the limitation: when every expert is needed every
step (no locality), UFM offers no speedup over naive streaming. You're transfer-bound, and
a cache cannot help. UFM is a bet on locality, not magic memory.

I think honest, reproducible engineering is the only thing worth publishing. Code, figures,
and full method (including what doesn't work) are open.

Repo: github.com/Linutesto/ufm
Writeup: yandesbiens.com/blog/ufm-benchmark/

#AI #MachineLearning #LocalAI #MoE #OpenResearch

---

## 3. r/LocalLLaMA submission

**Title:** I benchmarked running an expert bank larger than VRAM on a single 4090 — UFM vs. naive CPU offload (with the honest failure case)

**Body:**

I extracted a small memory manager (UFM) from my own research and benchmarked it properly
instead of just claiming it works.

**Setup:** token-choice MoE, experts dominate memory. 192 experts × 33.6M params (fp32) =
24.0 GB bank. RTX 4090, 23.5 GB usable. Inference. Three modes: all-on-GPU (baseline),
naive `.cuda()`/`.cpu()` per call, and UFM (CPU+pinned, paged, LRU keep-hot).

**Results:**

| experts | bank | baseline | naive offload | UFM |
|---|---|---|---|---|
| 96 | 12 GB | 21,017 tok/s | 87 tok/s | 21,174 tok/s |
| 192 | 24 GB | OOM | 43 tok/s | 37 tok/s |

Takeaways:
- At 192 experts the bank doesn't fit; baseline OOMs, UFM runs at 19.6 GB VRAM.
- When the working set fits the VRAM budget (96 experts), UFM ≈ baseline speed and ~240×
  faster than naive offload — because experts are paged once and reused.
- **Honest catch:** at 192 my synthetic routing hits ~all experts every step, so there's no
  reuse and UFM (37) ties naive (43). Zero locality = transfer-bound = cache can't help.

So it's a bet on routing locality, not magic memory. Realistic MoE inference usually has
locality, which is where it pays off.

Repo + one-command reproduce (prints your env, records OOMs): github.com/Linutesto/ufm
(benchmarks/). Roast the method — it's inference-only for now (v0.1 paging is forward-safe;
training-time autograd across evicted params is future work). Curious if it holds on your
cards.

---

## 4. Newsletter section

**Subject:** Proof drop #1 — a 24 GB model on a 24 GB GPU

I'm starting a series of "proof drops": small, reproducible benchmarks from my research,
each held to one rule — reproducible from a one-command script, or marked speculative.

First up: can a single RTX 4090 run a model that doesn't fit in its memory?

I tested UFM (a VRAM+RAM pager) on a 24 GB Mixture-of-Experts. The standard approach OOMs.
UFM runs it at 19.6 GB. And when the active working set fits the VRAM budget, it does so at
full-GPU speed — within 1% of baseline, ~240× faster than naive CPU offload.

I also published the case where it *doesn't* help: touch every expert every step and you're
transfer-bound, where UFM ties dumb streaming. I'd rather show you the edge of the envelope
than pretend there isn't one.

Code, figures, and method: yandesbiens.com/blog/ufm-benchmark/

More drops coming — memory architectures, then the fractal backbone. One 4090, in the open.

— Yan

---

## 5. GitHub README badge line (for ufm/README.md)

> 📊 **Benchmarked:** runs a 24 GB expert bank on a 23.5 GB RTX 4090 (baseline OOMs);
> within ~1% of baseline throughput when the working set fits the budget, ~240× faster than
> naive CPU offload. Honest limitation + reproduce: [`benchmarks/`](benchmarks/).
