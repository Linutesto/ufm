from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import pathlib
import torch


def _bytes_of(t: torch.Tensor) -> int:
    try:
        return int(t.numel() * t.element_size())
    except Exception:
        return 0


class UFM:
    """
    Unified Fractal Memory manager.

    Treats GPU VRAM + CPU pinned RAM as one elastic pool for swappable
    sub-modules (e.g. MoE experts, large fractal subtrees), so a single
    high-end GPU can train/serve models whose footprint exceeds device memory.

    - Tracks module parameter residency (GPU/CPU)
    - Asynchronous prefetch to GPU via a dedicated CUDA stream
    - LRU eviction when allocated VRAM exceeds a target

    Register modules (experts, large sub-blocks) with ``register_module(mod)``
    and wrap critical regions with ``with ufm.activate(mod): ...``.
    """

    def __init__(
        self,
        vram_target_gb: float = 22.0,
        gpu_headroom_gb: float = 1.0,
        offload_dir: str = "./ufm_store",
    ):
        self.vram_target = int(max(0.0, (vram_target_gb - gpu_headroom_gb)) * (1024 ** 3))
        self.offload_dir = pathlib.Path(offload_dir)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self.prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.registry: dict[torch.nn.Module, dict] = {}
        self.lru: OrderedDict[torch.nn.Module, None] = OrderedDict()
        self.stats = {
            "gpu_bytes": 0,
            "cpu_bytes": 0,
            "prefetch": 0,
            "evict": 0,
        }

    def _current_gpu(self) -> tuple[int, int]:
        if not torch.cuda.is_available():
            return 0, 0
        try:
            return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
        except Exception:
            return 0, 0

    def register_module(self, module: torch.nn.Module) -> None:
        params = [p for p in module.parameters() if p.requires_grad]
        nbytes = sum(_bytes_of(p.data) for p in params)
        self.registry[module] = {"state": "gpu", "params": params, "bytes": nbytes}
        self.lru[module] = None
        self.stats["gpu_bytes"] += nbytes

    def _evict_one(self) -> bool:
        # Simple LRU GPU->CPU pinned evict
        if not self.lru:
            return False
        victim, _ = self.lru.popitem(last=False)
        rec = self.registry.get(victim)
        if not rec or rec["state"] != "gpu":
            return False
        moved = 0
        with torch.no_grad():
            for p in rec["params"]:
                try:
                    if p.data.is_cuda:
                        buf = p.data.detach().cpu().pin_memory()
                        p.data = buf
                        moved += _bytes_of(buf)
                except Exception:
                    continue
        rec["state"] = "cpu"
        self.stats["gpu_bytes"] -= moved
        self.stats["cpu_bytes"] += moved
        self.stats["evict"] += 1
        return True

    def _ensure_headroom(self) -> None:
        alloc, _ = self._current_gpu()
        safety = 4 * (1024 ** 2)  # 4MB slack
        while alloc > max(0, self.vram_target - safety):
            if not self._evict_one():
                break
            alloc, _ = self._current_gpu()

    def _to_gpu_async(self, t: torch.Tensor) -> torch.Tensor:
        if self.prefetch_stream is None:
            return t.to(device="cuda") if torch.cuda.is_available() else t
        with torch.cuda.stream(self.prefetch_stream):
            return t.to(device="cuda", non_blocking=True)

    def prefetch(self, module: torch.nn.Module) -> None:
        rec = self.registry.get(module)
        if rec is None:
            return
        if rec["state"] == "gpu":
            # refresh recency
            self.lru[module] = None
            self.lru.move_to_end(module, last=True)
            return
        self._ensure_headroom()
        moved = 0
        with torch.no_grad():
            new_bufs: list[torch.Tensor] = []
            for p in rec["params"]:
                buf = self._to_gpu_async(p.data)
                new_bufs.append(buf)
            if self.prefetch_stream is not None:
                self.prefetch_stream.synchronize()
            for p, buf in zip(rec["params"], new_bufs):
                p.data = buf
                moved += _bytes_of(buf)
        rec["state"] = "gpu"
        self.stats["prefetch"] += 1
        self.stats["gpu_bytes"] += moved
        self.stats["cpu_bytes"] = max(0, self.stats["cpu_bytes"] - moved)
        self.lru[module] = None
        self.lru.move_to_end(module, last=True)
        self._ensure_headroom()

    @contextmanager
    def activate(self, module: torch.nn.Module):
        self.prefetch(module)
        try:
            yield
        finally:
            # update recency
            if module in self.lru:
                self.lru.move_to_end(module, last=True)

    def report(self) -> dict:
        alloc, reserved = self._current_gpu()
        gb = 1024 ** 3
        return {
            "gpu_alloc_gb": round(alloc / gb, 3),
            "gpu_reserved_gb": round(reserved / gb, 3),
            "gpu_target_gb": round(self.vram_target / gb, 3),
            **self.stats,
            "lru_len": len(self.lru),
        }
