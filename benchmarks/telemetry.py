"""
Background telemetry sampler for the UFM benchmark.

Samples, on a fixed interval and in a daemon thread:
  - GPU allocated / reserved VRAM (torch)
  - process RSS + system RAM used (psutil)
  - bytes resident in the UFM offload dir on NVMe (psutil/os) — Tier-2 hook;
    UFM v0.1 uses the RAM tier, so this is typically 0 and reported honestly.

Rows are written to a CSV so plots can reconstruct VRAM/RAM over wall-clock time.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path

import torch

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None

_GB = 1024 ** 3


def _dir_bytes(path: str) -> int:
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


class TelemetrySampler:
    def __init__(self, csv_path: str, offload_dir: str | None = None, interval_s: float = 0.05):
        self.csv_path = csv_path
        self.offload_dir = offload_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self.peak_gpu_gb = 0.0
        self.peak_ram_gb = 0.0

    def _sample(self):
        gpu_alloc = gpu_res = 0.0
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / _GB
            gpu_res = torch.cuda.memory_reserved() / _GB
        proc_ram = sys_ram = 0.0
        if psutil is not None:
            proc_ram = _PROC.memory_info().rss / _GB
            sys_ram = psutil.virtual_memory().used / _GB
        nvme = (_dir_bytes(self.offload_dir) / _GB) if self.offload_dir else 0.0
        self.peak_gpu_gb = max(self.peak_gpu_gb, gpu_alloc)
        self.peak_ram_gb = max(self.peak_ram_gb, proc_ram)
        return {
            "t": round(time.time() - self._t0, 4),
            "gpu_alloc_gb": round(gpu_alloc, 4),
            "gpu_reserved_gb": round(gpu_res, 4),
            "proc_ram_gb": round(proc_ram, 4),
            "sys_ram_gb": round(sys_ram, 4),
            "nvme_offload_gb": round(nvme, 4),
        }

    def _run(self):
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as fh:
            writer = None
            while not self._stop.is_set():
                row = self._sample()
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
                time.sleep(self.interval_s)

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return False
