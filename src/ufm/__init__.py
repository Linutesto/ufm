"""
UFM — Unified Fractal Memory.

Treat GPU VRAM + CPU pinned RAM as one elastic pool so a single high-end GPU
behaves like a mini-cluster: prefetch the subgraphs you need, evict the ones you
don't, and offload optimizer state to CPU.

Extracted from the Fractal Neurons research framework by Yan Desbiens.
"""

from .manager import UFM
from .offload import OffloadedAdam

__all__ = ["UFM", "OffloadedAdam"]
__version__ = "0.1.1"
