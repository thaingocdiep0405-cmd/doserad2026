#!/usr/bin/env python3
"""Split proton inference wall time into GPU forward vs host-side work.

The leaderboard runtime is dominated by per-map inference (2301 maps in the
graded set), so what decides which optimisation is worth doing is whether the
time sits in the convolutions or in the host-side conditioning. This wraps the
real benchmark so the measured path is exactly the submission path.
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.model import PhotonDoseUNet3D  # noqa: E402

STATS = {"forward_s": 0.0, "calls": 0, "patches": 0}
_original = PhotonDoseUNet3D.forward


def _timed(self, x, *args, **kwargs):
    if torch.is_tensor(x):
        STATS["patches"] += int(x.shape[0])
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = _original(self, x, *args, **kwargs)
    torch.cuda.synchronize()
    STATS["forward_s"] += time.perf_counter() - start
    STATS["calls"] += 1
    return out


PhotonDoseUNet3D.forward = _timed

wall_start = time.perf_counter()
sys.argv = ["benchmark_roi_modes.py"] + sys.argv[1:]
try:
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "benchmark_roi_modes.py"), run_name="__main__")
except SystemExit:
    pass
wall = time.perf_counter() - wall_start

forward = STATS["forward_s"]
print("\n=== profile ===")
print(f"wall (incl. I/O, metrics) : {wall:.2f} s")
print(f"gpu forward               : {forward:.2f} s ({100 * forward / wall:.1f} % of wall)")
print(f"everything else           : {wall - forward:.2f} s ({100 * (wall - forward) / wall:.1f} %)")
print(f"forward calls             : {STATS['calls']}")
print(f"patches                   : {STATS['patches']}")
if STATS["patches"]:
    print(f"per patch                 : {1000 * forward / STATS['patches']:.2f} ms")
