"""E1 step 8: read the pilot runs and freeze the settings. Prints the FROZEN block and writes frozen.json under --work.

  python scripts/e1_freeze.py 10m --work /shared/aayu/e1      pick mask and learning rate per arm from the 10M pilots (lowest tune log loss on the
                                                              primary readout), link the chosen runs as the grid's 10m r1 seed 0, apply K1
  python scripts/e1_freeze.py big --work /shared/aayu/e1      pick the learning rate per arm at 50m and 200m, then the projected GPU-hours and K3 cuts
"""
import argparse
import json
from pathlib import Path

import numpy as np

STEPS = {"signal": 19200, "latent": 14400}
GRID_CELLS = 5
SEEDS = 3


def tune(run: Path, key: str) -> float:
    return float(np.load(run / "probe.npz")[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("10m", "big"))
    ap.add_argument("--work", default="/shared/aayu/e1")
    a = ap.parse_args()
    work = Path(a.work)
    key = "tune_" + json.loads((work / "g0.json").read_text())["primary"]
    f = work / "frozen.json"
    frozen = json.loads(f.read_text()) if f.exists() else {}
    runs = work / "runs"
    if a.phase == "10m":
        rnd = np.mean([float(np.load(work / f"g0-random{s}.npz")[key]) for s in range(3)])
        for arm in STEPS:
            cands = [(tune(r, key), r) for r in sorted(runs.glob(f"{arm}-10m-r1-s0-m*-lr*")) if (r / "done.json").exists()]
            best_loss, best = min(cands)
            done = json.loads((best / "done.json").read_text())
            collapsed = [(r.name, (json.loads((r / "done.json").read_text()).get("target_std") or 1) < 0.05 or loss >= rnd) for loss, r in cands]
            frozen[arm] = {"mask": done["mask_ratio"], "lr": {"10m": done["lr"]}, "tune_loss_10m": best_loss, "random_tune_loss_10m": rnd,
                           "collapsed_runs": [n for n, c in collapsed if c], "pilot_runs": len(cands), "sec_per_step": {"10m": done["train_gpu_seconds"] / done["steps"]}}
            link = runs / f"{arm}-10m-r1-s0"
            if not link.exists():
                link.symlink_to(best.name)
        frozen["latent_failed"] = len(frozen["latent"]["collapsed_runs"]) == frozen["latent"]["pilot_runs"]
    else:
        for arm in STEPS:
            for size in ("50m", "200m"):
                cands = [(tune(r, key), r) for r in sorted(runs.glob(f"{arm}-{size}-r1-s0-pilot-lr*")) if (r / "done.json").exists()]
                _, best = min(cands)
                done = json.loads((best / "done.json").read_text())
                frozen[arm]["lr"][size] = done["lr"]
                frozen[arm]["sec_per_step"][size] = done["train_gpu_seconds"] / done["steps"]
        arms = ["signal"] if frozen.get("latent_failed") else list(STEPS)
        cut_seeds, cut_cells = 3, ["r1", "r4", "r16", "clean4", "dirty4"]

        def hours():
            total = 0.0
            for arm in arms:
                for size in ("10m", "50m", "200m"):
                    n = (cut_seeds * len(cut_cells)) if size == "200m" else GRID_CELLS * SEEDS
                    total += n * STEPS[arm] * frozen[arm]["sec_per_step"][size] / 3600 * 1.08  # 8% for the embeddings and probes
            return total

        cuts = []
        for what in ("seeds", "dirty4", "r16"):  # K3: 200m seeds 3 to 2, then dirty4 at 200m, then r16 at 200m
            if hours() <= 900:
                break
            if what == "seeds":
                cut_seeds = 2
            else:
                cut_cells.remove(what)
            cuts.append(what)
        frozen |= {"arms": arms, "projected_gpu_hours": round(hours(), 1), "k3_cuts": cuts, "seeds_200m": cut_seeds, "cells_200m": cut_cells,
                   "k3_over_cap_after_cuts": hours() > 900}
    f.write_text(json.dumps(frozen, indent=1))
    print("FROZEN_BEGIN")
    print(json.dumps(frozen, indent=1))
    print("FROZEN_END")


if __name__ == "__main__":
    main()
