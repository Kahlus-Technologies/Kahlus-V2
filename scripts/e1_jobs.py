"""Command lines for the E1 grid, from frozen.json: python scripts/e1_jobs.py rung1|rung2 --work /shared/aayu/e1 --run1 'env ... run_one.sh'"""
import argparse
import json
from pathlib import Path

STEPS = {"signal": 19200, "latent": 14400}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rung", choices=("rung1", "rung2"))
    ap.add_argument("--work", default="/shared/aayu/e1")
    ap.add_argument("--run1", required=True, help="the command prefix that runs one run: ARM SIZE CELL SEED LR MASK STEPS")
    a = ap.parse_args()
    work = Path(a.work)
    fz = json.loads((work / "frozen.json").read_text())
    if a.rung == "rung1":
        sizes, cells, seeds = ("50m", "10m"), ("r1", "r4", "r16", "clean4", "dirty4"), (0, 1, 2)  # the longest runs first
    else:
        k2 = json.loads((work / "k2.json").read_text())
        sizes, cells, seeds = ("200m",), tuple(k2["cells_200m"]), tuple(range(fz["seeds_200m"]))
    for size in sizes:
        for cell in cells:
            for seed in seeds:
                for arm in fz["arms"]:
                    if (work / "runs" / f"{arm}-{size}-{cell}-s{seed}" / "done.json").exists():
                        continue
                    print(f"{a.run1} {arm} {size} {cell} {seed} {fz[arm]['lr'][size]} {fz[arm]['mask']} {STEPS[arm]}")


if __name__ == "__main__":
    main()
