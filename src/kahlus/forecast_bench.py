"""Score a pretraining checkpoint against forecasting baselines under the honest-eval rules (issue #11).

    python -m kahlus.forecast_bench --ckpt RUN/ckpt-XXXXXXXX.pt --shards DIR --test-dataset shhs --out report.json

Task: from ``context`` seconds of signal, forecast the next ``horizon`` seconds. Every model reads the
context only, so every score is strictly-future. Two splits are scored:

* ``subject_held_out``: the training datasets, subjects that no model was fitted on;
* ``dataset_held_out``: a dataset that no model was fitted on.

Score: ``skill = 1 - MSE / MSE(zero forecast)``; zero is the causal running mean, so skill 0 means
"no better than the mean". Shift control: each model also forecasts the same targets from a context
taken half a night away in the same night. Skill that survives the shift comes from the person, not
from the signal just before the target, so ``gain_over_shift`` is the number to read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kahlus.nsrr import mesa_fit_allowed
from kahlus.nsrr_cache import git_state
from kahlus.shards import ShardSet, Windows

TUNE_MOD = 5  # one fit subject in five tunes baseline hyperparameters instead of fitting them


class BaselineUnavailable(RuntimeError):
    """The baseline cannot run in this environment. The report records the reason instead of a number."""


def is_tuning(night: dict) -> bool:
    return int(hashlib.sha256(f"tune:{night['dataset']}:{night['subject']}".encode()).hexdigest(), 16) % TUNE_MOD == 0


def eval_windows(shards: ShardSet, nights: list[dict], *, context: int, horizon: int, n: int, seed: int, shift: bool = False) -> dict:
    """``n`` windows as float32 arrays: ``context``, ``target`` and the subject of each window. With ``shift``
    also ``shifted``, the context from half a night away. Only scored windows need it."""
    length = context + horizon
    nights = [x for x in nights if x["n"] >= 4 * length]  # room for a shifted context that cannot overlap the target
    drawn = Windows(shards, nights, length).draw(np.random.default_rng(seed), n)
    full = np.stack([shards.read(night, start, length) for night, start in drawn]).astype(np.float32)
    out = {"context": full[:, :context], "target": full[:, context:],
           "subject": np.asarray([f"{night['dataset']}:{night['subject']}" for night, _ in drawn])}
    if shift:
        out["shifted"] = np.stack([shards.read(night, (start + night["n"] // 2) % (night["n"] - length + 1), context)
                                   for night, start in drawn]).astype(np.float32)
    return out


# Forecasters. Each builder takes (fit, tune, args) and returns ``predict(context) -> (n, horizon, channels, patch_len)``.

def zero_forecaster(fit, tune, args):
    return lambda c: np.zeros((c.shape[0], args.horizon, *c.shape[2:]), dtype=np.float32)


def persistence_forecaster(fit, tune, args):
    return lambda c: np.repeat(c[:, -1:], args.horizon, axis=1)


def ridge_forecaster(fit, tune, args):
    """Ridge from the last k context seconds to the whole target. k and alpha are tuned on the tuning subjects."""
    from sklearn.linear_model import Ridge

    def x(c, k):
        return c[:, -k:].reshape(c.shape[0], -1)

    y = fit["target"].reshape(fit["target"].shape[0], -1)
    best = None
    for k in (k for k in (1, 4, 16) if k <= fit["context"].shape[1]):
        for alpha in (1.0, 1e2, 1e4, 1e6):
            model = Ridge(alpha=alpha).fit(x(fit["context"], k), y)
            mse = float(np.mean((model.predict(x(tune["context"], k)).reshape(tune["target"].shape) - tune["target"]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, k, model)
    _, k, model = best
    return lambda c: model.predict(x(c, k)).reshape(c.shape[0], *fit["target"].shape[1:]).astype(np.float32)


def gru_forecaster(fit, tune, args):
    """One-layer GRU over the context seconds; the weights with the lowest tuning-subject MSE are kept."""
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    shape = fit["target"].shape[1:]
    gru = nn.GRU(int(np.prod(fit["context"].shape[2:])), args.gru_hidden, batch_first=True)
    head = nn.Linear(args.gru_hidden, int(np.prod(shape)))
    opt = torch.optim.Adam([*gru.parameters(), *head.parameters()], lr=1e-3)

    def run(c):
        c = torch.as_tensor(c)
        return head(gru(c.flatten(2))[0][:, -1]).view(c.shape[0], *shape)

    xs, ys = torch.as_tensor(fit["context"]), torch.as_tensor(fit["target"])
    rng = np.random.default_rng(args.seed)
    best = (np.inf, None)
    for step in range(1, args.gru_steps + 1):
        idx = torch.as_tensor(rng.integers(0, xs.shape[0], size=64))
        loss = ((run(xs[idx]) - ys[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.gru_steps:
            with torch.no_grad():
                mse = float(((run(tune["context"]) - torch.as_tensor(tune["target"])) ** 2).mean())
            if mse < best[0]:
                best = (mse, [{k: v.clone() for k, v in m.state_dict().items()} for m in (gru, head)])
    gru.load_state_dict(best[1][0])
    head.load_state_dict(best[1][1])

    def predict(c):
        with torch.no_grad():
            return run(c).numpy()

    return predict


def kahlus_forecaster(fit, tune, args):
    import torch

    from kahlus.model import KahlusPatchModel, ModelConfig

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = KahlusPatchModel(ModelConfig(**state["identity"]["model"]))
    model.load_state_dict(state["model"])
    model.eval().to(args.device)

    def predict(c):
        out = [model.forecast(torch.as_tensor(c[i : i + 256]).to(args.device)).float().cpu() for i in range(0, c.shape[0], 256)]
        return torch.cat(out).numpy()

    return predict


def timesfm3_forecaster(fit, tune, args):
    """Entry point for zero-shot TimesFM-3, named in issue #11. Not wired: it needs the TimesFM package
    and its pretrained weights, and neither is available on the build host or in CI. To add it, return a
    ``predict(context)`` that feeds each channel of the context to the model as one univariate series
    and returns ``(n, horizon, channels, patch_len)`` in the same z-scored units."""
    raise BaselineUnavailable("TimesFM-3 needs its package and pretrained weights; wire it in timesfm3_forecaster")


def sleepfm_forecaster(fit, tune, args):
    """Entry point for SleepFM. Not wired: it needs the SleepFM code and pretrained weights. SleepFM is an
    encoder, so it also needs a forecasting probe: fit a ridge from its embedding of the context to the
    target on ``fit``, tune alpha on ``tune``, and return that probe's ``predict(context)``."""
    raise BaselineUnavailable("SleepFM needs its code, pretrained weights and a forecasting probe; wire it in sleepfm_forecaster")


FORECASTERS = {
    "zero": zero_forecaster,
    "persistence": persistence_forecaster,
    "ridge": ridge_forecaster,
    "gru": gru_forecaster,
    "timesfm3": timesfm3_forecaster,
    "sleepfm": sleepfm_forecaster,
    "kahlus": kahlus_forecaster,
}


def skill_with_ci(se: np.ndarray, se_zero: np.ndarray, subject: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Skill and its 95% interval from resampling whole subjects."""
    _, inv = np.unique(subject, return_inverse=True)
    model, zero = np.bincount(inv, weights=se), np.bincount(inv, weights=se_zero)
    pick = np.random.default_rng(seed).integers(0, model.size, size=(n_boot, model.size))
    boot = 1.0 - model[pick].sum(1) / zero[pick].sum(1)
    return {"skill": float(1.0 - model.sum() / zero.sum()),
            "skill_ci_low": float(np.percentile(boot, 2.5)), "skill_ci_high": float(np.percentile(boot, 97.5))}


def score(predict, windows: dict, *, n_boot: int, seed: int) -> dict:
    def se(context):  # squared error per window and horizon second
        return ((predict(context) - windows["target"]) ** 2).mean(axis=(2, 3))

    zero = (windows["target"] ** 2).mean(axis=(2, 3))
    aligned, shift = se(windows["context"]), se(windows["shifted"])
    out = skill_with_ci(aligned.mean(1), zero.mean(1), windows["subject"], n_boot=n_boot, seed=seed)
    out["skill_shift_control"] = float(1.0 - shift.sum() / zero.sum())
    out["gain_over_shift"] = out["skill"] - out["skill_shift_control"]
    out["skill_by_horizon_s"] = (1.0 - aligned.sum(0) / zero.sum(0)).tolist()
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--test-dataset", help="dataset no model is fitted on; omitted: only the subject-held-out split is scored")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fit", type=int, default=20000, help="windows the baselines are fitted on")
    ap.add_argument("--n-eval", type=int, default=5000, help="windows per scored split")
    ap.add_argument("--gru-steps", type=int, default=3000)
    ap.add_argument("--gru-hidden", type=int, default=256)
    ap.add_argument("--bootstrap", choices=("smoke", "claim"), default="smoke", help="smoke: 300 resamples, claim: 2000")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch

    identity = torch.load(args.ckpt, map_location="cpu", weights_only=True)["identity"]
    train_datasets = identity["train_datasets"]
    if args.test_dataset in train_datasets:
        raise SystemExit(f"{args.test_dataset} is a training dataset of this checkpoint; it cannot be the held-out dataset")
    if "mesa" in train_datasets and not mesa_fit_allowed():
        raise SystemExit("KAHLUS_MESA=final is evaluation-only: baselines cannot be fitted on MESA")
    shards = ShardSet(args.shards)
    args.horizon = identity["model"]["horizon"]
    cut = {"context": identity["model"]["context"], "horizon": args.horizon}

    fit_nights = shards.select(train_datasets, heldout=False)
    fit = eval_windows(shards, [n for n in fit_nights if not is_tuning(n)], **cut, n=args.n_fit, seed=args.seed)
    tune = eval_windows(shards, [n for n in fit_nights if is_tuning(n)], **cut, n=args.n_eval, seed=args.seed + 1)
    splits = {
        "subject_held_out": eval_windows(shards, shards.select(train_datasets, heldout=True), **cut, n=args.n_eval, seed=args.seed + 2, shift=True),
    }
    if args.test_dataset:
        splits["dataset_held_out"] = eval_windows(shards, shards.select([args.test_dataset]), **cut, n=args.n_eval, seed=args.seed + 3, shift=True)
    n_boot = 2000 if args.bootstrap == "claim" else 300
    report = {
        "task": f"forecast {cut['horizon']} s from {cut['context']} s, strictly future",
        "checkpoint": {"path": str(args.ckpt), "size": identity["size"], "train_datasets": train_datasets,
                       "steps": identity["steps"], "shards_sha256": identity["shards_sha256"]},
        "test_dataset": args.test_dataset, "shards_sha256": shards.sha256, "git": git_state(),
        "bootstrap_mode": args.bootstrap, "bootstrap_samples": n_boot,
        "splits": {name: {"n_windows": int(w["subject"].size), "n_subjects": int(np.unique(w["subject"]).size), "models": {}}
                   for name, w in splits.items()},
    }
    for name, build in FORECASTERS.items():
        try:
            predict = build(fit, tune, args)
        except BaselineUnavailable as exc:
            for split in report["splits"].values():
                split["models"][name] = {"status": "not_run", "reason": str(exc)}
            continue
        for split, windows in splits.items():
            report["splits"][split]["models"][name] = {"status": "ok"} | score(predict, windows, n_boot=n_boot, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for split, body in report["splits"].items():
        for name, m in body["models"].items():
            print(f"{split:18s} {name:12s} " + (f"skill {m['skill']:+.4f}  shift {m['skill_shift_control']:+.4f}" if m["status"] == "ok" else m["reason"]))
    return report


if __name__ == "__main__":
    main()
