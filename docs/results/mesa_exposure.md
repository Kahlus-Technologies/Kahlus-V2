# MESA exposure record

MESA is the held-out test cohort (original plan: Kahlus-Program `kahlus-sleep/cluster/HANDOFF_KRISH.md`,
"MESA IS THE SEALED TEST SET"). The v2 dev-slice work (2026-11-04/05) used MESA during development
without a formal decision to drop that reservation. This file records what was seen and what it changed.
Revised 2026-11-08 after Krish's cluster audit; the 2026-11-07 version overstated the census influence
and understated the guard gaps.

## Participant manifests (committed, `docs/results/provenance/`)

| File | What | Count |
|---|---|---|
| `mesa_dev_slice_edfs.txt` | MESA nights whose full signals were used (= `configs/data/mesa_exposed_ids.txt`) | 25 (0001 to 0079, first 25 in NSRR listing order, not random) |
| `shhs_dev_slice_edfs.txt` | SHHS-1 dev nights | 25 (200001 to 200025) |
| `mesa_census_xmls.txt` | MESA annotation XMLs read by the census | 2,056 (all) |
| `shhs_census_xmls.txt` | SHHS-1 annotation XMLs listed; 203583 failed to download, so 5,792 read | 5,793 |

Fetch script: `scripts/nsrr_fetch_slice.sh` (same file that ran). Per-subject census output is
`artifacts/nsrr_arousal_census/{mesa,shhs}_per_subject.jsonl` (not committed: per-participant NSRR
derived data; sha256 `1fcb16cc...` MESA, `73358b54...` SHHS). Summary: `artifacts/nsrr_arousal_census/summary.json`.

## What the MESA annotation census changed

Short answer: no definition or anchor parameter changed in response to it.

- The subtype rules (respiratory = onset within 5 s after an apnea/hypopnea, limb = within 0.5 s of a
  limb movement, clean = 60 s quiet in sleep before onset) were written in the census script itself,
  in the same commit that produced the first output (`2aea248`, 2026-11-04 17:05). The values never
  changed afterwards (git `-S` history; the session log shows no other values were tried).
- The forecasting anchor rule (`QUIET_S = 60` in `heart_arousal.py`, `ef1bb4c`, 18:47) copies `CLEAN_PRE_S`.
  It has stayed 60 s. `f83c029` changed how the quiet window counts events (whole duration instead of onset only). That was a code-review fix, checked on the 25 dev nights, not on census numbers.
- What the census did inform: (1) a go decision. Counts were large enough (median 27 clean arousals
  per MESA subject) to keep clean spontaneous arousals as the pre-registered primary target of #1. Had
  counts been tiny, the target would have changed, so this is a feasibility look at all 2,056 MESA
  label sets. (2) Nothing else: the ECG-only transfer note comes from the SHHS montage (no PPG), not MESA.
- No signals, features or model outputs from the 2,031 non-dev MESA participants were computed or seen.
- Limit: the evidence is git history and the agent session log. Nothing timestamped the parameters before the census code ran.

## Signal-level use of the 25 MESA nights

Dev-slice studies (heart forecast, beat warning, conformal, CLAS replay, montage stress, event-locked
HR, and the v1 autonomic RFS gate) trained, cross-validated and tuned on these nights, including the normalization
choice (causal picked after seeing raw / night-z / causal), quiet-window and beat-mask fixes, and
thresholds. README hero is MESA 0002. All 25 are excluded from the final evaluation.

## Earlier pipeline (Kahlus-Program `kahlus-sleep`)

- P01: SHHS-1 only (`manifests/shhs1_pilot_manifest.json`, local stores `data/zarr` = 59 SHHS-1 nights,
  no MESA). Krish confirmed the recorded cluster P01 run was SHHS only.
- P02: code refuses the sealed MESA store (`src/kahlus_sleep/p02/train.py` `refuse_sealed_store`,
  `cluster/run_p02_pretrain.py`). Launch path: `cluster/run_p02_pretrain.sbatch` (step 7 of
  `HANDOFF_KRISH.md`), run dirs `runs/p02_<size>_<arch>_<init>/`, logs `logs/p02_<jobid>.out`,
  checkpoints `latest.ckpt` / `step_*.ckpt` carrying `config` (incl. `store`) and `config_hash`.
  No P02 run dir, log or checkpoint exists locally, and the Kahlus-V1 `a100-run-history.csv` records only MoABB smokes.
  Whether P02 ever ran is only answerable on the cluster.
- `sweep_mesa.log` (cluster): download verification, not tuning (Krish, 2026-11-08).

## Seal (enforced in `kahlus.nsrr.gate_mesa_ids`)

Every MESA reader goes through it: EDF discovery, the annotation census, and feature caches (`run_conformal_alarm.py`).
- `KAHLUS_MESA=dev` (default): only the 25 exposed nights.
- `KAHLUS_MESA=final`: only never-seen nights. It refuses unless `docs/results/full_cohort_plan.md` is tracked in
  git and the tree is clean. The commit, plan sha256, argv and time are logged to `docs/results/mesa_final_looks/<script>.json`.
  A second process for the same script refuses (single use).
- Evaluation-only in final mode: no MESA CV, no MESA-trained transfer, no MESA thresholds or calibration.
  Scripts run SHHS-trained models on MESA only. (In Kahlus-V1 the autonomic RFS gate, which cross-validates within MESA, refuses outright; it is not part of this repo.)
- Known ceiling: the look ledger is a local file. It is committed with results so a rerun is visible in
  git, but it is not tamper-proof.

## Disclosure text for the paper

"Annotation-level event counts for all MESA participants were examined during development to confirm
the primary target had enough events; no event definitions were changed afterwards. Full signals for
25 MESA participants were used for development and are excluded from the final MESA evaluation, which
used a model and analysis plan frozen on SHHS-1 and was run once."

If the cluster shows P02 or any other run trained or selected on MESA signals, MESA is reported as
exploratory and confirmation moves to a separate approved cohort (MrOS / CFS / SOF / WSC).
