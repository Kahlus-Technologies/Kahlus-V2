# MESA exposure record (2026-11-07)

MESA is the held-out test cohort (original plan: Kahlus-Program `kahlus-sleep/cluster/HANDOFF_KRISH.md`,
"MESA IS THE SEALED TEST SET"). The v2 dev-slice work (2026-11-04/05) used MESA during development
without a formal decision to drop that reservation. This file records what was seen.

## What was seen

| Exposure | Scope | Influenced choices? |
|---|---|---|
| Arousal census (`scripts/nsrr_arousal_census.py`) | Annotation XMLs for **all** MESA participants: event counts and types, no signals | Yes: clean-spontaneous arousal definition, anchor rules |
| Dev-slice studies (heart forecast, beat warning, conformal, CLAS replay, montage stress, event-locked HR, autonomic RFS gate) | Full signals for the **25** nights in `configs/data/mesa_exposed_ids.txt` (first 25 in NSRR listing order, not random) | Yes: normalization choice (causal picked after seeing raw / night-z / causal), review fixes, thresholds |
| README hero | MESA night 0002 | No |
| Earlier P01 (`kahlus-sleep`) | SHHS-1 pilot only; MESA sealed in code | No MESA exposure found locally |
| `sweep_mesa.log` | Not found locally | **Unknown, check the cluster** |
| P02 pretraining / earlier cluster runs | Not visible locally | **Unknown, check the cluster** (`data/zarr_mesa_SEALED/`, `data/MESA_LOOK_USED`, `SHHS_SELECT_FROZEN.json`) |

## Rule from here (enforced in `neurotwin.adapters.nsrr.discover_nsrr_recordings`)

- `KAHLUS_MESA=dev` (default): MESA returns only the 25 exposed nights. A full-cohort run cannot widen exposure.
- `KAHLUS_MESA=final`: MESA returns only never-seen nights, and refuses to run until
  `docs/results/full_cohort_plan.md` (frozen settings) is committed. Used once.
- Development and selection happen on full SHHS-1.

## Disclosure text for the paper

"Annotation-level event counts for all MESA participants and full signals for 25 participants were
examined during development. Those 25 participants are excluded from the final MESA evaluation."

If the cluster audit shows MESA signals were used for training or selection, MESA is reported as
exploratory and confirmation moves to a separate approved cohort (MrOS / CFS / SOF / WSC).
