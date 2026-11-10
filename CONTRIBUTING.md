# Contributing

- Branch from `main`, open a pull request, and keep commits small with full-sentence messages.
- Before pushing: `ruff check .` and `pytest`. CI runs both on Python 3.10 and 3.12.
- A change to a study script or `src/kahlus` must not move a committed result unless that is the point of the change. Prove it with `scripts/check_artifacts_unchanged.sh`.
- Never commit NSRR data or per-participant derived files. Aggregate JSON reports and participant ID lists are fine.
- Never set `KAHLUS_MESA=final` outside the single, planned final evaluation. See `docs/results/mesa_exposure.md`.
