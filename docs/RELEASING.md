# Releasing

Two kinds of tag, so code versions and result snapshots do not mix.

## Code: `v0.x.y`

Semantic versions, kept in step with `version` in `pyproject.toml`. Bump the minor number for a new feature or a change in behaviour, and the patch number for a fix. Stay below 1.0 until the captain says otherwise.

```
git tag -a v0.3.0 -m "v0.3.0" <commit>
git push origin v0.3.0
```

## Pre-registration: `prereg-<study>-v1`

After the PR that adds a file to `docs/prereg/` is merged, tag the merge commit with an annotated tag. This fixes the date before any confirmatory run. Do not tag before the merge.

```
git tag -a prereg-k7-v1 -m "Pre-registration for K7" <merge-commit>
git push origin prereg-k7-v1
```

A change after results exist goes in the Deviations section of the file. If the plan itself must change before any result, tag `-v2`.

## Results: `results-<study>-<date>-<short-commit>`

Tag the merge commit that added `results/<study>/<date>-<short-commit>/`. The tag name repeats the folder name, for example `results-k7-2026-12-03-ab12cd3`.

```
git tag -a results-k7-2026-12-03-ab12cd3 -m "K7 results" <merge-commit>
git push origin results-k7-2026-12-03-ab12cd3
```

The workflow `.github/workflows/results-release.yml` then creates a GitHub release. The notes link the experiment issue (the `issue` key in `result.json`), the pre-registration tag if one exists, and the results folder. The folder is attached as a zip. The job fails if the folder lacks `result.json` or `summary.md`, or holds data or checkpoint files (`.edf`, `.pt`, `.ckpt`, `.npy`, `.npz`, `.parquet`, `.h5`).
