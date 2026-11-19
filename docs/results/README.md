# Results convention

Every finished study writes its aggregate results to git in one folder:

```
results/<study>/<date>-<short-commit>/
  result.json
  summary.md
```

`<study>` is the lower-case study id (`k1`, `k7`). `<date>` is the UTC date of the run as `YYYY-MM-DD`. `<short-commit>` is the first 7 characters of the code commit that produced the result.

Earlier prose reports in this directory (`docs/results/*.md`) predate this convention and stay where they are.

## result.json

Provenance for the run. All keys are required. Use `null` where a key does not apply, for example `prereg_sha256` for a study with no pre-registration.

| Key | Meaning |
|---|---|
| `study` | Study id, same as the folder name |
| `issue` | Number of the experiment issue in this repo |
| `commit` | Full git commit of the code that ran |
| `config_sha256` | Hash of the full job configuration |
| `data_manifest_sha256` | Hash of the list of input files and their versions |
| `prereg_sha256` | SHA-256 of the pre-registration file the run followed |
| `seeds` | List of seeds used |
| `starts` | Number of times the run started. 1 means it was never resumed |
| `gpu_hours` | GPU-hours used, summed over all jobs |
| `package_versions` | Versions of the main packages |

## summary.md

The primary metric first, with its interval and the cell it belongs to. Then secondary numbers, each labelled secondary. Then the checks and kill rules from the pre-registration, with pass or fail. A deviation from the pre-registration is stated here and added to the Deviations section of the pre-registration file.

## What goes in git

Only aggregate numbers: tables, intervals, counts, and plots made from them.

Never commit participant-level or recording-level data, NSRR files, or anything derived from them per subject or per night. The NSRR data use agreement covers those. Checkpoints, per-night outputs, feature caches and logs stay off git, on the compute host or in storage that the data licence allows. `summary.md` may name where they are.

## Finishing a study

1. Open a PR that adds the results folder. Link the experiment issue.
2. After the PR merges, tag the merge commit as described in `docs/RELEASING.md`.
3. Update the issue: status label, compute used, results link.
