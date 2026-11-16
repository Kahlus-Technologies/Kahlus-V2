#!/usr/bin/env bash
# Pack the small results zip that the A100 operator sends back.
#
#   scripts/package_evidence.sh <work-dir> <out.zip>
#
# Takes only the run outputs of a100_handoff.sh in <work-dir> (verify.txt, preflight/, smoke/, devslice/,
# runs/, logs/), and of those only text records: summaries, metrics, configs, environment records,
# manifests and logs (.json .jsonl .md .txt .tsv .log .csv, each at most 20 MB).
# It never takes checkpoints, data, tokens, keys or .env files: it skips them by file name, and it skips
# every file whose text holds the value of ~/.nsrr_token, a private key block, a known token format or a
# KEY=value style secret. Python environments, NSRR data, shards and feature caches are skipped as whole
# directories. Symlinks, special files, unreadable files and odd file names are skipped, never fatal.
# The zip lists every file it holds with its SHA-256 in MANIFEST.sha256, and every skipped path with the
# reason in EXCLUDED.txt.
set -euo pipefail

if (($# != 2)); then
  echo "usage: scripts/package_evidence.sh <work-dir> <out.zip>" >&2
  exit 2
fi
[[ -d $1 ]] || { echo "package_evidence: $1 is not a directory" >&2; exit 2; }

exec python3 - "$1" "$2" <<'PY'
import fnmatch
import hashlib
import os
import re
import stat
import sys
import zipfile
from pathlib import Path

RUN_OUTPUTS = {"verify.txt", "preflight", "smoke", "devslice", "runs", "logs"}
KEEP_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".tsv", ".log", ".csv"}
MAX_BYTES = 20 * 1024 * 1024
SKIP_DIRS = ["venv*", "nsrr*", "shards", "*.partial", "cache", "home", "__pycache__", ".git"]
SECRET = [".env", ".env.*", "*.env", "*token*", "*secret*", "*password*", "*passwd*", "pw.txt", ".netrc", "*key*",
          "*credential*", "*.pem", "*.p12", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*"]
CHECKPOINT = ["ckpt-*", "*.pt", "*.pt.tmp", "*.pth", "*.ckpt", "*.safetensors"]
DATA = ["*.edf", "*.xml", "*.npy", "*.npz", "*_per_subject.jsonl", "*-dataset-*.csv"]  # NSRR data, per night or per subject
SECRET_TEXT = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    rb"|\b(?:tskey-[a-z]+-|hf_|gh[opsur]_|github_pat_|glpat-|xox[abprs]-|sk-)[A-Za-z0-9_-]{16,}"
    rb"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"
    rb"|(?i:[a-z0-9_]*(?:token|secret|api_?key|access_?key|password)[a-z0-9_]*[\"']?\s*[=:]\s*[\"']?[^\s\"',}]{8,})")
ODD_NAME = re.compile(r"[\x00-\x1f\x7f]|[\ud800-\udfff]")


def matches(name, patterns):
    return any(fnmatch.fnmatch(name.lower(), p) for p in patterns)


def shown(rel):
    return ascii(rel.as_posix())[1:-1]  # EXCLUDED.txt stays one line per path, whatever the name holds


def check(path, token):
    """Return (reason to skip, None) or (None, file body)."""
    name = path.name
    if ODD_NAME.search(name):
        return "odd file name", None
    for label, patterns in (("secret", SECRET), ("checkpoint", CHECKPOINT), ("data", DATA)):
        if matches(name, patterns):
            return label, None
    if path.suffix.lower() not in KEEP_SUFFIXES:
        return "not a text record", None
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            return "not a regular file", None
        if st.st_size > MAX_BYTES:
            return "larger than 20 MB", None
        body = path.read_bytes()
    except OSError as exc:
        return f"unreadable ({exc.strerror})", None
    if token and token in body:
        return "holds the NSRR token", None
    if SECRET_TEXT.search(body):
        return "holds a secret", None
    return None, body


def main(work, out):
    work, out = Path(work).resolve(), Path(out).resolve()
    token_file = Path.home() / ".nsrr_token"
    try:
        token = token_file.read_bytes().strip()
    except OSError:
        token = b""

    top = out.stem
    kept, skipped = [], []
    for root, dirs, files in os.walk(work):
        rel_root = Path(root).relative_to(work)
        entries = sorted(dirs + files)
        dirs[:] = []
        for name in entries:
            path, rel = Path(root) / name, rel_root / name
            if path == out:
                continue
            if rel_root == Path(".") and name not in RUN_OUTPUTS:
                skipped.append((rel, "not a run output"))
            elif path.is_symlink():
                skipped.append((rel, "symlink"))
            elif path.is_dir():
                if matches(name, SKIP_DIRS):
                    skipped.append((rel, "skipped directory"))
                else:
                    dirs.append(name)
            else:
                why, body = check(path, token)
                if why:
                    skipped.append((rel, why))
                else:
                    kept.append((rel, body))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, body in kept:
            z.writestr(f"{top}/{rel.as_posix()}", body)
        z.writestr(f"{top}/MANIFEST.sha256", "".join(f"{hashlib.sha256(b).hexdigest()}  {r.as_posix()}\n" for r, b in kept))
        z.writestr(f"{top}/EXCLUDED.txt", "".join(f"{shown(rel)}\t{why}\n" for rel, why in skipped))
    os.replace(tmp, out)
    print(f"{out}: {len(kept)} files, {out.stat().st_size // 1024} KB; {len(skipped)} skipped (see EXCLUDED.txt)")


main(*sys.argv[1:])
PY
