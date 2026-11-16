#!/usr/bin/env bash
# Pack the small results zip that the A100 operator sends back.
#
#   scripts/package_evidence.sh <work-dir> <out.zip>
#
# Takes only text records from <work-dir>: summaries, metrics, configs, environment records,
# manifests and logs (.json .jsonl .md .txt .tsv .log .csv, each at most 20 MB).
# It never takes checkpoints, data, tokens, keys or .env files. Python environments, NSRR data,
# shards and feature caches are skipped as whole directories. A file that holds the value of
# ~/.nsrr_token or a private key block is skipped too. The zip lists every file it holds with its
# SHA-256 in MANIFEST.sha256, and every skipped file with the reason in EXCLUDED.txt.
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
import sys
import zipfile
from pathlib import Path

KEEP_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".tsv", ".log", ".csv"}
MAX_BYTES = 20 * 1024 * 1024
SKIP_DIRS = ["venv*", "nsrr", "shards", "*.partial", "cache", "home", "__pycache__", ".git"]
SECRET = [".env", ".env.*", "*.env", "*token*", "*secret*", "*password*", "*passwd*", "pw.txt", ".netrc",
          "*.pem", "*.key", "*.p12", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", "credentials*"]
CHECKPOINT = ["ckpt-*", "*.pt", "*.pt.tmp", "*.pth", "*.ckpt", "*.safetensors"]
DATA = ["*.edf", "*.xml", "*.npy", "*.npz", "*_per_subject.jsonl"]  # NSRR data and data derived per night
PRIVATE_KEY = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def matches(name, patterns):
    return any(fnmatch.fnmatch(name.lower(), p) for p in patterns)


def reason(path, body_check):
    name = path.name
    for label, patterns in (("secret", SECRET), ("checkpoint", CHECKPOINT), ("data", DATA)):
        if matches(name, patterns):
            return label
    if path.suffix.lower() not in KEEP_SUFFIXES:
        return "not a text record"
    if path.stat().st_size > MAX_BYTES:
        return "larger than 20 MB"
    return body_check(path.read_bytes())


def main(work, out):
    work, out = Path(work).resolve(), Path(out).resolve()
    token_file = Path.home() / ".nsrr_token"
    token = token_file.read_bytes().strip() if token_file.is_file() else b""

    def body_check(body):
        if token and token in body:
            return "holds the NSRR token"
        if PRIVATE_KEY.search(body):
            return "holds a private key"
        return None

    top = out.stem
    kept, skipped = [], []
    for root, dirs, files in os.walk(work):
        rel_root = Path(root).relative_to(work)
        for d in sorted(dirs):
            if matches(d, SKIP_DIRS):
                skipped.append((rel_root / d, "skipped directory"))
        dirs[:] = sorted(d for d in dirs if not matches(d, SKIP_DIRS))
        for f in sorted(files):
            path = Path(root) / f
            if path == out or path.is_symlink():
                continue
            why = reason(path, body_check)
            if why:
                skipped.append((rel_root / f, why))
            else:
                kept.append(rel_root / f)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    manifest = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in kept:
            body = (work / rel).read_bytes()
            manifest.append(f"{hashlib.sha256(body).hexdigest()}  {rel.as_posix()}\n")
            z.writestr(f"{top}/{rel.as_posix()}", body)
        z.writestr(f"{top}/MANIFEST.sha256", "".join(manifest))
        z.writestr(f"{top}/EXCLUDED.txt", "".join(f"{rel.as_posix()}\t{why}\n" for rel, why in skipped))
    os.replace(tmp, out)
    print(f"{out}: {len(kept)} files, {out.stat().st_size // 1024} KB; {len(skipped)} skipped (see EXCLUDED.txt)")


main(*sys.argv[1:])
PY
