"""Per-night feature caches that record what produced them.

A cache dir holds one ``<cohort>-<subject>.npz`` per night plus
``cache_identity.<cohort>.json``. The identity records the producing script, the
cohort, the resolved data root and layout, the SHA-256 of the participant ID list,
the git commit (and a hash of uncommitted changes) and the KAHLUS_MESA mode. Each
npz carries the identity hash under ``_cache_identity``.

Readers refuse a dir that has cached nights but no identity file, a dir whose
identity differs from the current run, and any npz whose identity key is missing,
different or whose subject is not on the identity's ID list. A cache therefore
can't be reused across data roots, participant lists or commits.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from kahlus.nsrr import _REPO, DiscoveryReport, ids_sha256, mesa_mode

IDENTITY_KEY = "_cache_identity"
SCHEMA = 1


class CacheIdentityError(RuntimeError):
    pass


def git_state() -> dict:
    """Commit, dirty flag and a hash of the uncommitted diff (None outside a git checkout)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(_REPO), *args], capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        diff = hashlib.sha256(git("diff", "HEAD").encode()).hexdigest() if dirty else None
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "diff_sha256": None}
    return {"commit": commit, "dirty": dirty, "diff_sha256": diff}


def _hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class NightCache:
    def __init__(self, root: Path, cohort: str, identity: dict, ids: list[str]):
        self.root, self.cohort, self.identity, self.ids = Path(root), cohort, identity, frozenset(ids)
        self.hash = _hash(identity)

    @staticmethod
    def identity_path(root: Path, cohort: str) -> Path:
        return Path(root) / f"cache_identity.{cohort}.json"

    @classmethod
    def bind(cls, root: str | Path, cohort: str, *, producer: str, report: DiscoveryReport, ids) -> NightCache:
        """Producer side: create the cache with this run's identity, or reopen one that has exactly it."""
        root, ids = Path(root), sorted(ids)
        identity = {
            "schema": SCHEMA, "producer": producer, "cohort": cohort,
            "data_root": report.root, "layout": report.layout,
            "edf_dir": report.edf_dir, "annotation_dir": report.annotation_dir,
            "ids_sha256": ids_sha256(ids), "n_ids": len(ids),
            "kahlus_mesa": mesa_mode(), **git_state(),
        }
        root.mkdir(parents=True, exist_ok=True)
        path = cls.identity_path(root, cohort)
        body = identity | {"identity_sha256": _hash(identity), "ids": ids}
        if not path.exists():
            stray = sorted(root.glob(f"{cohort}-*.npz"))
            if stray:
                raise CacheIdentityError(
                    f"{root} holds {len(stray)} cached {cohort} nights but no {path.name}; "
                    "refusing a cache of unknown origin. Use a fresh output dir.")
            try:
                with open(path, "x") as f:  # exclusive: two binders cannot both write it
                    json.dump(body, f, indent=2)
                return cls(root, cohort, identity, ids)
            except FileExistsError:
                pass
        got = json.loads(path.read_text())
        if got.get("identity_sha256") != body["identity_sha256"]:
            diff = sorted(k for k in identity if got.get(k) != identity[k])
            raise CacheIdentityError(f"{path} was written by a different run (differs in: {', '.join(diff) or 'hash'}). "
                                     "Use a fresh output dir.")
        return cls(root, cohort, identity, ids)

    @classmethod
    def open(cls, root: str | Path, cohort: str, *, producer: str) -> NightCache:
        """Reader side: an existing cache made by ``producer`` from this commit and MESA mode."""
        path = cls.identity_path(root, cohort)
        if not path.exists():
            raise CacheIdentityError(f"no {path.name} in {root}; refusing a cache of unknown origin")
        body = json.loads(path.read_text())
        identity = {k: v for k, v in body.items() if k not in ("identity_sha256", "ids")}
        if _hash(identity) != body.get("identity_sha256") or ids_sha256(body.get("ids", [])) != identity.get("ids_sha256"):
            raise CacheIdentityError(f"{path} does not match its own hash")
        here = {"producer": producer, "kahlus_mesa": mesa_mode(), **git_state()}
        diff = sorted(k for k, v in here.items() if identity.get(k) != v)
        if diff:
            raise CacheIdentityError(f"{path} does not match this run (differs in: {', '.join(diff)})")
        return cls(Path(root), cohort, identity, body["ids"])

    def path(self, subject_id: str) -> Path:
        return self.root / f"{self.cohort}-{subject_id}.npz"

    def has(self, subject_id: str) -> bool:
        return self.path(subject_id).exists()

    def subjects(self) -> list[str]:
        prefix = f"{self.cohort}-"
        return sorted(f.stem[len(prefix):] for f in self.root.glob(f"{prefix}*.npz"))

    def save(self, subject_id: str, **arrays) -> None:
        if subject_id not in self.ids:
            raise CacheIdentityError(f"{self.cohort} {subject_id} is not on this cache's ID list")
        tmp = self.root / f".{self.cohort}-{subject_id}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **arrays, **{IDENTITY_KEY: np.array(self.hash)})
        os.replace(tmp, self.path(subject_id))

    def load(self, subject_id: str) -> dict[str, np.ndarray]:
        f = self.path(subject_id)
        if subject_id not in self.ids:
            raise CacheIdentityError(f"{f} is not on this cache's ID list")
        with np.load(f) as z:
            if IDENTITY_KEY not in z.files or str(z[IDENTITY_KEY]) != self.hash:
                raise CacheIdentityError(f"{f} lacks this cache's identity; refusing it")
            return {k: z[k] for k in z.files if k != IDENTITY_KEY}
