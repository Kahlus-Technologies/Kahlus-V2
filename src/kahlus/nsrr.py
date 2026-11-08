"""NSRR (National Sleep Research Resource) adapter — MESA / SHHS PSG + XML annotations.

Credentialed download required from https://sleepdata.org/. Raw data never committed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

AROUSAL_CONCEPT_RE = re.compile(r"arousal", re.IGNORECASE)
# Real NSRR labels: MESA EEG3 = C4-M1 (EEG1 = Fz-Cz, EEG2 = Cz-Oz); SHHS1 "EEG" = C4-A1, "EEG(sec)" = C3-A2.
# Order is priority: the C4 derivation shared across MESA and SHHS comes first.
EEG_CHANNEL_CANDIDATES = ("EEG C4-A1", "EEG3", "EEG", "EEG C3-A2", "EEG(sec)", "EEG Fpz-Cz", "EEG C3", "EEG C4")
ECG_CHANNEL_CANDIDATES = ("ECG", "EKG", "ECG1", "ECG2")
PPG_CHANNEL_CANDIDATES = ("Pleth", "PPG", "Plethysmogram")
EOG_CHANNEL_CANDIDATES = ("EOG E1-A2", "EOG E2-A1", "EOG", "EOG Left", "EOG Right", "EOG-L", "EOG-R", "EOG(L)", "EOG(R)")
EMG_CHANNEL_CANDIDATES = ("EMG Chin", "EMG", "Chin")
RESP_CHANNEL_CANDIDATES = ("AIRFLOW", "New AIR", "Flow", "Nasal Pressure", "Chest", "Abdomen", "Thor", "Abdo", "THOR RES", "ABDO RES")


@dataclass(frozen=True)
class NsrrRecording:
    edf_path: Path
    xml_path: Path
    subject_id: str
    dataset: str


_REPO = Path(__file__).resolve().parents[3]
MESA_EXPOSED_IDS = _REPO / "configs/data/mesa_exposed_ids.txt"
MESA_FROZEN_PLAN = _REPO / "docs/results/full_cohort_plan.md"
MESA_FINAL_LOOKS = _REPO / "docs/results/mesa_final_looks"


def mesa_exposed_ids() -> frozenset[str]:
    lines = MESA_EXPOSED_IDS.read_text().splitlines()
    return frozenset(s.strip() for s in lines if s.strip() and not s.startswith("#"))


def mesa_mode() -> str:
    mode = os.environ.get("KAHLUS_MESA", "dev")
    if mode not in ("dev", "final"):
        raise ValueError(f"KAHLUS_MESA must be 'dev' or 'final', got {mode!r}")
    return mode


def mesa_fit_allowed() -> bool:
    """False in final mode: held-out MESA is evaluation-only (no fitting, CV, thresholds or calibration)."""
    return mesa_mode() != "final"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_REPO), *args], capture_output=True, text=True, check=True).stdout


def _check_frozen() -> dict:
    """Final mode needs the frozen plan tracked in git and a clean tree, so the code that runs is a commit."""
    try:
        _git("ls-files", "--error-unmatch", str(MESA_FROZEN_PLAN))
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        head = _git("rev-parse", "HEAD").strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"KAHLUS_MESA=final needs {MESA_FROZEN_PLAN} committed to git") from exc
    if dirty:
        raise RuntimeError(f"KAHLUS_MESA=final needs a clean git tree (commit everything first):\n{dirty}")
    return {"commit": head, "plan_sha256": hashlib.sha256(MESA_FROZEN_PLAN.read_bytes()).hexdigest()}


def _claim_final_look() -> None:
    """One final look per script. The first call writes a ledger entry; any later process refuses.
    ponytail: a local file can be deleted. The ledger is committed with the results, so a rerun shows
    in git history. A tamper-proof seal (e.g. key held by a third party) is the upgrade path."""
    script = Path(sys.argv[0]).stem or "interactive"
    entry = MESA_FINAL_LOOKS / f"{script}.json"
    if entry.exists():
        if json.loads(entry.read_text()).get("pid") == os.getpid():
            return
        raise RuntimeError(f"MESA final look already used by {script}: {entry}. Rerunning would make it a second look.")
    info = _check_frozen() | {"script": script, "argv": sys.argv, "pid": os.getpid(),
                              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    MESA_FINAL_LOOKS.mkdir(parents=True, exist_ok=True)
    with open(entry, "x") as f:  # exclusive create: two racing processes cannot both claim
        json.dump(info, f, indent=2)
    print(f"KAHLUS_MESA=final: single final look for {script}, logged to {entry}. Commit it.", file=sys.stderr)


def gate_mesa_ids(ids) -> list[str]:
    """MESA holdout seal on subject IDs. Every MESA reader (EDF, XML or feature cache) goes through this.
    dev (default): only the 25 already-exposed nights, so no run can widen exposure.
    final: only never-seen nights, after the frozen plan is committed, once per script.
    See docs/results/mesa_exposure.md."""
    exposed = mesa_exposed_ids()
    if mesa_mode() == "dev":
        return [i for i in ids if i in exposed]
    _claim_final_look()
    return [i for i in ids if i not in exposed]


def _mesa_gate(recs: list[NsrrRecording]) -> list[NsrrRecording]:
    keep = set(gate_mesa_ids([r.subject_id for r in recs]))
    return [r for r in recs if r.subject_id in keep]


def discover_nsrr_recordings(root: str | Path, *, dataset: str) -> list[NsrrRecording]:
    recs = _discover(root, dataset)
    is_mesa = dataset.lower().startswith("mesa") or any(r.edf_path.name.startswith("mesa") for r in recs)
    return _mesa_gate(recs) if is_mesa else recs


def _discover(root: str | Path, dataset: str) -> list[NsrrRecording]:
    root = Path(root)
    if not root.is_dir():
        return []
    recordings: list[NsrrRecording] = []
    for edf in sorted(root.rglob("*.edf")):
        if edf.name.startswith("."):
            continue
        xml_candidates = [
            edf.with_name(edf.stem + "-nsrr.xml"),
            edf.with_name(edf.stem + ".xml"),
            edf.parent / f"{edf.stem}-nsrr.xml",
        ]
        xml_path = next((p for p in xml_candidates if p.is_file()), None)
        if xml_path is None:
            continue
        subject_id = _subject_id_from_stem(edf.stem)
        recordings.append(
            NsrrRecording(
                edf_path=edf,
                xml_path=xml_path,
                subject_id=subject_id,
                dataset=dataset,
            )
        )
    return recordings


def _subject_id_from_stem(stem: str) -> str:
    parts = stem.replace("_", "-").split("-")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return stem


def parse_nsrr_events(xml_path: str | Path) -> list[tuple[str, float, float]]:
    """All scored events as ``(concept, start_sec, duration_sec)``; concept is the text before ``|``."""
    out: list[tuple[str, float, float]] = []
    for ev in ET.parse(xml_path).getroot().iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "").split("|")[0].strip()
        try:
            out.append((concept, float(ev.findtext("Start")), float(ev.findtext("Duration"))))
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class Arousal:
    onset: float
    kind: str  # "respiratory" | "limb" | "spontaneous"
    clean: bool  # spontaneous, in sleep, nothing disturbing in the CLEAN_PRE_S before


# NSRR rarely labels arousal subtype, so it is inferred from timing.
RESP_GAP_S = 5.0  # ponytail: fixed window, sensitivity sweep (3/5/10 s) is the upgrade path
LIMB_GAP_S = 0.5
CLEAN_PRE_S = 60.0
SLEEP_STAGES = ("Stage 1 sleep", "Stage 2 sleep", "Stage 3 sleep", "Stage 4 sleep", "REM sleep")


def sleep_stage_epochs(events: list[tuple[str, float, float]]) -> list[tuple[float, float, str]]:
    """Sorted ``(start, end, stage)`` for scored sleep stages and wake."""
    return sorted((s, s + d, c) for c, s, d in events if c in SLEEP_STAGES or c == "Wake")


def stage_at(stages: list[tuple[float, float, str]], t: float) -> str | None:
    for s, e, c in stages:
        if s <= t < e:
            return c
    return None


def event_kind(concept: str) -> str | None:
    """Coarse category of an NSRR EventConcept: arousal, respiratory, limb, or None."""
    c = concept.lower()
    if "arousal" in c:
        return "arousal"
    if "apnea" in c or "hypopnea" in c:
        return "respiratory"
    if "limb" in c or "leg movement" in c:
        return "limb"
    return None


def classify_arousals(events: list[tuple[str, float, float]]) -> list[Arousal]:
    kinds = [(event_kind(c), s, d) for c, s, d in events]
    onsets = sorted(s for k, s, _ in kinds if k == "arousal")
    resp = [(s, s + d) for k, s, d in kinds if k == "respiratory"]
    limb = [(s, s + d) for k, s, d in kinds if k == "limb"]
    stages = sleep_stage_epochs(events)
    disturb = onsets + [s for s, _ in resp] + [s for s, _ in limb] + [s for s, _, c in stages if c == "Wake"]

    out: list[Arousal] = []
    for t in onsets:
        if any(s <= t <= e + RESP_GAP_S for s, e in resp):
            out.append(Arousal(t, "respiratory", False))
        elif any(s - LIMB_GAP_S <= t <= e + LIMB_GAP_S for s, e in limb):
            out.append(Arousal(t, "limb", False))
        else:
            quiet = not any(t - CLEAN_PRE_S <= x < t for x in disturb)
            asleep = stage_at(stages, t) in SLEEP_STAGES and stage_at(stages, t - CLEAN_PRE_S) in SLEEP_STAGES
            out.append(Arousal(t, "spontaneous", quiet and asleep))
    return out


def parse_nsrr_arousal_events(xml_path: str | Path) -> list[tuple[float, float]]:
    """Return list of (start_sec, duration_sec) for arousal events."""
    return [(s, d) for c, s, d in parse_nsrr_events(xml_path) if AROUSAL_CONCEPT_RE.search(c)]


def epoch_arousal_mask(
    events: list[tuple[float, float]],
    *,
    n_epochs: int,
    epoch_seconds: float,
) -> np.ndarray:
    """Binary mask: 1 if any arousal overlaps epoch."""
    mask = np.zeros(n_epochs, dtype=np.int64)
    for start, duration in events:
        end = start + duration
        first = max(0, int(start // epoch_seconds))
        last = min(n_epochs - 1, int(end // epoch_seconds))
        if first <= last:
            mask[first : last + 1] = 1
    return mask


def load_nsrr_epoch_matrix(
    edf_path: str | Path,
    *,
    epoch_seconds: float = 30.0,
    channel_groups: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Load NSRR EDF as per-epoch matrices shaped ``(n_epochs, samples, n_channels)``.

    Retains all matched candidate channels per modality group (not just the first).
    """
    import mne

    groups = channel_groups or {
        "eeg": EEG_CHANNEL_CANDIDATES,
        "ecg": ECG_CHANNEL_CANDIDATES,
        "ppg": PPG_CHANNEL_CANDIDATES,
        "eog": EOG_CHANNEL_CANDIDATES,
        "emg": EMG_CHANNEL_CANDIDATES,
        "resp": RESP_CHANNEL_CANDIDATES,
    }
    # Pick channels from the header first, then read only those. NSRR EDFs mix sampling
    # rates (MESA: EEG 256 Hz, resp 32 Hz), so read without preloading everything.
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    picked: dict[str, list[str]] = {}
    for group, candidates in groups.items():
        matched = [label for label in candidates if label in raw.ch_names]
        if matched:
            picked[group] = matched

    if "eeg" not in picked:
        eeg_fallback = next((c for c in raw.ch_names if "EEG" in c.upper()), None)
        if eeg_fallback is None:
            raise ValueError(f"no EEG channel in {edf_path}")
        picked["eeg"] = [eeg_fallback]

    wanted = sorted({label for labels in picked.values() for label in labels})
    raw.pick(wanted)
    raw.load_data(verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = max(1, int(round(epoch_seconds * sfreq)))
    n_epochs = int(raw.n_times // samples_per_epoch)
    if n_epochs < 4:
        raise ValueError(f"too few epochs in {edf_path}")

    data = raw.get_data().astype(np.float32)
    ch_index = {name: idx for idx, name in enumerate(raw.ch_names)}
    n_keep = n_epochs * samples_per_epoch
    epochs: dict[str, np.ndarray] = {}
    for group, labels in picked.items():
        # (n_epochs, samples, n_channels_in_group): one reshape per group, no per-epoch loop.
        rows = data[[ch_index[label] for label in labels], :n_keep]
        epochs[group] = np.ascontiguousarray(
            rows.reshape(len(labels), n_epochs, samples_per_epoch).transpose(1, 2, 0)
        )

    return {
        "epoch_seconds": epoch_seconds,
        "sfreq": sfreq,
        "n_epochs": n_epochs,
        "channels": picked,
        "epochs": epochs,
    }
