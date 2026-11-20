"""Mirror NSRR folders to a local tree: parallel, resumable, MD5-verified. Python standard library only.

    NSRR_TOKEN=... SSL_CERT_FILE=ca-bundle.pem python3 nsrr_mirror.py --root /shared/aayu/nsrr \
        --dataset shhs --folder polysomnography/annotations-events-nsrr/shhs1 --folder polysomnography/edfs/shhs1

Layout matches the nsrr gem: ROOT/<dataset>/<full_path>. Facts this design rests on (measured 2026-11-20):
  * sleepdata.org answers each file request after about 4.7 s, then streams at about 3.9 MB/s per connection.
  * nginx refuses more than about 10 concurrent requests with an instant 429 (no Retry-After). The cap is shared
    by all our clients (same token, or the whole site): with 40 gem streams running on Engram, 3 of 4 requests
    from another server got 429. Run one mirror at a time.
  * files.json lists a whole folder (5,793 entries) with size and MD5 in one call.
So the speed comes from about 8 parallel streams, never more than 9, and every 429 is retried after a backoff.

Exit codes: 0 done (prints NSRR_MIRROR_DONE), 1 incomplete after the pass, 3 token refused, 75 stalled
(prints NSRR_MIRROR_STALL; rerun to resume). The token is never printed: only file names reach the log.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import ssl
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOST = "sleepdata.org"
GEM_TAG = "nsrr-gem-v8-0-0"  # the download path the official gem uses
CHUNK = 1 << 20


class TokenRefused(Exception):
    pass


class Mirror:
    def __init__(self, a):
        self.a, self.root = a, Path(a.root)
        self.token = os.environ.get("NSRR_TOKEN", "").strip()
        if not self.token:
            sys.exit("nsrr_mirror: NSRR_TOKEN is not set")
        self.ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
        self.state = self.root / ".mirror"
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.Lock()
        self.verified_path = self.state / "verified.txt"
        self.verified = set(self.verified_path.read_text().split()) if self.verified_path.exists() else set()
        self.bytes = self.files_done = self.n429 = self.errors = 0
        self.last_byte = time.monotonic()
        self.t0 = time.monotonic()
        self.stop = threading.Event()

    # ---- HTTP ----
    def conn(self) -> http.client.HTTPSConnection:
        return http.client.HTTPSConnection(HOST, context=self.ctx, timeout=self.a.timeout)

    def backoff(self, attempt: int) -> None:
        time.sleep(min(60.0, 2.0 * 2 ** min(attempt, 5)) * (0.5 + random.random()))

    def list_folder(self, folder: str) -> list[dict]:
        q = urllib.parse.urlencode({"auth_token": self.token, "path": folder})
        for attempt in range(30):
            c = self.conn()
            try:
                c.request("GET", f"/api/v1/datasets/{self.a.dataset}/files.json?{q}")
                r = c.getresponse()
                body = r.read()
                if r.status == 200:
                    return [f for f in json.loads(body) if f.get("is_file")]
                if r.status in (401, 403):
                    raise TokenRefused
                print(f"listing {folder}: HTTP {r.status}, retry", flush=True)
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
                print(f"listing {folder}: {type(e).__name__}, retry", flush=True)
            finally:
                c.close()
            self.backoff(attempt)
        sys.exit(f"nsrr_mirror: could not list {folder}")

    # ---- files ----
    def local(self, f: dict) -> Path:
        return self.root / self.a.dataset / f["full_path"]

    def md5(self, p: Path) -> str:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            while b := fh.read(CHUNK):
                h.update(b)
        return h.hexdigest()

    def mark(self, f: dict) -> None:
        with self.lock:
            self.verified.add(f["full_path"])
            with open(self.verified_path, "a") as fh:
                fh.write(f["full_path"] + "\n")

    def is_done(self, f: dict) -> bool:
        """Done = present with the listed size and MD5. A size match not yet hashed is hashed once here."""
        p = self.local(f)
        if not p.exists() or p.stat().st_size != f["file_size"]:
            return False
        if f["full_path"] in self.verified:
            return True
        if self.md5(p) == f["file_checksum_md5"]:
            self.mark(f)
            return True
        p.unlink()  # wrong content with the right size: fetch it again
        return False

    def fetch(self, f: dict) -> None:
        if self.stop.is_set() or self.is_done(f):
            return
        p = self.local(f)
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        part = p.with_name(p.name + ".part")
        path = urllib.parse.quote(f"/datasets/{self.a.dataset}/files/a/{self.token}/m/{GEM_TAG}/{f['full_path']}")
        attempt = 0
        while not self.stop.is_set():
            c = self.conn()
            try:
                c.request("GET", path)
                r = c.getresponse()
                if r.status == 200:
                    h, n = hashlib.md5(), 0
                    with open(part, "wb") as out:
                        while chunk := r.read(CHUNK):
                            out.write(chunk)
                            h.update(chunk)
                            n += len(chunk)
                            with self.lock:
                                self.bytes += len(chunk)
                                self.last_byte = time.monotonic()
                    if n == f["file_size"] and h.hexdigest() == f["file_checksum_md5"]:
                        os.replace(part, p)
                        self.mark(f)
                        with self.lock:
                            self.files_done += 1
                        return
                    print(f"checksum mismatch {f['file_name']}, retry", flush=True)
                elif r.status in (301, 302, 401, 403):
                    raise TokenRefused  # the gem reads a redirect as "token not authorized"
                elif r.status == 429:
                    r.read()
                    with self.lock:
                        self.n429 += 1
                else:
                    r.read()
                    print(f"HTTP {r.status} {f['file_name']}, retry", flush=True)
                    with self.lock:
                        self.errors += 1
            except (OSError, http.client.HTTPException) as e:
                print(f"{type(e).__name__} {f['file_name']}, retry", flush=True)
                with self.lock:
                    self.errors += 1
            finally:
                c.close()
                if part.exists():
                    part.unlink()
            self.backoff(attempt)
            attempt += 1

    # ---- progress and stall alarm ----
    def report(self, total_files: int, total_bytes: int, have_files: int, have_bytes: int) -> str:
        with self.lock:
            done_f, done_b, n429, errs, idle = have_files + self.files_done, have_bytes + self.bytes, self.n429, self.errors, time.monotonic() - self.last_byte
        el = max(1.0, time.monotonic() - self.t0)
        rate = self.bytes / el
        eta = (total_bytes - done_b) / rate / 3600 if rate > 0 else float("inf")
        line = (f"PROGRESS {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} files={done_f}/{total_files} "
                f"GB={done_b / 1e9:.2f}/{total_bytes / 1e9:.2f} MBps={rate / 1e6:.1f} eta_h={eta:.2f} "
                f"http429={n429} errors={errs} idle_s={idle:.0f}")
        with open(self.state / "progress.log", "a") as fh:
            fh.write(line + "\n")
        return line

    def run(self) -> int:
        os.umask(0o077)
        files: list[dict] = []
        for folder in self.a.folder:
            files += self.list_folder(folder)
        if self.a.include:  # MESA: only the 25 development EDFs may be fetched (the seal)
            pat = re.compile(self.a.include)
            files = [f for f in files if not f["file_name"].endswith(".edf") or pat.search(f["file_name"])]
        if self.a.limit:
            files = files[: self.a.limit]
        files.sort(key=self.priority)
        total_b = sum(f["file_size"] for f in files)
        print(f"listed {len(files)} files, {total_b / 1e9:.2f} GB; checking what is already here", flush=True)
        with ThreadPoolExecutor(max(2, len(os.sched_getaffinity(0)))) as ex:  # hash existing files on the CPUs Slurm gave us
            done_mask = list(ex.map(self.is_done, files))
        todo = [f for f, d in zip(files, done_mask) if not d]
        have_f = len(files) - len(todo)
        have_b = sum(f["file_size"] for f, d in zip(files, done_mask) if d)
        print(f"already complete {have_f}, to fetch {len(todo)}, streams {self.a.conc}", flush=True)
        print(self.report(len(files), total_b, have_f, have_b), flush=True)

        def watch():
            while not self.stop.wait(self.a.progress_every):
                print(self.report(len(files), total_b, have_f, have_b), flush=True)
                if time.monotonic() - self.last_byte > self.a.stall_minutes * 60:
                    print(f"NSRR_MIRROR_STALL no bytes for {self.a.stall_minutes} min", flush=True)
                    os._exit(75)

        self.last_byte = time.monotonic()
        threading.Thread(target=watch, daemon=True).start()
        try:
            with ThreadPoolExecutor(self.a.conc) as ex:
                list(ex.map(self.fetch, todo))
        except TokenRefused:
            self.stop.set()
            print("NSRR_MIRROR_TOKEN_REFUSED: renew the token or the data request", flush=True)
            return 3
        self.stop.set()
        missing = [f for f in files if not self.is_done(f)]  # final pass: every file, size and MD5
        print(self.report(len(files), total_b, have_f, have_b), flush=True)
        if missing:
            print(f"NSRR_MIRROR_INCOMPLETE missing={len(missing)} first={missing[0]['file_name']}", flush=True)
            return 1
        print(f"NSRR_MIRROR_DONE files={len(files)} GB={total_b / 1e9:.2f} root={self.root} md5=all", flush=True)
        return 0

    def priority(self, f: dict) -> tuple:
        """Held-out subjects first (kahlus.shards.is_heldout: sha256('shhs:<id>') % 10 == 0), then small files."""
        digits = [p for p in f["file_name"].replace("_", "-").split(".")[0].split("-") if p.isdigit()]
        sid = digits[-1] if digits else f["file_name"]
        held = int(hashlib.sha256(f"{self.a.dataset}:{sid}".encode()).hexdigest(), 16) % 10 == 0
        return (not held, f["file_size"] > 5_000_000, f["full_path"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="shhs")
    ap.add_argument("--folder", action="append", required=True)
    ap.add_argument("--conc", type=int, default=8, help="parallel streams; nginx refuses more than about 10 in total")
    ap.add_argument("--timeout", type=float, default=120.0, help="socket timeout in seconds")
    ap.add_argument("--progress-every", type=float, default=120.0)
    ap.add_argument("--stall-minutes", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="first N listed files only (tests)")
    ap.add_argument("--include", default="", help="regex an EDF file name must match (other files are always kept)")
    a = ap.parse_args()
    if a.conc > 9:
        sys.exit("nsrr_mirror: --conc above 9 only buys 429s (nginx allows about 10 requests at once in total)")
    sys.exit(Mirror(a).run())


if __name__ == "__main__":
    main()
