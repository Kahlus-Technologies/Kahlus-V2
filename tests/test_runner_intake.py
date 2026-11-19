"""The launcher's intake of RESULT lines from a remote log: a bad line is dropped and its job counts as missing."""
import json

from kahlus import runner

PROV = {"commit": "abc"}


def out(seed):
    cfg = {"fn": "x:y", "seed": seed}
    return {"job": runner.job_hash(cfg, PROV), "config": cfg, "provenance": PROV, "result": {"loss": 0.5}}


def test_good_lines_pass_and_bad_ones_are_dropped():
    a, b, c, d = out(0), out(1), out(2), out(3)
    lines = [runner.result_line(x) for x in (a, b, c)]
    cut = lines[1][:-20]  # a cut line
    forged = lines[2].replace('"loss": 0.5', '"loss": 0.1')  # payload changed, digest stale
    mixed = "(worker) " + lines[0][:40] + "(worker) " + lines[0][40:]  # two writers interleaved
    log = "\n".join(["(worker) " + lines[0], cut, forged, mixed, "(worker) " + runner.result_line(d)])
    got = runner.intake(log, {a["job"], b["job"], c["job"]})  # d is not in the job list
    assert list(got) == [a["job"]] and json.loads(got[a["job"]]) == a


def test_a_payload_whose_hash_is_not_its_own_is_dropped():
    a, b = out(0), out(1)
    swapped = {**a, "job": b["job"]}  # claims another job's hash, with a valid digest
    assert runner.intake(runner.result_line(swapped), {b["job"]}) == {}
