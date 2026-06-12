"""Recompute the MATCHED (task,subset,seed,object) distractor + task pass deltas
between a branch smoke jsonl and the main baseline jsonl.

Usage: _diag_matched_delta.py <branch.jsonl> <main.jsonl>
"""
from __future__ import annotations

import sys


def load(p: str) -> dict:
    import json

    out = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "passed" not in r:
            continue
        key = "|".join([r["task"], ",".join(r["subset"]), str(r["seed"]), r["name"]])
        out[key] = r
    return out


def main() -> int:
    branch, main = load(sys.argv[1]), load(sys.argv[2])
    keys = set(branch) & set(main)
    for label, pred in (
        ("DISTRACTOR", lambda k: k.split("|")[3].startswith("distractor")),
        ("TASK", lambda k: not k.split("|")[3].startswith("distractor")),
    ):
        ks = [k for k in keys if pred(k)]
        n = len(ks)
        mp = sum(main[k]["passed"] for k in ks)
        bp = sum(branch[k]["passed"] for k in ks)
        if not n:
            print(f"{label}: no matched cells")
            continue
        print(
            f"{label:11} matched n={n:4}  MAIN {100*mp/n:5.1f}%  BRANCH {100*bp/n:5.1f}%  "
            f"delta {100*(bp-mp)/n:+.1f}pp"
        )
        # Per-task distractor breakdown for the regression cells.
        if label == "DISTRACTOR":
            from collections import defaultdict

            agg = defaultdict(lambda: [0, 0, 0])  # task -> [main_pass, branch_pass, n]
            for k in ks:
                t = k.split("|")[0].split("/")[-1]
                agg[t][0] += main[k]["passed"]
                agg[t][1] += branch[k]["passed"]
                agg[t][2] += 1
            for t, (mpx, bpx, nx) in sorted(agg.items()):
                flag = "  <-- regressed" if bpx < mpx else ""
                print(f"    {t:52} main {mpx}/{nx} branch {bpx}/{nx}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
