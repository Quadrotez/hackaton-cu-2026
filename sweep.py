#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

from experiment import BASE, modify
import submit_sml as S

rows = list(csv.reader(open(BASE, newline="", encoding="utf-8")))[1:]
items = []
for r in rows:
    if r[1] == "none":
        continue
    o = r[1].split(";")[0].split()
    items.append((int(r[0][2:]), r[0], (o[5], o[6], o[9])))

groups = []
cur = []
for n, i, key in items:
    if cur and (n > cur[-1][0] + 2 or cur[-1][2] != key):
        groups.append(cur)
        cur = []
    cur.append((n, i, key))
if cur:
    groups.append(cur)


def sweep(field, values, tag):
    log = Path(f"sweep_{tag}.jsonl")
    seen = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            seen.add((e["field"], e["value"], tuple(e["frames"])))
    for g in groups:
        frames = [x[1] for x in g]
        for value in values:
            key = (field, value, tuple(frames))
            if key in seen:
                continue
            spec = {"name": f"{tag}_{frames[0]}_{field}_{value}".replace(".", "_"),
                    "frames": frames, "field": field, "value": value}
            out, changed = modify(BASE, spec)
            res = S.submit(out, spec["name"])
            sid = res.get("id") or res.get("submission_id")
            score, status = S.wait_score(sid)
            entry = {**spec, "score": score, "status": status, "id": sid, "changed": changed}
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(json.dumps(entry, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    field, tag = sys.argv[1], sys.argv[2]
    values = []
    for v in sys.argv[3:]:
        try:
            values.append(int(v))
        except ValueError:
            values.append(v)
    sweep(field, values, tag)
