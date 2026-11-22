#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import submit_sml as S

BASE = Path("submissions/submission_v46_physical_take_and_purge_champion.csv")
LOG = Path("experiments.jsonl")


def modify(path, spec):
    rows = list(csv.reader(open(path, newline="", encoding="utf-8")))
    header, body = rows[0], rows[1:]
    frames = set(spec.get("frames", []))
    all_objs = bool(spec.get("all_objects", False))
    field = spec.get("field")
    value = spec.get("value")
    changed = 0
    for row in body:
        if row[0] not in frames or row[1] == "none":
            continue
        objs = [o.split() for o in row[1].split(";")]
        targets = objs if all_objs else objs[:1]
        for o in targets:
            if field == "take":
                o[9] = str(int(value))
            elif field == "volume":
                o[7] = str(value)
            elif field == "mass":
                o[8] = str(value)
            elif field == "material":
                o[6] = value
            elif field == "form":
                o[5] = value
            elif field == "conf":
                o[4] = str(value)
        changed += 1
        row[1] = ";".join(" ".join(o) for o in objs)
    out = Path(f"candidates/{spec['name']}.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    return out, changed


def run(spec):
    out, changed = modify(BASE, spec)
    res = S.submit(out, spec["name"])
    sid = res.get("id") or res.get("submission_id")
    score, status = S.wait_score(sid)
    entry = {**spec, "score": score, "status": status, "id": sid, "file": str(out), "changed": changed}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)
    return entry


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        spec = json.loads(arg)
        run(spec)
