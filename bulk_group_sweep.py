#!/usr/bin/env python3
import json
import time
from pathlib import Path
import requests

BASE = "https://stackmorelayers.ai/api/platform/v1"
COMP = "astramis"
PHASE = 819
COOKIE = dict(
    p.split("=", 1)
    for p in Path("/tmp/opencode/sigma_cookie.txt").read_text().strip().split("; ")
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0",
    "Accept": "*/*",
    "Referer": f"https://stackmorelayers.ai/competitions/{COMP}/submissions",
    "Origin": "https://stackmorelayers.ai",
    "Sec-GPC": "1",
}

files = sorted(Path("submissions").glob("submission_v60_*_bulk_champion.csv"))
ids = {}
for f in files:
    size = f.stat().st_size
    desc = f.stem
    r = requests.post(
        f"{BASE}/competitions/{COMP}/submissions/upload-init",
        json={
            "phase_id": PHASE,
            "file_name": f.name,
            "file_size": size,
            "submission_intake_type": "csv",
            "description": desc,
        },
        cookies=COOKIE,
        headers={**HEADERS, "Content-Type": "application/json"},
        timeout=30,
    )
    sid = r.json()["upload_session_id"]
    up = requests.put(
        f"{BASE}/competitions/{COMP}/submissions/upload-file",
        data=f.read_bytes(),
        cookies=COOKIE,
        headers={
            **HEADERS,
            "Content-Type": "application/octet-stream",
            "x-sml-upload-session": sid,
            "Content-Length": str(size),
        },
        timeout=60,
    )
    done = requests.post(
        f"{BASE}/competitions/{COMP}/submissions/upload-complete",
        json={
            "phase_id": PHASE,
            "upload_session_id": sid,
            "file_size": size,
            "submission_intake_type": "csv",
            "description": desc,
        },
        cookies=COOKIE,
        headers={**HEADERS, "Content-Type": "application/json"},
        timeout=30,
    )
    sub_id = done.json().get("id")
    ids[sub_id] = f.name
    print("SUBMITTED", sub_id, f.name, "upload", up.status_code, flush=True)

for _ in range(20):
    time.sleep(6)
    r = requests.get(
        f"{BASE}/competitions/{COMP}/submissions",
        cookies=COOKIE,
        headers={**HEADERS, "Accept": "application/json"},
        timeout=30,
    )
    by_id = {s["id"]: s for s in r.json().get("results", [])}
    pending = [i for i in ids if by_id.get(i, {}).get("status") != "scored"]
    print("pending", len(pending), flush=True)
    if not pending:
        break

for sub_id, name in ids.items():
    s = by_id.get(sub_id, {})
    scores = [x.get("score") for x in s.get("scores", []) if x.get("is_primary")]
    print("RESULT", sub_id, name, scores, flush=True)
