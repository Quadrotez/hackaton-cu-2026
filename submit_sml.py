#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

BASE = "https://stackmorelayers.ai/api/platform/v1"
COMP = "astramis"
PHASE = 819
COOKIES = "/tmp/opencode/astramis-final/cookies.json"


def session():
    s = requests.Session()
    s.cookies.update(json.load(open(COOKIES)))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0",
        "Accept": "application/json, text/html, */*",
        "Referer": f"https://stackmorelayers.ai/competitions/{COMP}/submissions",
        "Origin": "https://stackmorelayers.ai",
        "Content-Type": "application/json",
    })
    return s


def fresh(s):
    s.headers.update({
        "x-request-id": uuid.uuid4().hex,
        "x-sml-page-id": uuid.uuid4().hex,
        "x-sml-tab-id": uuid.uuid4().hex,
    })


def submit(path, desc=""):
    path = Path(path).resolve()
    s = session()
    fresh(s)
    init = s.post(f"{BASE}/competitions/{COMP}/submissions/upload-init", json={
        "phase_id": PHASE,
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "submission_intake_type": "csv",
        "description": desc or path.stem,
    }, timeout=30)
    if init.status_code != 200:
        raise RuntimeError(f"init {init.status_code}: {init.text[:400]}")
    sid = init.json()["upload_session_id"]

    fresh(s)
    h = dict(s.headers)
    h["Content-Type"] = "application/octet-stream"
    h["x-sml-upload-session"] = sid
    with open(path, "rb") as f:
        r = requests.put(f"{BASE}/competitions/{COMP}/submissions/upload-file",
                         data=f, cookies=s.cookies, headers=h, timeout=60)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"upload {r.status_code}: {r.text[:400]}")

    fresh(s)
    comp = s.post(f"{BASE}/competitions/{COMP}/submissions/upload-complete", json={
        "phase_id": PHASE,
        "upload_session_id": sid,
        "file_size": path.stat().st_size,
        "submission_intake_type": "csv",
        "description": desc or path.stem,
    }, timeout=30)
    if comp.status_code != 200:
        raise RuntimeError(f"complete {comp.status_code}: {comp.text[:400]}")
    return comp.json()


def score_of(sub):
    for key in ("safe_metrics", "scores", "metrics"):
        for m in sub.get(key, []) or []:
            if m.get("is_primary") or "score" in m:
                try:
                    return float(m["score"])
                except (TypeError, ValueError):
                    pass
    return None


def wait_score(sub_id, timeout=180):
    s = session()
    end = time.time() + timeout
    while time.time() < end:
        fresh(s)
        r = s.get(f"{BASE}/competitions/{COMP}/submissions?phase_id={PHASE}", timeout=30)
        if r.status_code == 200:
            for sub in r.json().get("results", []):
                if sub.get("id") == sub_id:
                    sc = score_of(sub)
                    if sub.get("status") in ("scored", "completed", "finished") and sc is not None:
                        return sc, sub.get("status")
                    if sub.get("status") in ("failed", "error"):
                        return None, sub.get("status")
        time.sleep(3)
    return None, "timeout"


if __name__ == "__main__":
    desc = sys.argv[2] if len(sys.argv) > 2 else ""
    res = submit(sys.argv[1], desc)
    sid = res.get("id") or res.get("submission_id")
    print("submission_id", sid, flush=True)
    sc, status = wait_score(sid)
    print("score", sc, "status", status, flush=True)
