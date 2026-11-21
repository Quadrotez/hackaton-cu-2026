#!/usr/bin/env python3
"""
Automated submission tool for StackMoreLayers (Astramis hackathon).
Uses extracted cookies from local Zen browser session to upload submissions
and retrieve leaderboard scores automatically.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Any

import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import get_submissions

BASE_URL = "https://stackmorelayers.ai/api/platform/v1"
COMPETITION = "astramis"
PHASE_ID = 819


def get_auth_cookies() -> Dict[str, str]:
    db_path = get_submissions.find_zen_cookies_db()
    if not db_path:
        raise RuntimeError("Zen browser cookies.sqlite not found")
    cookies = get_submissions.extract_cookies_from_sqlite(db_path)
    if "sml_platform_session" not in cookies:
        raise RuntimeError("sml_platform_session not found in cookies")
    return cookies


def submit_csv(file_path: Path, description: str = "") -> Dict[str, Any]:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    file_name = file_path.name
    cookies = get_auth_cookies()

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0",
        "Accept": "*/*",
        "Referer": f"https://stackmorelayers.ai/competitions/{COMPETITION}/submissions",
        "Origin": "https://stackmorelayers.ai",
        "Sec-GPC": "1",
    }

    # Step 1: upload-init
    init_url = f"{BASE_URL}/competitions/{COMPETITION}/submissions/upload-init"
    init_payload = {
        "phase_id": PHASE_ID,
        "file_name": file_name,
        "file_size": file_size,
        "submission_intake_type": "csv",
        "description": description or f"Auto-submit {file_name}"
    }

    print(f"[*] Initializing upload for {file_name} ({file_size} bytes)...")
    resp = requests.post(init_url, json=init_payload, cookies=cookies, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"upload-init failed ({resp.status_code}): {resp.text}")

    init_data = resp.json()
    upload_session_id = init_data.get("upload_session_id")
    if not upload_session_id:
        raise RuntimeError(f"No upload_session_id returned: {init_data}")

    # Step 2: upload-file
    file_url = f"{BASE_URL}/competitions/{COMPETITION}/submissions/upload-file"
    file_headers = dict(headers)
    file_headers["content-type"] = "application/octet-stream"
    file_headers["x-sml-upload-session"] = upload_session_id
    file_headers["Content-Length"] = str(file_size)

    print(f"[*] Uploading binary content...")
    with open(file_path, "rb") as f:
        put_resp = requests.put(file_url, data=f, cookies=cookies, headers=file_headers, timeout=60)
    if put_resp.status_code not in (200, 204):
        raise RuntimeError(f"upload-file failed ({put_resp.status_code}): {put_resp.text}")

    # Step 3: upload-complete
    complete_url = f"{BASE_URL}/competitions/{COMPETITION}/submissions/upload-complete"
    complete_payload = {
        "phase_id": PHASE_ID,
        "upload_session_id": upload_session_id,
        "file_size": file_size,
        "submission_intake_type": "csv",
        "description": description or f"Auto-submit {file_name}"
    }

    print(f"[*] Completing upload...")
    comp_resp = requests.post(complete_url, json=complete_payload, cookies=cookies, headers=headers, timeout=30)
    if comp_resp.status_code != 200:
        raise RuntimeError(f"upload-complete failed ({comp_resp.status_code}): {comp_resp.text}")

    comp_data = comp_resp.json()
    submission_id = comp_data.get("id") or comp_data.get("submission_id") or comp_data.get("submission", {}).get("id")
    print(f"[+] Successfully submitted! Submission ID: {submission_id}")

    return {
        "submission_id": submission_id,
        "upload_session_id": upload_session_id,
        "file_name": file_name,
        "init_response": init_data,
        "complete_response": comp_data
    }


def wait_for_score(submission_id: Optional[int] = None, file_name: Optional[str] = None, max_wait_sec: int = 180, poll_interval: int = 10) -> Optional[float]:
    cookies = get_auth_cookies()
    start_time = time.time()
    print(f"[*] Polling for score (target id: {submission_id}, filename: {file_name})...")

    while time.time() - start_time < max_wait_sec:
        time.sleep(poll_interval)
        try:
            subs = get_submissions.fetch_submissions(COMPETITION, cookies)
            for s in subs:
                s_id = s.get("id")
                s_file = s.get("filename")
                s_status = s.get("status")
                if (submission_id and s_id == submission_id) or (file_name and s_file == file_name):
                    score = get_submissions.get_primary_score(s)
                    print(f"    Status: {s_status}, Current score: {score}")
                    if s_status == "scored" and score is not None:
                        print(f"\n[★] FINAL SCORE for {s_file} (ID: {s_id}): {score:.6f}")
                        return score
                    elif s_status in ("failed", "error"):
                        print(f"[!] Submission {s_id} failed on platform evaluation.")
                        return None
        except Exception as e:
            print(f"    [!] Polling warning: {e}")

    print("[!] Timeout waiting for score.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StackMoreLayers submitter")
    parser.add_argument("file", help="Path to submission CSV")
    parser.add_argument("-d", "--desc", default="", help="Description")
    parser.add_argument("--wait", action="store_true", help="Wait for score")
    args = parser.parse_args()

    res = submit_csv(Path(args.file), args.desc)
    if args.wait:
        wait_for_score(res["submission_id"], res["file_name"])
