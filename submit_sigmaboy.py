#!/usr/bin/env python3
"""
Submit tool for 'sigma boy' account on StackMoreLayers (Astramis hackathon).
Uses the session cookie provided by user for account usr_a2e86286ac4c4adcafa8667a4ad80a43.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional

import requests

BASE_URL = "https://stackmorelayers.ai/api/platform/v1"
COMPETITION = "astramis"
PHASE_ID = 819

COOKIE_STR = 'sml_tz=Europe%2FMoscow; sml_platform_session=eyJjcmVhdGVkX2F0IjoxNzk1MzM0NTkyLCJleHBpcmVzX2F0IjoxNzk1OTM5MzkyLCJzZXNzaW9uIjp7ImFjY291bnRfc2VjdXJpdHlfaWRlbnRpdHlfc3RhdGUiOnsiZW1haWxfY29sbGlzaW9uIjpmYWxzZSwicmVhc29uX2NvZGUiOm51bGwsInN0YXRlIjoiYXZhaWxhYmxlIn0sImFjY291bnRfc2VjdXJpdHlfcGxhdGZvcm1fdXNlcl9pZCI6InVzcl9hMmU4NjI4NmFjNGM0YWRjYWZhODY2N2E0YWQ4MGE0MyIsImF1dGhfbW9kZSI6InppdGFkZWwiLCJhdXRoZW50aWNhdGVkIjp0cnVlLCJhdXRob3JpdHlfZXBvY2giOiJmYmQzYWRmYzhlY2FkMGUwM2M5MTdhYjE0MDI4MjU1ZSIsImF2YWlsYWJsZV9hY3Rpb25zIjpbImJyb3dzZV9jb21wZXRpdGlvbnMiLCJjcmVhdGVfY29sbGVjdGlvbiIsImNyZWF0ZV9jb21wZXRpdGlvbiIsImNyZWF0ZV9wcmFjdGljZSIsInVzZV9jb2xsZWN0aW9ucyIsInVzZV9kYXRhc2V0cyIsInVzZV9tb2RlbHMiLCJ1c2VfcHJhY3RpY2VzIiwidXNlX3NldHRpbmdzIiwidXNlX3NvbHV0aW9ucyIsInVzZV9zdWJtaXNzaW9ucyIsInVzZV93b3Jrc3BhY2VzIiwidmlld19kYXNoYm9hcmQiLCJ2aWV3X3N0dWRpbyJdLCJjYXBhYmlsaXRpZXMiOnt9LCJjcmVhdGVkX2F0IjoxNzk1MzM0NTkyLCJleHBpcmVzX2F0IjoxNzk1OTM5MzkyLCJnbG9iYWxfcm9sZXMiOlsiYXV0aGVudGljYXRlZCIsInBhcnRpY2lwYW50Il0sImlkZW50aXR5Ijp7ImVtYWlsX3ZlcmlmaWNhdGlvbl9zdGF0ZSI6InZlcmlmaWVkIiwiaWRlbnRpdHlfc291cmNlIjoieml0YWRlbCIsInByb3ZpZGVyX2tleSI6InppdGFkZWwiLCJwcm92aWRlcl9zdWJqZWN0X3JlZiI6InppdGFkZWw6ZDlmYTEyYjE2MGZhOGE2NmZjZTA5MTQ4In0sImlkZW50aXR5X3ByaXZhdGUiOnsiYXBwbGljYXRpb25fYXV0aF9iaW5kaW5nIjoiZ0FBQUFBQnJBcUhBQlZTOTVVUC04a1QtWUp5Tmc1ampVanRjcHlJOFdTRHdvSXlNNThIdHRtTHNIc3M4amNxYjl0ZkQyTTZ4N2tGd2Nab0hLNXB1UXh3YkVqN3NxWVVKQlNrU0tTM0Y2N2xfTXVCNG50czNrQy1wa3gwVXlZUjhSaU82aU4tZDdNSXE4b200RzZ4cC11R0VOdEk4SG9RUWdhNF9XN19XTHNtbkhkRHNjZUR5cDBpT0t4enJzN3V2QlJQa29QQkQ5Ui1FYnZzNEZWUUIzYlBEcFYtanRTOGNxM2JQdXlESUI3aXR6WkpDcXJLUXZxdTBwbXR1OTJEdXcxSXhsQlozVEZnRzNsNGNELWtfeFF6MjFsdTdIRUtkalZCVjV0SnMyejR0THRkMnRtTFRab1JQemo4bHR6dW9zN3pxQld4NkJFRWhzNEhLaWZQTm1MRW9jSW0tQ2IzMzBRPT0iLCJ6aXRhZGVsX3VzZXJfY2lwaGVydGV4dCI6ImdBQUFBQUJyQXFIQTNzMy1mZWdqd1NmZFk5OVMxMUJYdXluRWZVZUREbjFJbGhuaENWOTYycXhZOHJwdk5xbXo2V0RITzNOWXdtXzdfYmZlZTdaT0t2clNKRG96UVpBLUdKMUtJVk01WDBnYVhReHZtZERIRURJPSJ9LCJpZGVudGl0eV9zb3VyY2UiOiJ6aXRhZGVsIiwicm9sZXMiOlsiYXV0aGVudGljYXRlZCIsInBhcnRpY2lwYW50Il0sInNpZCI6Ijc2MjQzODY2LTI0MDgtNGFkMS1iMWZkLTM3ZDM2NWY4NThhYiIsInVzZXIiOnsiZGlzcGxheV9uYW1lIjoic2lnbWEgYm95IiwiZW1haWwiOiJ6dWJrb3ZuczEwQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWNhdGlvbl9zdGF0ZSI6InZlcmlmaWVkIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZlZGVyYXRlZF9lbWFpbCI6IiIsImZlZGVyYXRlZF9lbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaWQiOjIxNDksImlzX3N0YWZmIjpmYWxzZSwiaXNfc3VwZXJ1c2VyIjpmYWxzZSwidXNlcm5hbWUiOiJ1c3JfYTJlODYyODZhYzRjNGFkY2FmYTg2NjdhNGFkODBhNDMifX0sInZlcnNpb24iOjF9.cb443ebbe87aee0e128fc92eaf2c7568883613d0037da59463804d8859c64c80'


def parse_cookie_str(s: str) -> Dict[str, str]:
    cookies = {}
    for part in s.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k] = v
    return cookies


def submit_csv(file_path: Path, description: str = "") -> Dict[str, Any]:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    file_name = file_path.name
    cookies = parse_cookie_str(COOKIE_STR)

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
    resp = None
    for attempt in range(1, 4):
        try:
            resp = requests.post(init_url, json=init_payload, cookies=cookies, headers=headers, timeout=30)
            if resp.status_code == 200:
                break
            print(f"[-] Init attempt {attempt} returned {resp.status_code}: {resp.text}, retrying...")
        except Exception as e:
            print(f"[-] Init attempt {attempt} network exception: {e}, retrying in 3s...")
            time.sleep(3)

    if resp is None or resp.status_code != 200:
        raise RuntimeError(f"upload-init failed after 3 attempts")

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
    put_resp = None
    for attempt in range(1, 4):
        try:
            with open(file_path, "rb") as f:
                put_resp = requests.put(file_url, data=f, cookies=cookies, headers=file_headers, timeout=60)
            if put_resp.status_code in (200, 204):
                break
            print(f"[-] Upload attempt {attempt} returned {put_resp.status_code}: {put_resp.text}, retrying...")
        except Exception as e:
            print(f"[-] Upload attempt {attempt} network exception: {e}, retrying in 3s...")
            time.sleep(3)

    if put_resp is None or put_resp.status_code not in (200, 204):
        raise RuntimeError(f"upload-file failed after 3 attempts")

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
    comp_resp = None
    for attempt in range(1, 4):
        try:
            comp_resp = requests.post(complete_url, json=complete_payload, cookies=cookies, headers=headers, timeout=30)
            if comp_resp.status_code in (200, 201):
                break
            print(f"[-] Complete attempt {attempt} returned {comp_resp.status_code}: {comp_resp.text}, retrying...")
        except Exception as e:
            print(f"[-] Complete attempt {attempt} network exception: {e}, retrying in 3s...")
            time.sleep(3)

    if comp_resp is None or comp_resp.status_code not in (200, 201):
        raise RuntimeError(f"upload-complete failed after 3 attempts")

    sub_result = comp_resp.json()
    sub_id = sub_result.get("id") or sub_result.get("submission_id") or sub_result.get("pk")
    print(f"[+] Successfully submitted! Submission ID: {sub_id}")
    return sub_result


def poll_submission(sub_id: int, max_wait_sec: int = 180, interval_sec: int = 5) -> Optional[float]:
    cookies = parse_cookie_str(COOKIE_STR)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0",
        "Accept": "*/*",
    }
    subs_url = f"{BASE_URL}/competitions/{COMPETITION}/submissions"
    start_time = time.time()

    print(f"[*] Waiting for scoring of submission {sub_id}...")
    while time.time() - start_time < max_wait_sec:
        try:
            resp = requests.get(subs_url, cookies=cookies, headers=headers, timeout=10)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                for item in results:
                    if item.get("id") == sub_id:
                        status = item.get("status")
                        print(f"    Status: {status} ({int(time.time() - start_time)}s elapsed)")
                        if status == "scored":
                            scores = item.get("scores", [])
                            for sc in scores:
                                if sc.get("is_primary") or sc.get("column_key") == "astramis_score_public":
                                    score_val = float(sc.get("score"))
                                    print(f"[🏆] RESULT FOR SUBMISSION {sub_id}: {score_val:.6f}")
                                    return score_val
                            safe_metrics = item.get("safe_metrics", [])
                            for sm in safe_metrics:
                                if sm.get("split") == "public":
                                    score_val = float(sm.get("score"))
                                    print(f"[🏆] RESULT FOR SUBMISSION {sub_id}: {score_val:.6f}")
                                    return score_val
                        elif status in ("failed", "error"):
                            print(f"[!] Submission failed: {item.get('safe_error_summary')}")
                            return None
        except Exception as e:
            print(f"    Warning: poll request error: {e}")
        time.sleep(interval_sec)

    print(f"[!] Timeout waiting for submission {sub_id} to score.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit CSV to Astramis for sigma boy account")
    parser.add_argument("file", help="Path to submission CSV")
    parser.add_argument("-d", "--desc", default="", help="Description")
    parser.add_argument("--poll", action="store_true", default=True, help="Poll for score")
    args = parser.parse_args()

    sub_res = submit_csv(Path(args.file), args.desc)
    sub_id = sub_res.get("id") or sub_res.get("submission_id") or sub_res.get("pk")
    if args.poll and sub_id:
        poll_submission(sub_id)
