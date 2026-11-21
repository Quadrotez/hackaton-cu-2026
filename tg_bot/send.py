"""Agent-side sender: echoes a message into the relay bot as if from the agent."""
import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
TOKEN = (ROOT / "token.txt").read_text().strip()
USERS = ROOT / "users.json"
API = f"https://api.telegram.org/bot{TOKEN}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="sender", default="Агент")
    ap.add_argument("--to", type=int, help="only this chat_id")
    ap.add_argument("text")
    args = ap.parse_args()
    users = json.loads(USERS.read_text()) if USERS.exists() else {}
    ids = [args.to] if args.to else [int(k) for k in users]
    body = f"[{args.sender} → всем]\n{args.text}"
    for cid in ids:
        r = requests.post(f"{API}/sendMessage", json={"chat_id": cid, "text": body}, timeout=20)
        print(cid, r.json().get("ok"))


if __name__ == "__main__":
    main()
