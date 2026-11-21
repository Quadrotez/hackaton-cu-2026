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
    ap.add_argument("--file", help="local path to broadcast as a document")
    ap.add_argument("text", nargs="?")
    args = ap.parse_args()
    users = json.loads(USERS.read_text()) if USERS.exists() else {}
    ids = [args.to] if args.to else [int(k) for k in users]
    for cid in ids:
        if args.file:
            caption = f"[{args.sender} → всем]\n{args.text or Path(args.file).name}"
            with open(args.file, "rb") as fh:
                r = requests.post(f"{API}/sendDocument",
                                  data={"chat_id": cid, "caption": caption},
                                  files={"document": fh}, timeout=120)
        else:
            body = f"[{args.sender} → всем]\n{args.text or ''}"
            r = requests.post(f"{API}/sendMessage", json={"chat_id": cid, "text": body}, timeout=20)
        print(cid, r.json().get("ok"), r.json().get("description", ""))


if __name__ == "__main__":
    main()
