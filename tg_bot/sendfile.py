"""Send a local file to one or all Telegram relay participants."""
import argparse, json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
TOKEN = (ROOT / "token.txt").read_text().strip()
API = f"https://api.telegram.org/bot{TOKEN}"

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--to", type=int)
ap.add_argument("--caption", default="")
args = ap.parse_args()
users = json.loads((ROOT / "users.json").read_text()) if (ROOT / "users.json").exists() else {}
ids = [args.to] if args.to else [int(k) for k in users]
for cid in ids:
    with open(args.path, "rb") as fh:
        r = requests.post(f"{API}/sendDocument", data={"chat_id": cid, "caption": args.caption},
                          files={"document": fh}, timeout=60)
    print(cid, r.json().get("ok"), r.json().get("description", ""))
