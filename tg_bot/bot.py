"""Relay bot: every message goes to all registered participants with sender label.

Run: .venv/bin/python tg_bot/bot.py
Agents can post via tg_bot/send.py --from "Limo" "text".
"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
TOKEN = (ROOT / "token.txt").read_text().strip()
USERS = ROOT / "users.json"
LOG = ROOT / "messages.jsonl"
API = f"https://api.telegram.org/bot{TOKEN}"
OFFSET = ROOT / "offset.txt"


def api(method, **params):
    try:
        r = requests.post(f"{API}/{method}", json=params, timeout=20)
        return r.json()
    except Exception as e:
        print("api error", method, e, flush=True)
        return {"ok": False}


def load_users():
    if USERS.exists():
        return json.loads(USERS.read_text())
    return {}


def save_users(users):
    USERS.write_text(json.dumps(users, ensure_ascii=False, indent=2))


def broadcast(text, sender, exclude=None):
    users = load_users()
    sent = []
    for cid, info in users.items():
        if exclude is not None and int(cid) == int(exclude):
            continue
        label = info.get("name") or info.get("username") or str(cid)
        body = f"[{sender} → всем]\n{text}"
        r = api("sendMessage", chat_id=int(cid), text=body)
        sent.append((cid, r.get("ok")))
    return sent


def main():
    offset = int(OFFSET.read_text()) if OFFSET.exists() else 0
    print("bot started", flush=True)
    while True:
        data = api("getUpdates", offset=offset, timeout=30, allowed_updates=["message"])
        if not data.get("ok"):
            time.sleep(5); continue
        for upd in data.get("result", []):
            offset = max(offset, upd["update_id"] + 1)
            msg = upd.get("message", {})
            chat = msg.get("chat", {})
            cid = chat.get("id")
            fromu = msg.get("from", {})
            name = fromu.get("first_name") or fromu.get("username") or str(fromu.get("id"))
            text = msg.get("text", "")
            doc = msg.get("document")
            if not cid: continue
            users = load_users()
            if text.startswith("/start"):
                users[str(cid)] = {"name": name, "username": fromu.get("username", "")}
                save_users(users)
                api("sendMessage", chat_id=cid, text="Ты в сети. Сообщения будут рассылаться всем участникам с указанием автора.")
                continue
            users.setdefault(str(cid), {"name": name, "username": fromu.get("username", "")})
            save_users(users)
            if text:
                import datetime
                with LOG.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                                         "chat_id": cid, "sender": name, "text": text},
                                        ensure_ascii=False) + "\n")
            if text.startswith("/") and not doc:
                continue
            sender = users[str(cid)].get("name", str(cid))
            if doc:
                caption = f"[{sender} → всем]\n{text or doc.get('file_name','файл')}"
                for target in load_users():
                    api("copyMessage", chat_id=int(target), from_chat_id=cid,
                        message_id=msg["message_id"], caption=caption)
                print("relayed file", sender, doc.get("file_name"), flush=True)
            else:
                res = broadcast(text, sender)
                print("relayed", sender, len(text), res, flush=True)
        OFFSET.write_text(str(offset))


if __name__ == "__main__":
    main()
