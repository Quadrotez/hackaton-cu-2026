#!/usr/bin/env bash
cd /home/gregory/Documents/Projects/astramis
while true; do
  .venv/bin/python -u tg_bot/bot.py >> /tmp/opencode/tg_bot.log 2>&1
  sleep 5
done
