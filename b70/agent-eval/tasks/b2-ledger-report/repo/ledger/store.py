"""JSON file storage. Each entry is a dict:

    {"date": "2026-09-01", "amount": 1250, "category": "food", "note": ""}

`date` is an ISO date string and `amount` is integer cents (negative for
refunds).
"""
import json
import os


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save(path, entries):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=1)
    os.replace(tmp, path)


def append(path, entry):
    entries = load(path)
    entries.append(entry)
    save(path, entries)
