"""Where the last export run stopped, kept in a small JSON file."""
import json
import os


def load(path):
    """The saved position, or None if there is no checkpoint yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["updated_at"]


def save(path, position):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"updated_at": position}, f)
    os.replace(tmp, path)
