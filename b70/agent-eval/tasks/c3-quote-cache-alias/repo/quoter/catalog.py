"""The price list: plans and add-ons, read from the JSON files under data/.

Each file is read once per process and then served from memory: a batch
quotes hundreds of orders, and in production the price list lives on a slow
network share.
"""
import json
import os

DATA_DIR = os.environ.get("QUOTER_DATA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

_cache = {}


class UnknownItem(LookupError):
    pass


def _load(relpath):
    """The parsed contents of DATA_DIR/relpath."""
    if relpath not in _cache:
        with open(os.path.join(DATA_DIR, relpath), encoding="utf-8") as f:
            _cache[relpath] = json.load(f)
    return _cache[relpath]


def get_plan(name):
    """The plan `name`: {"name", "lines": [{"sku", "desc", "unit_cents", "per_seat"}, ...]}."""
    try:
        return _load(os.path.join("plans", name + ".json"))
    except FileNotFoundError:
        raise UnknownItem(f"no such plan: {name}") from None


def get_addon(sku):
    """The add-on `sku`: a line like a plan's, plus "plans", the plans it is sold on."""
    addons = _load("addons.json")
    if sku not in addons:
        raise UnknownItem(f"no such add-on: {sku}")
    return addons[sku]


def clear_cache():
    """Forget every file read so far (the tests start each case from a clean slate)."""
    _cache.clear()
