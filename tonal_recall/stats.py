import os
import json
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

STATS_FILE = os.path.join(os.path.dirname(__file__), "stats.json")


def load_stats():
    if not os.path.exists(STATS_FILE):
        return {"high_score_nps": 0, "fastest_note": None, "history": []}
    with open(STATS_FILE, "r") as f:
        return json.load(f)


def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def update_stats(nps, fastest):
    stats = load_stats()
    updated = False
    if nps > stats.get("high_score_nps", 0):
        stats["high_score_nps"] = nps
        updated = True
    if fastest is not None and (
        stats.get("fastest_note") is None or fastest < stats["fastest_note"]
    ):
        stats["fastest_note"] = fastest
        updated = True
    # Optionally store history for stats over time
    stats.setdefault("history", []).append({"nps": nps, "fastest": fastest})
    save_stats(stats)
    return stats, updated
