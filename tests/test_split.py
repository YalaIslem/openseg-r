"""Vérifie l'intégrité de configs/split_v1.json."""
import json
from collections import Counter
from pathlib import Path

SPLIT = Path(__file__).resolve().parents[1] / "configs" / "split_v1.json"
GROUPS = {"bottle", "can", "bag", "cigarette"}


def load():
    return json.loads(SPLIT.read_text())


def test_dev_test_disjoint():
    s = load()
    dev_ids = {r["image_id"] for r in s["dev"]}
    test_ids = {r["image_id"] for r in s["test"]}
    assert not (dev_ids & test_ids), "Fuite : des images sont à la fois dans dev et test"


def test_each_image_used_once():
    s = load()
    ids = [r["image_id"] for r in s["dev"] + s["test"]]
    assert len(ids) == len(set(ids)), "Une image apparaît dans plusieurs paires"


def test_counts_per_group():
    s = load()
    for key, expected in (("dev", s["n_dev_per_group"]), ("test", s["n_test_per_group"])):
        counts = Counter(r["group"] for r in s[key])
        assert set(counts) == GROUPS, f"{key} : groupes inattendus {set(counts)}"
        assert all(c == expected for c in counts.values()), f"{key} : effectifs {dict(counts)}"


def test_every_pair_has_ground_truth():
    s = load()
    assert all(r["n_instances"] >= 1 for r in s["dev"] + s["test"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK  ", name)
