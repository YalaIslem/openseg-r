cat > tests/test_metrics.py << 'EOF'
"""Tests des métriques sur des cas dont la réponse est connue."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openseg.metrics import (  # noqa: E402
    aggregate_group, bootstrap_macro_ci, evaluate_pair, union_iou,
)

H, W = 40, 40


def box(r0, r1, c0, c1):
    """Un masque (1, H, W) rectangulaire."""
    m = np.zeros((1, H, W), dtype=bool)
    m[0, r0:r1, c0:c1] = True
    return m


def stack(*masks):
    return np.concatenate(masks, axis=0)


def empty():
    return np.zeros((0, H, W), dtype=bool)


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_identical_masks_give_iou_one():
    g = box(0, 10, 0, 10)
    assert close(union_iou(g, g), 1.0)


def test_disjoint_masks_give_zero():
    assert close(union_iou(box(0, 10, 0, 10), box(20, 30, 20, 30)), 0.0)


def test_partial_overlap_known_value():
    # gt : colonnes 0-9, pred : colonnes 5-14 (10 lignes) -> inter 50, union 150
    assert close(union_iou(box(0, 10, 5, 15), box(0, 10, 0, 10)), 1 / 3)


def test_no_prediction_counts_as_zero():
    r = evaluate_pair(empty(), box(0, 10, 0, 10))
    assert close(r["union_iou"], 0.0)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)


def test_one_of_two_objects_found():
    g1, g2 = box(0, 10, 0, 10), box(20, 30, 20, 30)
    r = evaluate_pair(g1, stack(g1, g2))
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1)
    assert close(r["union_iou"], 0.5)  # 100 px trouvés sur 200 px de référence


def test_one_prediction_cannot_match_two_objects():
    g1, g2 = box(0, 10, 0, 10), box(0, 10, 10, 20)
    pred = box(0, 10, 0, 15)  # IoU 0.667 avec g1, 0.25 avec g2
    r = evaluate_pair(pred, stack(g1, g2))
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1)


def test_duplicate_predictions_are_false_positive():
    g = box(0, 10, 0, 10)
    r = evaluate_pair(stack(g, g), g)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)


def test_iou_below_threshold_is_not_a_match():
    g = box(0, 10, 0, 10)
    pred = box(0, 10, 0, 4)  # inter 40, union 100 -> IoU 0.4
    r = evaluate_pair(pred, g, iou_thr=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert close(r["union_iou"], 0.4)


def test_aggregate_group():
    rows = [
        {"union_iou": 0.8, "tp": 1, "fp": 0, "fn": 0, "matched_ious": [0.9]},
        {"union_iou": 0.0, "tp": 0, "fp": 2, "fn": 1, "matched_ious": []},
    ]
    a = aggregate_group(rows)
    assert (a["tp"], a["fp"], a["fn"]) == (1, 2, 1)
    assert close(a["precision"], 1 / 3)
    assert close(a["recall"], 1 / 2)
    assert close(a["f1"], 0.4)
    assert close(a["mean_union_iou"], 0.4)
    assert close(a["mean_matched_iou"], 0.9)


def test_bootstrap_constant_values():
    p, lo, hi = bootstrap_macro_ci({"a": [0.5] * 10, "b": [0.5] * 10})
    assert close(p, 0.5) and close(lo, 0.5) and close(hi, 0.5)


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(1)
    vals = {"a": rng.random(40), "b": rng.random(40)}
    p, lo, hi = bootstrap_macro_ci(vals)
    assert lo <= p <= hi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK  ", name)
EOF
echo ok
