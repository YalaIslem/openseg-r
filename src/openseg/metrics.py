cat > src/openseg/metrics.py << 'EOF'
"""Métriques d'évaluation : IoU union, appariement par instance, bootstrap.

Convention : les masques sont des tableaux booléens de forme (N, H, W).
Ce module n'importe que numpy et scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU de deux masques booléens (H, W). Deux masques vides valent 1.0."""
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return np.count_nonzero(a & b) / union


def union_mask(masks: np.ndarray, shape) -> np.ndarray:
    """Fusionne (N, H, W) en (H, W). Sans masque : un masque vide."""
    if len(masks) == 0:
        return np.zeros(shape, dtype=bool)
    return np.any(masks, axis=0)


def union_iou(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    """IoU entre l'union des prédictions et l'union des références.
    Sans aucune prédiction, le résultat est 0 (la paire n'est pas écartée)."""
    shape = gt_masks.shape[1:]
    return mask_iou(union_mask(pred_masks, shape), union_mask(gt_masks, shape))


def iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """Matrice (n_pred, n_gt) des IoU entre masques pris deux à deux."""
    m = np.zeros((len(pred_masks), len(gt_masks)), dtype=float)
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            m[i, j] = mask_iou(p, g)
    return m


def match_instances(pred_masks, gt_masks, iou_thr: float = 0.5):
    """Appariement un-à-un (hongrois). Renvoie une liste de (i_pred, j_gt, iou)
    pour les paires dont l'IoU est >= iou_thr."""
    m = iou_matrix(pred_masks, gt_masks)
    if m.size == 0:
        return []
    # Les IoU sous le seuil sont mis à 0 pour que l'algorithme maximise
    # le nombre de paires valides, pas la somme brute des IoU.
    scores = np.where(m >= iou_thr, m, 0.0)
    rows, cols = linear_sum_assignment(-scores)
    return [(int(i), int(j), float(m[i, j])) for i, j in zip(rows, cols) if m[i, j] >= iou_thr]


def evaluate_pair(pred_masks: np.ndarray, gt_masks: np.ndarray, iou_thr: float = 0.5) -> dict:
    """Toutes les mesures pour une paire (image, groupe)."""
    matches = match_instances(pred_masks, gt_masks, iou_thr)
    tp = len(matches)
    return {
        "union_iou": union_iou(pred_masks, gt_masks),
        "n_pred": len(pred_masks),
        "n_gt": len(gt_masks),
        "tp": tp,
        "fp": len(pred_masks) - tp,
        "fn": len(gt_masks) - tp,
        "matched_ious": [mt[2] for mt in matches],
    }


def aggregate_group(rows: list[dict]) -> dict:
    """Agrège les paires d'un groupe. tp/fp/fn sont sommés (agrégation micro).
    Une quantité indéfinie (division par zéro) vaut NaN, jamais 0."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    matched = [x for r in rows for x in r["matched_ious"]]
    nan = float("nan")
    return {
        "n_pairs": len(rows),
        "mean_union_iou": float(np.mean([r["union_iou"] for r in rows])),
        "precision": tp / (tp + fp) if tp + fp > 0 else nan,
        "recall": tp / (tp + fn) if tp + fn > 0 else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else nan,
        "mean_matched_iou": float(np.mean(matched)) if matched else nan,
        "tp": tp, "fp": fp, "fn": fn,
    }


def bootstrap_macro_ci(values_by_group: dict, n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05):
    """Macro-moyenne (moyenne des moyennes par groupe) et intervalle de confiance
    bootstrap. On ré-échantillonne les paires AVEC remise, dans chaque groupe.
    Renvoie (estimation, borne basse, borne haute)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(v, dtype=float) for v in values_by_group.values()]
    point = float(np.mean([a.mean() for a in arrays]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arrays])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
EOF
echo ok
