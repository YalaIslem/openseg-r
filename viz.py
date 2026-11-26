"""Visualisation des détections et masques."""
from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

COLORS = [(1, 0, 0), (0, 0.4, 1), (0, 0.7, 0), (1, 0.6, 0), (0.7, 0, 0.9)]


def show_result(image, result: dict, title: str = "", ax=None):
    """Affiche l'image avec masques colorés (+ contour), boîtes et scores."""
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(7, 6))

    overlay = np.array(image.convert("RGB")).astype(float) / 255
    for i, mask in enumerate(result["masks"]):
        c = np.array(COLORS[i % len(COLORS)])
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * c
    ax.imshow(overlay)

    # Contour blanc épais puis noir fin : visible sur tout fond
    for mask in result["masks"]:
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=2.5)
            ax.contour(mask.astype(float), levels=[0.5], colors="black", linewidths=1)

    for i, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
        x0, y0, x1, y1 = box
        c = COLORS[i % len(COLORS)]
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=c, linewidth=2))
        ax.text(x0, max(y0 - 4, 10), f"{score:.2f}", color="white", fontsize=10,
                bbox=dict(facecolor=c, alpha=0.8, pad=1, edgecolor="none"))

    n = len(result["boxes"])
    ax.set_title(f"{title} : {n} détection(s)" if title else f"{n} détection(s)")
    ax.axis("off")
    if own_fig:
        plt.show()
    return ax
