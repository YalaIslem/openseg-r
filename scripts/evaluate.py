"""Évaluation du pipeline sur le découpage TACO (protocole : docs/protocol.md).

Modes :
  full   : Grounding DINO (texte -> boîtes) puis SAM 2 (boîtes -> masques)
  oracle : SAM 2 reçoit les boîtes englobantes des masques de référence
           (borne haute de SAM 2 seul, indépendante du détecteur)

Une paire qui plante est consignée dans <out>.errors.json et exclue du CSV
(ce n'est PAS un IoU de 0). Une paire sans aucune prédiction compte pour 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openseg.metrics import aggregate_group, bootstrap_macro_ci, evaluate_pair  # noqa: E402

FIELDS = [
    "subset", "mode", "group", "image_id", "file_name", "prompt",
    "box_threshold", "text_threshold", "n_gt", "n_pred", "tp", "fp", "fn",
    "union_iou", "mean_matched_iou", "matched_ious", "max_score", "t_det", "t_sam",
]


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Boîtes [x_min, y_min, x_max, y_max] (pixels) des masques de référence."""
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def summarize(csv_path: Path) -> None:
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    if not rows:
        print("CSV vide.")
        return
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append({
            "union_iou": float(r["union_iou"]),
            "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]),
            "matched_ious": json.loads(r["matched_ious"]),
        })

    print(f"\n{'groupe':10s} {'paires':>6s} {'IoU union':>10s} {'précision':>10s} "
          f"{'rappel':>8s} {'F1':>6s} {'IoU appariés':>13s}")
    for g, rs in sorted(by_group.items()):
        a = aggregate_group(rs)
        print(f"{g:10s} {a['n_pairs']:6d} {a['mean_union_iou']:10.3f} {a['precision']:10.3f} "
              f"{a['recall']:8.3f} {a['f1']:6.3f} {a['mean_matched_iou']:13.3f}")

    point, lo, hi = bootstrap_macro_ci(
        {g: [r["union_iou"] for r in rs] for g, rs in by_group.items()}
    )
    print(f"\nmIoU macro (IoU union, moyenne des {len(by_group)} groupes) : "
          f"{point:.3f}  IC95 % [{lo:.3f} ; {hi:.3f}]")
    t_det = [float(r["t_det"]) for r in rows if r["t_det"] != ""]
    t_sam = [float(r["t_sam"]) for r in rows]
    if t_det:
        print(f"Temps moyen par paire : détecteur {np.mean(t_det):.2f} s | SAM 2 {np.mean(t_sam):.2f} s")
    else:
        print(f"Temps moyen par paire : SAM 2 {np.mean(t_sam):.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dossier contenant annotations.json et les images")
    ap.add_argument("--split", default="configs/split_v1.json")
    ap.add_argument("--categories", default="configs/categories.json")
    ap.add_argument("--subset", choices=["dev", "test"], default="dev")
    ap.add_argument("--mode", choices=["full", "oracle"], default="full")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.3)
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--limit", type=int, default=0, help="n paires espacées (test rapide)")
    ap.add_argument("--out", required=True, help="fichier CSV de sortie")
    ap.add_argument("--final", action="store_true", help="obligatoire pour --subset test")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.summarize_only:
        summarize(out)
        return
    if args.subset == "test" and not args.final:
        sys.exit("Le test n'est lancé qu'une fois, configuration figée : ajoute --final.")

    import torch  # importés ici pour que --summarize-only reste léger
    from openseg.data import TacoSubset
    from openseg.pipeline import OpenVocabSegmenter

    # Un fichier de configuration par CSV : jamais deux configurations dans un même fichier
    cfg = {"subset": args.subset, "mode": args.mode, "detector": args.detector, "sam": args.sam}
    if args.mode == "full":
        cfg.update(box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = out.with_suffix(".config.json")
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        sys.exit(f"Configuration différente de celle de {out.name} : utilise un autre --out.")
    cfg_path.write_text(json.dumps(cfg, indent=1))

    ds = TacoSubset(args.root, args.split, args.categories, args.subset)
    idx = list(range(len(ds)))
    if args.limit:
        step = max(1, len(ds) // args.limit)
        idx = idx[::step][: args.limit]

    done = set()
    if out.exists() and out.stat().st_size > 0:
        with open(out, newline="", encoding="utf-8") as f:
            done = {(int(r["image_id"]), r["group"]) for r in csv.DictReader(f)}
        print(f"Reprise : {len(done)} paires déjà dans {out.name}")
    new_file = not out.exists() or out.stat().st_size == 0

    seg = OpenVocabSegmenter(detector_id=args.detector, sam_id=args.sam)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    errors = []
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for n, i in enumerate(idx, 1):
            row = ds.items[i]
            if (row["image_id"], row["group"]) in done:
                continue
            try:
                s = ds[i]
                image, gt = s["image"], s["gt_masks"]

                t0 = time.perf_counter()
                if args.mode == "full":
                    boxes, scores, _ = seg.detect(image, s["prompt"],
                                                  args.box_threshold, args.text_threshold)
                    max_score = f"{float(scores.max()):.4f}" if len(scores) else ""
                else:
                    boxes, max_score = boxes_from_masks(gt), ""
                sync()
                t1 = time.perf_counter()
                masks = seg.segment_boxes(image, boxes)
                sync()
                t2 = time.perf_counter()

                if masks.shape[1:] != gt.shape[1:]:
                    raise ValueError(f"forme des masques {masks.shape[1:]} != référence {gt.shape[1:]}")

                res = evaluate_pair(masks, gt)
                matched = res["matched_ious"]
                w.writerow({
                    "subset": args.subset, "mode": args.mode, "group": s["group"],
                    "image_id": s["image_id"], "file_name": s["file_name"], "prompt": s["prompt"],
                    "box_threshold": args.box_threshold if args.mode == "full" else "",
                    "text_threshold": args.text_threshold if args.mode == "full" else "",
                    "n_gt": res["n_gt"], "n_pred": res["n_pred"],
                    "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
                    "union_iou": f"{res['union_iou']:.6f}",
                    "mean_matched_iou": f"{np.mean(matched):.6f}" if matched else "",
                    "matched_ious": json.dumps([round(x, 4) for x in matched]),
                    "max_score": max_score,
                    "t_det": f"{t1 - t0:.3f}" if args.mode == "full" else "",
                    "t_sam": f"{t2 - t1:.3f}",
                })
                fh.flush()
            except Exception as e:  # noqa: BLE001
                errors.append({"image_id": row["image_id"], "file_name": row["file_name"],
                               "group": row["group"], "error": f"{type(e).__name__}: {e}"})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if n % 10 == 0 or n == len(idx):
                print(f"{n}/{len(idx)} paires traitées ({len(errors)} erreur(s))")

    if errors:
        out.with_suffix(".errors.json").write_text(json.dumps(errors, indent=1, ensure_ascii=False))
        print("\nPaires en erreur (exclues du CSV) :")
        for e in errors[:10]:
            print("  ", e["file_name"], e["group"], "->", e["error"])
    summarize(out)


if __name__ == "__main__":
    main()
