"""Construit le découpage dev/test. Unité d'évaluation : la paire (image, groupe).

Chaque image apparaît dans au plus UNE paire, donc dev et test sont disjoints.
Le fichier produit (split_v1.json) fait foi : il est versionné dans Git.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="chemin de annotations.json (TACO)")
    ap.add_argument("--categories", default="configs/categories.json")
    ap.add_argument("--out", default="configs/split_v1.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=40, help="paires de test par groupe")
    ap.add_argument("--n-dev", type=int, default=20, help="paires de dev par groupe")
    args = ap.parse_args()

    coco = json.loads(Path(args.annotations).read_text())
    cfg = json.loads(Path(args.categories).read_text())["groups"]

    # catégorie TACO -> groupe
    name2id = {c["name"]: c["id"] for c in coco["categories"]}
    id2group = {}
    for group, spec in cfg.items():
        for name in spec["taco_categories"]:
            if name not in name2id:
                raise SystemExit(f"Catégorie TACO introuvable : {name!r} (groupe {group})")
            id2group[name2id[name]] = group

    file_name = {im["id"]: im["file_name"] for im in coco["images"]}

    # nombre d'instances de référence par (image, groupe)
    n_inst = defaultdict(int)
    for ann in coco["annotations"]:
        group = id2group.get(ann["category_id"])
        if group is not None:
            n_inst[(ann["image_id"], group)] += 1

    candidates = defaultdict(list)
    for (image_id, group) in n_inst:
        candidates[group].append(image_id)
    for group in candidates:
        candidates[group].sort()  # ordre déterministe avant tirage

    rng = random.Random(args.seed)
    n_total = args.n_dev + args.n_test
    used, dev, test = set(), [], []

    # On traite d'abord les groupes les plus rares, pour qu'ils ne manquent pas d'images
    for group in sorted(cfg, key=lambda g: len(candidates[g])):
        pool = [i for i in candidates[group] if i not in used]
        k = min(len(pool), n_total)
        if k < n_total:
            print(f"ATTENTION : groupe {group!r} : {k} paires disponibles sur {n_total} demandées")
        chosen = rng.sample(pool, k)
        n_dev_g = round(k * args.n_dev / n_total)
        for j, image_id in enumerate(chosen):
            used.add(image_id)
            row = {
                "image_id": image_id,
                "file_name": file_name[image_id],
                "group": group,
                "n_instances": n_inst[(image_id, group)],
            }
            (dev if j < n_dev_g else test).append(row)

    key = lambda r: (r["group"], r["image_id"])
    dev.sort(key=key)
    test.sort(key=key)

    out = {
        "seed": args.seed,
        "n_dev_per_group": args.n_dev,
        "n_test_per_group": args.n_test,
        "categories_file": args.categories,
        "dev": dev,
        "test": test,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # Résumé
    print(f"\n{'groupe':10s} {'disponibles':>12s} {'dev':>5s} {'test':>5s} {'inst. moy. (test)':>18s}")
    for group in cfg:
        d = [r for r in dev if r["group"] == group]
        t = [r for r in test if r["group"] == group]
        mean_inst = sum(r["n_instances"] for r in t) / len(t) if t else 0
        print(f"{group:10s} {len(candidates[group]):12d} {len(d):5d} {len(t):5d} {mean_inst:18.2f}")
    print(f"\nTotal : dev = {len(dev)} paires, test = {len(test)} paires -> {args.out}")


if __name__ == "__main__":
    main()
