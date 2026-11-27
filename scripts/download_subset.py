"""Télécharge uniquement les images du découpage dev/test de TACO.

- Télécharge l'image ORIGINALE (flickr_url), celle sur laquelle les masques sont définis.
- Reprend là où il s'est arrêté (fichiers déjà présents ignorés).
- Ne s'arrête pas à la première erreur : les échecs sont comptés et listés.
- Vérifie la taille de chaque image contre width/height de l'annotation.
Seule la bibliothèque standard et Pillow sont utilisées.
"""
import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

from PIL import Image


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 30):
    """Télécharge url vers dest. Renvoie None si OK, sinon un message d'erreur."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "openseg-r-student-project"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp.write_bytes(data)
            tmp.replace(dest)  # écriture atomique
            return None
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--split", default="configs/split_v1.json")
    ap.add_argument("--out-dir", required=True, help="dossier de destination (ex. sur Drive)")
    ap.add_argument("--limit", type=int, default=0, help="test rapide : n images réparties dans le découpage")
    args = ap.parse_args()

    coco = json.loads(Path(args.annotations).read_text())
    split = json.loads(Path(args.split).read_text())
    images = {im["id"]: im for im in coco["images"]}

    rows = split["dev"] + split["test"]
    if args.limit:
        step = max(1, len(rows) // args.limit)
        rows = rows[::step][: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.annotations, out_dir / "annotations.json")

    failed, size_mismatch, exif_rotated, n_new = [], [], [], 0
    for k, row in enumerate(rows, 1):
        im = images[row["image_id"]]
        dest = out_dir / im["file_name"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            err = fetch(im["flickr_url"], dest)
            if err:
                failed.append((im["file_name"], err))
                continue
            n_new += 1

        try:
            with Image.open(dest) as img:
                size = img.size
                orientation = img.getexif().get(0x0112, 1)
        except Exception as e:  # noqa: BLE001
            failed.append((im["file_name"], f"image illisible : {e}"))
            continue

        if size != (im["width"], im["height"]):
            size_mismatch.append((im["file_name"], size, (im["width"], im["height"])))
        if orientation not in (None, 1):
            exif_rotated.append((im["file_name"], orientation))

        if k % 20 == 0 or k == len(rows):
            print(f"{k}/{len(rows)} traitées")

    print("\n=== Bilan ===")
    print(f"Images visées               : {len(rows)}")
    print(f"Téléchargées (nouvelles)    : {n_new}")
    print(f"Échecs                      : {len(failed)}")
    print(f"Taille != annotation        : {len(size_mismatch)}")
    print(f"Orientation EXIF non triviale : {len(exif_rotated)}")
    for name, err in failed[:10]:
        print("  ÉCHEC", name, "->", err)
    for name, got, exp in size_mismatch[:10]:
        print("  TAILLE", name, "lue", got, "attendue", exp)
    for name, o in exif_rotated[:10]:
        print("  EXIF", name, "orientation", o)

    report = {"failed": failed, "size_mismatch": size_mismatch, "exif_rotated": exif_rotated}
    (out_dir / "download_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
