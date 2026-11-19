"""Construit le dataset binaire fertile / non_fertile à partir de dossiers de types de sols.

Usage (depuis la racine du projet) :  python ml/prepare_dataset.py
"""
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

RAW = Path("data_raw")
OUT = Path("dataset")
SEED = 42
RATIOS = (0.70, 0.15)          # train, val ; le reste = test
EXTS = {".jpg", ".jpeg", ".png"}
MAX_SIDE = 512                 # réduit le poids pour l'upload sur Colab

# Mots-clés (minuscules) cherchés dans le nom du dossier parent de chaque image
FERTILE_KW = ("alluvial", "black")
POOR_KW = ("laterite", "arid", "sand")

stats = defaultdict(int)


def label_from_folder(name: str):
    n = name.lower()
    if any(k in n for k in FERTILE_KW):
        return "fertile"
    if any(k in n for k in POOR_KW):
        return "non_fertile"
    return None


def load_rgb(path: Path):
    """Ouvre l'image, force le RGB, limite la taille. Retourne None si corrompue."""
    try:
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
        rgb.thumbnail((MAX_SIDE, MAX_SIDE))
        return rgb
    except Exception:
        stats["corrompues"] += 1
        return None


def save_set(paths, label, split, prefix):
    dest = OUT / split / label
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, p in enumerate(paths):
        im = load_rgb(p)
        if im is None:
            continue
        im.save(dest / f"{prefix}_{label}_{i:04d}.jpg", quality=92)
        count += 1
    return count


def main():
    random.seed(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)

    seen_hashes = set()
    items = defaultdict(list)      # label -> [chemins]
    terrain = defaultdict(list)    # photos personnelles
    ignored = defaultdict(int)     # dossiers non mappés

    for p in sorted(RAW.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        parts = [x.lower() for x in p.relative_to(RAW).parts]

        # Photos personnelles : jeu de test séparé
        if parts[0] == "terrain":
            lab = p.parent.name.lower()
            if lab in ("fertile", "non_fertile"):
                terrain[lab].append(p)
            continue

        # Images augmentées : exclues (risque de fuite de données)
        if any("cyaug" in x or "augment" in x for x in parts):
            stats["augmentées ignorées"] += 1
            continue

        label = label_from_folder(p.parent.name)
        if label is None:
            ignored[p.parent.name] += 1
            continue

        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen_hashes:
            stats["doublons"] += 1
            continue
        seen_hashes.add(digest)
        items[label].append(p)

    # Découpage stratifié train / val / test
    summary = defaultdict(dict)
    for label, paths in items.items():
        random.shuffle(paths)
        n = len(paths)
        n_tr, n_va = int(n * RATIOS[0]), int(n * RATIOS[1])
        splits = {
            "train": paths[:n_tr],
            "val": paths[n_tr:n_tr + n_va],
            "test": paths[n_tr + n_va:],
        }
        for split, sub in splits.items():
            summary[split][label] = save_set(sub, label, split, "pub")

    for label, paths in terrain.items():
        summary["test_terrain"][label] = save_set(paths, label, "test_terrain", "terrain")

    # Rapport
    print("\n=== RÉSUMÉ DU DATASET ===")
    for split in ("train", "val", "test", "test_terrain"):
        if split in summary:
            f = summary[split].get("fertile", 0)
            nf = summary[split].get("non_fertile", 0)
            print(f"{split:13s} fertile={f:4d}  non_fertile={nf:4d}  total={f + nf}")
    print("\nDossiers ignorés (non mappés) :", dict(ignored) or "aucun")
    print("Autres :", dict(stats) or "rien à signaler")


if __name__ == "__main__":
    main()