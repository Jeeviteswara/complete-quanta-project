import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CATEGORIES = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def image_hash(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.load()
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def validate_manifest(rows):
    if not rows:
        raise ValueError('Dataset is empty.')
    seen = {}
    for row in rows:
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Unknown dataset split.')
        if row['label'] not in {0, 1}:
            raise ValueError('Labels must be 0 (normal) or 1 (defective).')
        digest = image_hash(row['path'])
        if row.get('sha256') and digest != row['sha256']:
            raise ValueError(f"Image changed after manifest creation: {row['path']}")
        row['sha256'] = digest
        if digest in seen and seen[digest] != row['split']:
            raise ValueError('Duplicate image content crosses train, validation, or test boundaries.')
        seen[digest] = row['split']
        if row.get('mask'):
            with Image.open(row['path']) as image, Image.open(row['mask']) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Mask size does not match image: {row['path']}")
                if not np.asarray(mask.convert('L')).any():
                    raise ValueError(f"Defective image has an empty mask: {row['path']}")
    return rows


def inspect_dataset(root, category, seed=42):
    root = Path(root).resolve() / category
    if category not in CATEGORIES:
        raise ValueError('Unknown MVTec category.')
    normal = sorted(p for p in (root / 'train' / 'good').glob('*') if p.suffix.lower() in EXTENSIONS)
    if len(normal) < 10:
        raise ValueError(f'Expected at least 10 official train/good images in {root}. Dataset has not been provided.')
    grouped = {}
    for path in normal:
        grouped.setdefault(image_hash(path), []).append(path)
    digests = sorted(grouped)
    if len(digests) < 10:
        raise ValueError('At least 10 unique normal source images are required.')
    order = np.random.default_rng(seed).permutation(len(digests))
    validation = {digests[index] for index in order[:max(2, round(len(digests) * .2))]}
    rows = []
    for digest, paths in grouped.items():
        for path in paths:
            rows.append(dict(path=str(path), category=category, label=0, mask=None,
                             split='val' if digest in validation else 'train', sha256=digest))
    test_paths = sorted(p for p in (root / 'test').glob('*/*') if p.suffix.lower() in EXTENSIONS)
    if not test_paths:
        raise ValueError('Official test images are missing.')
    for path in test_paths:
        defect = path.parent.name
        mask = root / 'ground_truth' / defect / f'{path.stem}_mask.png'
        if defect != 'good' and not mask.is_file():
            raise ValueError(f'Missing ground-truth mask: {mask}')
        rows.append(dict(path=str(path), category=category, label=int(defect != 'good'),
                         mask=str(mask) if defect != 'good' else None, split='test', sha256=image_hash(path)))
    return validate_manifest(rows)


def load_manifest(path, category=None):
    document = json.loads(Path(path).read_text())
    rows = validate_manifest(document['images'])
    return [row for row in rows if category is None or row.get('category') == category]


def main():
    parser = argparse.ArgumentParser(description='Inspect MVTec files and create an immutable source-level split.')
    parser.add_argument('--root', required=True)
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = inspect_dataset(args.root, args.category, args.seed)
    counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'val', 'test')}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(seed=args.seed, category=args.category, counts=counts, images=rows), indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
