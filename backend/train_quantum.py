import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from backend.dataset import load_manifest
from backend.quantum import QuantumComparison
from backend.spatial import SpatialModel


def train(args):
    official = load_manifest(args.mvtec_manifest, args.category)
    labeled = load_manifest(args.labeled_manifest, args.category)
    train_rows = [row for row in labeled if row['split'] == 'train']
    val_rows = [row for row in labeled if row['split'] == 'val']
    excluded = {row['sha256'] for row in official if row['split'] == 'test'}
    if excluded.intersection(row['sha256'] for row in train_rows + val_rows):
        raise ValueError('Official test images cannot enter quantum training or validation.')
    if any(not row.get('group') for row in train_rows + val_rows):
        raise ValueError('Every labeled training/validation record needs a source group ID. Assign all variants from a source to one group.')
    if {r['group'] for r in train_rows}.intersection(r['group'] for r in val_rows):
        raise ValueError('Source groups cross train/validation boundaries.')
    if len(train_rows) > 256:
        raise ValueError('Provide a predeclared stratified subset of at most 256 training samples for BOTH models.')
    model = SpatialModel.load(Path(args.artifacts) / f'{args.category}.pt', args.device)
    if set(model.training_hashes).intersection(row['sha256'] for row in val_rows):
        raise ValueError('Validation images were used to fit the spatial model.')
    def features(rows):
        vectors = []
        for row in rows:
            with Image.open(row['path']) as image:
                vectors.append(model.embedding(image))
        return np.asarray(vectors), np.asarray([row['label'] for row in rows])
    train_x, train_y = features(train_rows)
    val_x, val_y = features(val_rows)
    comparison = QuantumComparison(args.seed)
    report = comparison.fit(train_x, train_y, val_x, val_y)
    comparison.training_hashes = [row['sha256'] for row in train_rows]
    comparison.validation_hashes = [row['sha256'] for row in val_rows]
    output = Path(args.artifacts)
    joblib.dump(comparison, output / f'{args.category}-quantum.joblib')
    (output / f'{args.category}-quantum-validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mvtec-manifest', required=True)
    parser.add_argument('--labeled-manifest', required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    train(parser.parse_args())
