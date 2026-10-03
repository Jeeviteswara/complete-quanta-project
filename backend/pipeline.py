import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from backend.dataset import CATEGORIES, inspect_dataset


def run_pipeline(args):
    if args.category not in CATEGORIES:
        raise ValueError('Unknown MVTec category.')
    data = Path(args.data).resolve()
    artifacts = Path(args.artifacts).resolve()
    category = args.category
    manifest_path = data / f'{category}-manifest.json'
    names = [f'{category}.pt', f'{category}-training.json', f'{category}-evaluation.json']
    quantum_names = [f'{category}-quantum.joblib', f'{category}-quantum-validation.json']
    existing = [path for path in [manifest_path, *[artifacts / name for name in names + quantum_names]] if path.exists()]
    if existing and not args.overwrite:
        raise ValueError('Category outputs already exist. Use the individual stage commands to resume, or --overwrite to explicitly replace a complete run.')
    if not args.labeled_manifest and any((artifacts / name).exists() for name in quantum_names):
        raise ValueError('An existing quantum comparison would become stale. Supply --labeled-manifest to retrain it with this run, or use a separate artifacts directory.')
    if args.labeled_manifest and not Path(args.labeled_manifest).is_file():
        raise ValueError('The independent labeled manifest does not exist.')
    if not 128 <= args.memory_size <= 32768:
        raise ValueError('Memory bank size must be between 128 and 32768 patches.')

    print('Stage 1: inspect images, masks, and split isolation', flush=True)
    rows = inspect_dataset(args.root, category, args.seed)
    document = dict(category=category, seed=args.seed,
                    counts={split: sum(row['split'] == split for row in rows) for split in ('train', 'val', 'test')}, images=rows)
    from backend.spatial import train as train_spatial
    from backend.evaluate import evaluate

    artifacts.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    # Failed training/evaluation must not replace a previously usable experiment.
    with tempfile.TemporaryDirectory(prefix='.quanta-run-', dir=artifacts) as temporary:
        staging = Path(temporary)
        staged_manifest = staging / manifest_path.name
        staged_manifest.write_text(json.dumps(document, indent=2), encoding='utf-8')
        print('Stage 2: train and calibrate spatial baseline', flush=True)
        train_spatial(staged_manifest, category, staging, args.device, args.seed, args.memory_size)
        if args.labeled_manifest:
            from backend.train_quantum import train as train_quantum
            print('Stage 3: train matched quantum and classical classifiers', flush=True)
            train_quantum(SimpleNamespace(mvtec_manifest=staged_manifest, labeled_manifest=args.labeled_manifest,
                                          category=category, artifacts=staging, device=args.device, seed=args.seed))
            names.extend(quantum_names)
        else:
            print('Stage 3: quantum comparison omitted; independent labeled data was not supplied', flush=True)
        print('Stage 4: evaluate on the untouched official test split', flush=True)
        evaluate(SimpleNamespace(manifest=staged_manifest, category=category, artifacts=staging,
                                 device=args.device, output=staging / f'{category}-evaluation.json'))
        for name in names:
            (staging / name).replace(artifacts / name)
        # The data directory may be on a different drive from artifacts on Windows.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=data, suffix='.json', delete=False) as output:
            temporary_manifest = Path(output.name)
            output.write(json.dumps(document, indent=2))
        try:
            temporary_manifest.replace(manifest_path)
        finally:
            temporary_manifest.unlink(missing_ok=True)
        # The dashboard's historical-report check uses mtimes; commit the report last.
        (artifacts / f'{category}-evaluation.json').touch()
    print(f'Completed {category}. Reports are ready for review; this is not manufacturing certification.', flush=True)
    return artifacts / f'{category}-evaluation.json'


def main():
    parser = argparse.ArgumentParser(description='Run the existing dataset, spatial, optional quantum, and evaluation stages without fabricating data.')
    parser.add_argument('--root', required=True, help='MVTec AD root containing category folders.')
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument('--data', default='data')
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--labeled-manifest', default=None, help='Independent labeled train/validation data for the optional quantum comparison.')
    parser.add_argument('--device', default=None, choices=('cpu', 'cuda'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--memory-size', type=int, default=8192)
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace existing category outputs only after all computation succeeds.')
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Pipeline stopped: {error}\n')


if __name__ == '__main__':
    main()
