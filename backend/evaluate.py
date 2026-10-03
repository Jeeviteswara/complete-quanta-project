import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

from backend.artifacts import validate_comparison
from backend.dataset import load_manifest
from backend.localization import extract_regions, image_metrics, match_boxes
from backend.spatial import SpatialModel


def evaluate(args):
    rows = [row for row in load_manifest(args.manifest, args.category, require_test_masks=True) if row['split'] == 'test']
    if not rows:
        raise ValueError('No test images in this manifest.')
    artifacts = Path(args.artifacts)
    model = SpatialModel.load(artifacts / f'{args.category}.pt', args.device, expected_category=args.category)
    test_hashes = {row['sha256'] for row in rows}
    if test_hashes.intersection(model.training_hashes + model.validation_hashes):
        raise ValueError('Test data overlaps spatial training or threshold calibration.')
    quantum_path = artifacts / f'{args.category}-quantum.joblib'
    comparison = joblib.load(quantum_path) if quantum_path.is_file() else None
    if comparison:
        validate_comparison(comparison, model)
    if comparison and test_hashes.intersection(comparison.training_hashes + comparison.validation_hashes):
        raise ValueError('Test data overlaps quantum/classical fitting or selection.')
    labels, predictions, embeddings, records = [], [], [], []
    intersection = union = predicted_pixels = target_pixels = 0
    box_totals = dict(tp=0, fp=0, fn=0, matched_iou_sum=0., ground_truth_count=0)
    positive_iou, positive_dice = [], []
    started = time.perf_counter()
    for row in rows:
        with Image.open(row['path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            result, mask = model.predict(image, include_heatmap=False)
            if comparison:
                embeddings.append(model.embedding(image))
        if row['mask']:
            with Image.open(row['mask']) as source:
                truth = np.asarray(ImageOps.exif_transpose(source).convert('L')) > 0
        else:
            truth = np.zeros_like(mask)
        if truth.shape != mask.shape:
            raise ValueError('Ground truth must align with original-image coordinates.')
        ground_truth, _ = extract_regions(truth.astype(float), .5, min_area=1)
        matches = match_boxes(result['boxes'], ground_truth)
        for key in box_totals:
            box_totals[key] += matches[key]
        i, u = int(np.sum(mask & truth)), int(np.sum(mask | truth))
        predicted_count, target_count = int(mask.sum()), int(truth.sum())
        intersection += i; union += u; predicted_pixels += predicted_count; target_pixels += target_count
        if row['label']:
            positive_iou.append(i / u if u else 1.)
            positive_dice.append(2 * i / (predicted_count + target_count) if predicted_count + target_count else 1.)
        labels.append(row['label'])
        predictions.append(int(bool(result['boxes'])))
        records.append(dict(path=row['path'], sha256=row['sha256'], target=row['label'], result=result,
                            pixel_iou=i / u if u else 1., box_matches=matches))
        print(f'Evaluated {len(records)}/{len(rows)}', flush=True)
    report = dict(status='evaluated', category=args.category, test_images=len(rows), elapsed_seconds=time.perf_counter() - started,
                  classification=image_metrics(labels, predictions),
                  localization=dict(pixel_iou=intersection / union if union else None,
                                    pixel_dice=2 * intersection / (predicted_pixels + target_pixels) if predicted_pixels + target_pixels else None,
                                    defective_image_macro_iou=float(np.mean(positive_iou)) if positive_iou else None,
                                    defective_image_macro_dice=float(np.mean(positive_dice)) if positive_dice else None,
                                    box_iou_with_unmatched_truth_as_zero=box_totals['matched_iou_sum'] / box_totals['ground_truth_count'] if box_totals['ground_truth_count'] else None,
                                    box_precision=box_totals['tp'] / (box_totals['tp'] + box_totals['fp']) if box_totals['tp'] + box_totals['fp'] else None,
                                    box_recall=box_totals['tp'] / (box_totals['tp'] + box_totals['fn']) if box_totals['tp'] + box_totals['fn'] else None,
                                    box_matching_iou_threshold=.5, box_counts=box_totals),
                  quantum_comparison=None,
                  limitations=['Anomalies are not proof of physical damage.', 'Classification treats REVIEW with valid regions as positive for benchmark scoring.', 'NORMAL means no valid retained regions, not certified defect-free.', 'Boxes are axis-aligned connected-component bounds.', 'No assertion of quantum advantage. Repeat on multiple seeds and report uncertainty.'],
                  images=records)
    if comparison:
        started = time.perf_counter()
        outputs = comparison.predict(np.asarray(embeddings))
        report['quantum_comparison'] = {name: image_metrics(labels, outputs[name]) for name in ('classical', 'quantum')}
        report['quantum_comparison']['combined_inference_seconds'] = time.perf_counter() - started
        report['quantum_comparison']['protocol'] = 'Both models evaluated on the identical complete official test split; preprocessing fit only on labeled train.'
    output = Path(args.output) if args.output else artifacts / f'{args.category}-evaluation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Measured report written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--output', default=None, help='Defaults to <artifacts>/<category>-evaluation.json for the research dashboard.')
    parser.add_argument('--device', default=None)
    evaluate(parser.parse_args())
