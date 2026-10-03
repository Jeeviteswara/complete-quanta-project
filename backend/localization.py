import base64
from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def extract_regions(scores, threshold, min_area=32):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('Expected a finite two-dimensional spatial score map.')
    if not np.isfinite(threshold) or min_area < 1:
        raise ValueError('Threshold and minimum region area are invalid.')
    labeled, count = ndimage.label(scores > threshold, structure=np.ones((3, 3)))
    mask = np.zeros(scores.shape, dtype=bool)
    boxes = []
    slices = ndimage.find_objects(labeled)
    for index in range(1, count + 1):
        region = labeled == index
        if np.count_nonzero(region) < min_area:
            continue
        y, x = slices[index - 1]
        mask[region] = True
        boxes.append(dict(id=len(boxes) + 1, label='Anomaly region', x=int(x.start), y=int(y.start),
                          width=int(x.stop - x.start), height=int(y.stop - y.start),
                          score=float(scores[region].mean())))
    return boxes, mask


def image_metrics(target, predicted):
    target, predicted = np.asarray(target, dtype=bool), np.asarray(predicted, dtype=bool)
    tp = int(np.sum(target & predicted)); tn = int(np.sum(~target & ~predicted))
    fp = int(np.sum(~target & predicted)); fn = int(np.sum(target & ~predicted))
    safe = lambda a, b: a / b if b else None
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    return dict(precision=precision, recall=recall, f1=safe(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                false_positive_rate=safe(fp, fp + tn), missed_defect_rate=safe(fn, fn + tp),
                tp=tp, tn=tn, fp=fp, fn=fn)


def box_iou(a, b):
    overlap_w = max(0, min(a['x'] + a['width'], b['x'] + b['width']) - max(a['x'], b['x']))
    overlap_h = max(0, min(a['y'] + a['height'], b['y'] + b['height']) - max(a['y'], b['y']))
    intersection = overlap_w * overlap_h
    union = a['width'] * a['height'] + b['width'] * b['height'] - intersection
    return intersection / union if union else 0.0


def match_boxes(predicted, truth, iou_threshold=.5):
    if not predicted or not truth:
        return dict(tp=0, fp=len(predicted), fn=len(truth), matched_iou_sum=0.0, ground_truth_count=len(truth))
    overlaps = np.array([[box_iou(a, b) for b in truth] for a in predicted])
    # Prioritize the number of valid one-to-one matches, then their IoU.
    objective = np.where(overlaps >= iou_threshold, 1 + overlaps / (min(overlaps.shape) + 1), 0)
    rows, cols = linear_sum_assignment(-objective)
    matched = [overlaps[r, c] for r, c in zip(rows, cols) if overlaps[r, c] >= iou_threshold]
    return dict(tp=len(matched), fp=len(predicted) - len(matched), fn=len(truth) - len(matched),
                matched_iou_sum=float(sum(matched)), ground_truth_count=len(truth))


def heatmap_data_uri(scores, threshold):
    normalized = np.clip(np.asarray(scores) / max(threshold * 1.5, 1e-8), 0, 1)
    rgb = np.stack([255 * normalized, 200 * (1 - np.abs(2 * normalized - 1)), 210 * (1 - normalized)], axis=-1)
    image = Image.fromarray(rgb.astype(np.uint8)).convert('RGBA')
    image.putalpha(Image.fromarray((normalized * 200).astype(np.uint8)))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
