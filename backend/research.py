import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.dataset import CATEGORIES

MAX_REPORT_BYTES = 32 * 1024 * 1024
Rate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, strict=True)]


class ReportModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class Classification(ReportModel):
    precision: Rate | None
    recall: Rate | None
    f1: Rate | None
    balanced_accuracy: Rate | None
    false_positive_rate: Rate | None
    missed_defect_rate: Rate | None
    tp: Count
    tn: Count
    fp: Count
    fn: Count


class BoxCounts(ReportModel):
    tp: Count
    fp: Count
    fn: Count
    ground_truth_count: Count
    matched_iou_sum: Annotated[float, Field(ge=0)]


class Localization(ReportModel):
    pixel_iou: Rate | None
    pixel_dice: Rate | None
    defective_image_macro_iou: Rate | None
    defective_image_macro_dice: Rate | None
    box_iou_with_unmatched_truth_as_zero: Rate | None
    box_precision: Rate | None
    box_recall: Rate | None
    box_matching_iou_threshold: Rate
    box_counts: BoxCounts


class Comparison(ReportModel):
    quantum: Classification
    classical: Classification
    combined_inference_seconds: Annotated[float, Field(ge=0)]
    protocol: str


class Evaluation(ReportModel):
    status: Literal['evaluated']
    category: str
    test_images: Annotated[int, Field(gt=0, strict=True)]
    elapsed_seconds: Annotated[float, Field(ge=0)]
    classification: Classification
    localization: Localization
    quantum_comparison: Comparison | None
    limitations: list[str]


class Training(ReportModel):
    status: Literal['trained_not_tested']
    category: str
    seed: int
    training_images: Count
    validation_images: Count
    memory_patches: Count
    device: str
    pixel_threshold: Annotated[float, Field(ge=0)]
    image_threshold: Annotated[float, Field(ge=0)]
    threshold_protocol: str
    physical_defect_detection_validated: Literal[False]


def read_document(path):
    # Bound local report reads and never deserialize model artifacts for the dashboard.
    with path.open('rb') as source:
        content = source.read(MAX_REPORT_BYTES + 1)
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError('Report exceeds size limit.')
    document = json.loads(content)
    if not isinstance(document, dict):
        raise ValueError('Report must be an object.')
    return document


def dataset_summary(path, category):
    document = read_document(path)
    if document.get('category') != category:
        raise ValueError('Manifest category mismatch.')
    rows = document.get('images')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest has no images.')
    counts = dict(train=0, val=0, test=0)
    normal = defective = masks = 0
    for row in rows:
        if not isinstance(row, dict) or row.get('category') != category or not isinstance(row.get('split'), str) or row['split'] not in counts or type(row.get('label')) is not int or row['label'] not in (0, 1):
            raise ValueError('Invalid manifest row.')
        if row['split'] != 'test' and row['label'] != 0:
            raise ValueError('Spatial train and validation must be normal-only.')
        counts[row['split']] += 1
        normal += row['label'] == 0
        defective += row['label'] == 1
        masks += bool(row.get('mask'))
    seed = document.get('seed')
    if type(seed) is not int:
        raise ValueError('Manifest seed must be an integer.')
    return dict(counts=counts, total=len(rows), normal=normal, defective=defective, masks=masks, seed=seed,
                note='Recorded manifest counts. Source files and hashes are revalidated by training and evaluation, not by this dashboard.')


def workspace_report(data: Path, artifacts: Path, category: str):
    if category not in CATEGORIES:
        raise ValueError('Unknown category.')
    warnings = []
    dataset = training = evaluation = None
    paths = {
        'dataset': data / f'{category}-manifest.json',
        'training': artifacts / f'{category}-training.json',
        'evaluation': artifacts / f'{category}-evaluation.json',
    }
    for name, path in paths.items():
        if not path.is_file():
            continue
        try:
            if name == 'dataset':
                dataset = dataset_summary(path, category)
            else:
                model = Training if name == 'training' else Evaluation
                report = model.model_validate(read_document(path))
                if report.category != category:
                    raise ValueError('Report category mismatch.')
                if name == 'training':
                    training = report.model_dump()
                else:
                    measurements = [report.classification]
                    if report.quantum_comparison:
                        measurements.extend([report.quantum_comparison.quantum, report.quantum_comparison.classical])
                    if any(counts.tp + counts.tn + counts.fp + counts.fn != report.test_images for counts in measurements):
                        raise ValueError('Evaluation counts do not match test size.')
                    # Only aggregate metrics are exposed; per-image records contain local paths.
                    evaluation = report.model_dump()
        except (OSError, ValueError, ValidationError):
            warnings.append(f'The {name} report could not be read or validated. Regenerate {path.name} with the included pipeline.')
    stale = False
    if evaluation:
        evaluated_at = paths['evaluation'].stat().st_mtime_ns
        stale = any(path.is_file() and path.stat().st_mtime_ns > evaluated_at for path in [
            paths['dataset'], paths['training'], artifacts / f'{category}.pt', artifacts / f'{category}-quantum.joblib',
        ])
        if stale:
            warnings.append('The evaluation predates a current dataset or model artifact. Rerun evaluation before interpreting these metrics as current.')
    return dict(category=category, dataset=dataset, training=training, evaluation=evaluation,
                evaluation_stale=stale, warnings=warnings,
                categories=[dict(category=item,
                                 manifest_available=(data / f'{item}-manifest.json').is_file(),
                                 spatial_available=(artifacts / f'{item}.pt').is_file(),
                                 quantum_available=(artifacts / f'{item}-quantum.joblib').is_file(),
                                 evaluation_available=(artifacts / f'{item}-evaluation.json').is_file()) for item in CATEGORIES])
