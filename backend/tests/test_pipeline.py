import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.api import app
from backend.dataset import image_hash, inspect_dataset, validate_manifest
from backend.localization import box_iou, extract_regions, image_metrics, match_boxes
from backend.quantum import QuantumComparison, QuantumKernel


def test_normal_map_returns_no_regions():
    boxes, mask = extract_regions(np.zeros((80, 120)), .5)
    assert boxes == []
    assert not mask.any()


def test_distinct_regions_have_tight_original_coordinates():
    scores = np.zeros((80, 120))
    scores[3:13, 7:17] = .9
    scores[40:47, 90:100] = .8
    boxes, mask = extract_regions(scores, .5)
    assert [(b['x'], b['y'], b['width'], b['height']) for b in boxes] == [(7, 3, 10, 10), (90, 40, 10, 7)]
    assert mask.sum() == 170


def test_tiny_noise_is_not_boxed():
    scores = np.zeros((64, 64)); scores[20, 20] = 1
    assert extract_regions(scores, .5)[0] == []


def test_edge_region_is_not_clipped():
    scores = np.zeros((10, 20)); scores[5:, 12:] = 1
    box = extract_regions(scores, .5)[0][0]
    assert (box['x'], box['y'], box['width'], box['height']) == (12, 5, 8, 5)


def test_invalid_map_is_rejected():
    with pytest.raises(ValueError):
        extract_regions(np.array([[np.nan]]), .5)


def test_box_iou_and_one_to_one_matching():
    box = dict(x=0, y=0, width=10, height=10)
    assert box_iou(box, box) == 1
    assert match_boxes([box, box], [box]) == dict(tp=1, fp=1, fn=0, matched_iou_sum=1., ground_truth_count=1)


def test_unmatched_defects_are_counted():
    result = match_boxes([], [dict(x=0, y=0, width=10, height=10)])
    assert result['fn'] == 1 and result['matched_iou_sum'] == 0


def test_classification_metrics_and_undefined_denominators():
    metrics = image_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert metrics['precision'] == metrics['recall'] == metrics['balanced_accuracy'] == .5
    assert metrics['false_positive_rate'] == metrics['missed_defect_rate'] == .5
    assert image_metrics([0], [0])['recall'] is None


def test_duplicate_content_cannot_cross_splits(tmp_path):
    a, b = tmp_path / 'a.png', tmp_path / 'b.png'
    Image.new('RGB', (32, 32), 'red').save(a)
    Image.new('RGB', (32, 32), 'red').save(b)
    assert image_hash(a) == image_hash(b)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_manifest([dict(path=str(a), split='train', label=0), dict(path=str(b), split='test', label=0)])


def test_changed_manifest_image_is_rejected(tmp_path):
    path = tmp_path / 'image.png'; Image.new('RGB', (16, 16), 'red').save(path)
    row = dict(path=str(path), split='train', label=0, sha256=image_hash(path))
    Image.new('RGB', (16, 16), 'blue').save(path)
    with pytest.raises(ValueError, match='changed'):
        validate_manifest([row])


def make_fixture(root):
    rng = np.random.default_rng(42)
    for folder in ['train/good', 'test/good', 'test/scratch', 'ground_truth/scratch']:
        (root / 'screw' / folder).mkdir(parents=True)
    for index in range(12):
        image = Image.fromarray(rng.integers(0, 255, (32, 48, 3), dtype=np.uint8))
        image.save(root / 'screw' / 'train' / 'good' / f'{index:03}.png')
    Image.fromarray(rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(root / 'screw/test/good/000.png')
    Image.fromarray(rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(root / 'screw/test/scratch/000.png')
    mask = np.zeros((32, 48), dtype=np.uint8); mask[8:15, 12:22] = 255
    Image.fromarray(mask).save(root / 'screw/ground_truth/scratch/000_mask.png')


def test_official_test_split_never_enters_training(tmp_path):
    make_fixture(tmp_path)
    rows = inspect_dataset(tmp_path, 'screw')
    assert {r['split'] for r in rows if '/test/' in r['path']} == {'test'}
    assert all(r['label'] == 0 for r in rows if r['split'] in {'train', 'val'})
    assert rows == inspect_dataset(tmp_path, 'screw')


def test_missing_masks_block_dataset_inspection(tmp_path):
    make_fixture(tmp_path)
    (tmp_path / 'screw/ground_truth/scratch/000_mask.png').unlink()
    with pytest.raises(ValueError, match='Missing ground-truth'):
        inspect_dataset(tmp_path, 'screw')


def test_quantum_fidelity_kernel_is_valid():
    kernel = QuantumKernel()
    values = np.random.default_rng(7).uniform(0, np.pi, (6, 4))
    states = kernel.states(values)
    matrix = kernel.matrix(states, states)
    assert states.shape == (6, 16)
    assert np.allclose(np.diag(matrix), 1)
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > -1e-8


def test_quantum_comparison_really_fits_both_models():
    rng = np.random.default_rng(9)
    model = QuantumComparison()
    report = model.fit(rng.normal(size=(12, 8)), np.tile([0, 1], 6), rng.normal(size=(6, 8)), np.tile([0, 1], 3))
    output = model.predict(rng.normal(size=(2, 8)))
    assert output['quantum'].shape == (2,) and output['classical'].shape == (2,)
    assert len(report['models']['quantum']['tuning']) == len(report['models']['classical']['tuning']) == 3
    assert model.transform[0].n_samples_seen_ == 12


def test_quantum_training_requires_both_classes():
    with pytest.raises(ValueError, match='Both train and validation'):
        QuantumComparison().fit(np.ones((10, 8)), np.zeros(10), np.ones((4, 8)), np.zeros(4))


def test_spatial_feature_shape():
    import torch
    from backend.spatial import SpatialModel
    torch.set_num_threads(2)
    model = SpatialModel(device='cpu', pretrained=False)
    features = model.features(Image.new('RGB', (320, 180)))
    assert tuple(features.shape) == (1, 384, 32, 32)
    assert model.embedding(Image.new('RGB', (120, 80))).shape == (384,)


def test_high_image_score_cannot_create_a_box():
    from backend.spatial import SpatialModel
    model = object.__new__(SpatialModel)
    model.category = 'screw'; model.pixel_threshold = .5; model.image_threshold = .7
    scores = np.zeros((80, 120), dtype=np.float32); scores[10, 10] = 1
    model.score_map = lambda _: scores
    result, mask = model.predict(Image.new('RGB', (120, 80)), include_heatmap=False)
    assert result['score'] == 1 and result['boxes'] == [] and result['status'] == 'NORMAL'
    assert not mask.any()


def test_anomaly_is_review_not_physical_damage_claim():
    from backend.spatial import SpatialModel
    model = object.__new__(SpatialModel)
    model.category = 'screw'; model.pixel_threshold = .5; model.image_threshold = .7
    scores = np.zeros((80, 120), dtype=np.float32); scores[10:20, 15:25] = 1
    model.score_map = lambda _: scores
    result, _ = model.predict(Image.new('RGB', (120, 80)), include_heatmap=False)
    assert result['status'] == 'REVIEW' and len(result['boxes']) == 1


def test_api_health_and_input_validation(monkeypatch, tmp_path):
    import backend.api as api
    monkeypatch.setattr(api, 'ARTIFACTS', tmp_path)
    client = TestClient(app)
    assert client.get('/health').json()['models'] == []
    assert client.post('/inspect', data={'category': '../secret'}, files={'image': ('a.png', b'x', 'image/png')}).status_code == 422
    assert client.post('/inspect', data={'category': 'screw'}, files={'image': ('a.svg', b'x', 'image/svg+xml')}).status_code == 415
    assert client.post('/inspect', data={'category': 'screw'}, files={'image': ('a.png', b'not image data', 'image/png')}).status_code == 422


def test_untrained_api_returns_no_fake_prediction(monkeypatch, tmp_path):
    import backend.api as api
    monkeypatch.setattr(api, 'ARTIFACTS', tmp_path)
    buffer = BytesIO(); Image.new('RGB', (40, 40)).save(buffer, format='PNG')
    response = TestClient(app).post('/inspect', data={'category': 'screw'}, files={'image': ('a.png', buffer.getvalue(), 'image/png')})
    assert response.status_code == 503
    assert 'No trained' in response.json()['detail']
    assert 'boxes' not in response.json()


def test_quantum_positive_does_not_create_localization(monkeypatch, tmp_path):
    import backend.api as api
    monkeypatch.setattr(api, 'ARTIFACTS', tmp_path)
    (tmp_path / 'screw.pt').write_bytes(b'test artifact marker')
    (tmp_path / 'screw-quantum.joblib').write_bytes(b'test artifact marker')
    spatial = SimpleNamespace(predict=lambda image: (dict(boxes=[], status='NORMAL', quantum=None, warning='Research only'), None), embedding=lambda image: np.zeros(384))
    quantum = SimpleNamespace(predict=lambda features: dict(quantum=np.array([1]), classical=np.array([0]), quantum_margin=np.array([.8])))
    monkeypatch.setattr(api, 'spatial_model', lambda *args: spatial)
    monkeypatch.setattr(api, 'quantum_model', lambda *args: quantum)
    result = api.infer(Image.new('RGB', (40, 40)), 'screw', True)
    assert result['boxes'] == [] and result['status'] == 'NORMAL'
    assert result['quantum']['prediction'] == 'DEFECTIVE'
    assert result['quantum']['classical_prediction'] == 'NORMAL'


def test_training_and_evaluation_fixture_smoke(tmp_path, monkeypatch):
    import torch
    from backend.spatial import SpatialModel, train
    from backend.evaluate import evaluate
    torch.set_num_threads(2)
    make_fixture(tmp_path)
    rows = inspect_dataset(tmp_path, 'screw')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'images': rows}))
    original_init = SpatialModel.__init__
    monkeypatch.setattr(SpatialModel, '__init__', lambda self, device=None, pretrained=True: original_init(self, device, pretrained=False))
    artifacts = tmp_path / 'artifacts'
    train(manifest, 'screw', artifacts, device='cpu', memory_size=128)
    report_path = tmp_path / 'report.json'
    evaluate(SimpleNamespace(manifest=manifest, category='screw', artifacts=artifacts, device='cpu', output=report_path))
    report = json.loads(report_path.read_text())
    assert report['test_images'] == 2
    assert report['quantum_comparison'] is None
    assert report['classification']['tp'] + report['classification']['fn'] == 1
    assert report['localization']['box_counts']['ground_truth_count'] == 1
    from backend.research import workspace_report
    (artifacts / 'screw-evaluation.json').write_text(json.dumps(report))
    workspace = workspace_report(tmp_path, artifacts, 'screw')
    assert 0 < workspace['training']['memory_patches'] <= 128
    assert workspace['evaluation']['test_images'] == 2
    assert workspace['warnings'] == []
