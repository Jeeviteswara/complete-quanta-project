import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.artifacts import artifact_hash, validate_comparison
from backend.dataset import image_hash, validate_manifest
from backend.pipeline import run_pipeline
from backend.tests.test_pipeline import make_fixture


def test_changed_mask_is_rejected(tmp_path):
    image, mask = tmp_path / 'image.png', tmp_path / 'mask.png'
    Image.new('RGB', (16, 16), 'red').save(image)
    Image.new('L', (16, 16), 255).save(mask)
    row = dict(path=str(image), mask=str(mask), label=1, split='test')
    validate_manifest([row], require_test_masks=True)
    assert row['mask_sha256'] == image_hash(mask)
    Image.new('L', (16, 16), 128).save(mask)
    with pytest.raises(ValueError, match='Mask changed'):
        validate_manifest([row], require_test_masks=True)


def test_missing_test_mask_is_rejected_but_classification_training_allows_it(tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGB', (16, 16), 'red').save(path)
    row = dict(path=str(path), mask=None, label=1, split='test')
    with pytest.raises(ValueError, match='ground-truth mask'):
        validate_manifest([row], require_test_masks=True)
    row['split'] = 'train'
    assert validate_manifest([row])[0]['label'] == 1


def test_normal_image_cannot_have_positive_mask(tmp_path):
    path, mask = tmp_path / 'image.png', tmp_path / 'mask.png'
    Image.new('RGB', (16, 16), 'red').save(path)
    Image.new('L', (16, 16), 255).save(mask)
    with pytest.raises(ValueError, match='Normal image'):
        validate_manifest([dict(path=str(path), mask=str(mask), label=0, split='test')])


def test_normal_image_can_have_empty_mask(tmp_path):
    path, mask = tmp_path / 'image.png', tmp_path / 'mask.png'
    Image.new('RGB', (16, 16), 'red').save(path)
    Image.new('L', (16, 16), 0).save(mask)
    assert validate_manifest([dict(path=str(path), mask=str(mask), label=0, split='test')])


def test_mask_dimensions_use_exif_normalized_coordinates(tmp_path):
    path, mask = tmp_path / 'image.jpg', tmp_path / 'mask.png'
    exif = Image.Exif(); exif[274] = 6
    Image.new('RGB', (16, 32), 'red').save(path, exif=exif)
    Image.new('L', (32, 16), 255).save(mask)
    assert validate_manifest([dict(path=str(path), mask=str(mask), label=1, split='test')])


@pytest.mark.parametrize('category,digest', [('screw', 'old'), ('bottle', 'current'), (None, None)])
def test_stale_wrong_category_and_legacy_comparisons_are_rejected(category, digest):
    spatial = SimpleNamespace(category='screw', artifact_sha256='current')
    with pytest.raises(ValueError, match='Retrain'):
        validate_comparison(SimpleNamespace(category=category, spatial_sha256=digest), spatial)


def test_compatible_comparison_is_accepted(tmp_path):
    path = tmp_path / 'model.pt'; path.write_bytes(b'locally trained artifact')
    digest = artifact_hash(path)
    assert len(digest) == 64
    validate_comparison(SimpleNamespace(category='screw', spatial_sha256=digest),
                        SimpleNamespace(category='screw', artifact_sha256=digest))


@pytest.mark.parametrize('failure', ['stale', 'corrupt'])
def test_optional_quantum_failure_preserves_spatial_evidence(monkeypatch, tmp_path, failure):
    import backend.api as api
    monkeypatch.setattr(api, 'ARTIFACTS', tmp_path)
    (tmp_path / 'screw.pt').write_bytes(b'artifact')
    (tmp_path / 'screw-quantum.joblib').write_bytes(b'artifact')
    box = dict(id=1, x=2, y=3, width=10, height=10)
    spatial = SimpleNamespace(category='screw', artifact_sha256='current',
                              predict=lambda image: (dict(boxes=[box], status='REVIEW', quantum=None, warning='Research only.'), None),
                              embedding=lambda image: np.zeros(384))
    monkeypatch.setattr(api, 'spatial_model', lambda *args: spatial)
    def load_quantum(*args):
        if failure == 'corrupt':
            raise ValueError('Corrupt artifact')
        return SimpleNamespace(category='screw', spatial_sha256='old')
    monkeypatch.setattr(api, 'quantum_model', load_quantum)
    result = api.infer(Image.new('RGB', (40, 40)), 'screw', True)
    assert result['status'] == 'REVIEW' and result['boxes'] == [box]
    assert result['quantum'] is None and 'spatial results are unchanged' in result['warning']
    assert not api.INFERENCE_LOCK.locked()


def pipeline_args(tmp_path, **changes):
    values = dict(root=tmp_path / 'mvtec', category='screw', data=tmp_path / 'data', artifacts=tmp_path / 'artifacts',
                  labeled_manifest=None, device='cpu', seed=42, memory_size=128, overwrite=False)
    values.update(changes)
    return SimpleNamespace(**values)


def test_runner_requires_real_data_without_creating_artifacts(tmp_path):
    args = pipeline_args(tmp_path)
    with pytest.raises(ValueError, match='Dataset has not been provided'):
        run_pipeline(args)
    assert not args.artifacts.exists()


def test_runner_does_not_overwrite_existing_outputs(tmp_path):
    args = pipeline_args(tmp_path)
    args.artifacts.mkdir()
    path = args.artifacts / 'screw.pt'; path.write_bytes(b'previous model')
    with pytest.raises(ValueError, match='already exist'):
        run_pipeline(args)
    assert path.read_bytes() == b'previous model'


def test_runner_refuses_to_leave_stale_quantum_model(tmp_path):
    args = pipeline_args(tmp_path, overwrite=True)
    args.artifacts.mkdir()
    (args.artifacts / 'screw-quantum.joblib').write_bytes(b'previous comparison')
    with pytest.raises(ValueError, match='would become stale'):
        run_pipeline(args)


def test_failed_training_keeps_previous_experiment(monkeypatch, tmp_path):
    import backend.spatial as spatial
    args = pipeline_args(tmp_path, overwrite=True)
    make_fixture(args.root)
    args.artifacts.mkdir()
    path = args.artifacts / 'screw.pt'; path.write_bytes(b'previous model')
    def fail(*args):
        raise RuntimeError('Training interrupted')
    monkeypatch.setattr(spatial, 'train', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        run_pipeline(args)
    assert path.read_bytes() == b'previous model'
    assert not (args.data / 'screw-manifest.json').exists()
    assert not list(args.artifacts.glob('.quanta-run-*'))


@pytest.mark.parametrize('with_quantum', [False, True])
def test_full_runner_and_category_guard_with_random_fixture(monkeypatch, tmp_path, with_quantum):
    import torch
    from backend.spatial import SpatialModel
    from backend.research import workspace_report
    torch.set_num_threads(2)
    args = pipeline_args(tmp_path)
    make_fixture(args.root)
    if with_quantum:
        rng = np.random.default_rng(123)
        labeled = []
        for index in range(12):
            path = tmp_path / f'independent-{index}.png'
            Image.fromarray(rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(path)
            labeled.append(dict(path=str(path), category='screw', split='train' if index < 8 else 'val',
                                label=index % 2, group=f'part-{index}', mask=None))
        args.labeled_manifest = tmp_path / 'labeled.json'
        args.labeled_manifest.write_text(json.dumps(dict(images=labeled)))
    original_init = SpatialModel.__init__
    monkeypatch.setattr(SpatialModel, '__init__', lambda self, device=None, pretrained=True: original_init(self, device, pretrained=False))
    report_path = run_pipeline(args)
    report = json.loads(report_path.read_text())
    assert report['test_images'] == 2
    assert (report['quantum_comparison'] is not None) == with_quantum
    if with_quantum:
        import joblib
        comparison = joblib.load(args.artifacts / 'screw-quantum.joblib')
        assert comparison.spatial_sha256 == artifact_hash(args.artifacts / 'screw.pt')
        assert comparison.category == 'screw'
        assert report['quantum_comparison']['quantum']['tp'] + report['quantum_comparison']['quantum']['fn'] == 1
    workspace = workspace_report(args.data, args.artifacts, 'screw')
    assert workspace['dataset']['total'] == 14 and not workspace['evaluation_stale']
    assert workspace['warnings'] == []
    with pytest.raises(ValueError, match='category'):
        SpatialModel.load(args.artifacts / 'screw.pt', device='cpu', expected_category='bottle')
    assert not list(args.artifacts.glob('.quanta-run-*'))
