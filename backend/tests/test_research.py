import json
import os

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.research import workspace_report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def measured_report():
    return dict(
        status='evaluated', category='screw', test_images=2, elapsed_seconds=1.5,
        classification=dict(precision=None, recall=0., f1=0., balanced_accuracy=.5,
                            false_positive_rate=0., missed_defect_rate=1., tp=0, tn=1, fp=0, fn=1),
        localization=dict(pixel_iou=0., pixel_dice=0., defective_image_macro_iou=0.,
                          defective_image_macro_dice=0., box_iou_with_unmatched_truth_as_zero=0.,
                          box_precision=None, box_recall=0., box_matching_iou_threshold=.5,
                          box_counts=dict(tp=0, fp=0, fn=1, ground_truth_count=1, matched_iou_sum=0.)),
        quantum_comparison=None, limitations=['Synthetic fixture; not a product accuracy claim.'],
        images=[dict(path='/private/workstation/image.png')],
    )


def test_empty_workspace_has_no_invented_reports(tmp_path):
    report = workspace_report(tmp_path / 'data', tmp_path / 'artifacts', 'screw')
    assert report['dataset'] is report['training'] is report['evaluation'] is None
    assert len(report['categories']) == 15
    assert not any(row['spatial_available'] for row in report['categories'])
    assert report['warnings'] == []


def test_dataset_summary_counts_records_without_exposing_paths(tmp_path):
    rows = [dict(category='screw', split=split, label=label, path='/private/image.png', mask=mask)
            for split, label, mask in [('train', 0, None), ('val', 0, None), ('test', 1, '/private/mask.png')]]
    write_json(tmp_path / 'data/screw-manifest.json', dict(category='screw', seed=42, images=rows))
    report = workspace_report(tmp_path / 'data', tmp_path / 'artifacts', 'screw')
    assert report['dataset']['counts'] == dict(train=1, val=1, test=1)
    assert report['dataset']['masks'] == 1
    assert report['dataset']['defective'] == 1
    assert '/private/' not in json.dumps(report)


def test_evaluation_preserves_undefined_and_zero_metrics(tmp_path):
    write_json(tmp_path / 'artifacts/screw-evaluation.json', measured_report())
    report = workspace_report(tmp_path / 'data', tmp_path / 'artifacts', 'screw')
    assert report['evaluation']['classification']['precision'] is None
    assert report['evaluation']['classification']['recall'] == 0
    assert 'images' not in report['evaluation']
    assert not report['evaluation_stale']
    assert '/private/' not in json.dumps(report)


@pytest.mark.parametrize('change', [
    {'category': 'bottle'}, {'test_images': 3}, {'test_images': 0},
    {'classification': {}}, {'elapsed_seconds': float('nan')},
])
def test_invalid_evaluation_is_not_displayed(tmp_path, change):
    document = measured_report()
    document.update(change)
    write_json(tmp_path / 'artifacts/screw-evaluation.json', document)
    report = workspace_report(tmp_path / 'data', tmp_path / 'artifacts', 'screw')
    assert report['evaluation'] is None
    assert len(report['warnings']) == 1


def test_changed_artifact_marks_evaluation_historical(tmp_path):
    report_path = tmp_path / 'artifacts/screw-evaluation.json'
    write_json(report_path, measured_report())
    artifact = tmp_path / 'artifacts/screw.pt'
    artifact.write_bytes(b'not deserialized by research endpoint')
    os.utime(report_path, ns=(1_000_000_000, 1_000_000_000))
    report = workspace_report(tmp_path / 'data', tmp_path / 'artifacts', 'screw')
    assert report['evaluation_stale']
    assert report['evaluation'] is not None
    assert report['categories'][9]['spatial_available']


def test_malformed_manifest_does_not_hide_other_reports(tmp_path):
    write_json(tmp_path / 'data/screw-manifest.json', dict(category='screw', seed=42, images=[dict(category='screw', split='train', label=1)]))
    write_json(tmp_path / 'artifacts/screw-evaluation.json', measured_report())
    report = workspace_report(tmp_path / 'data', tmp_path / 'artifacts', 'screw')
    assert report['dataset'] is None
    assert report['evaluation'] is not None
    assert 'dataset' in report['warnings'][0]


def test_report_size_is_bounded(tmp_path, monkeypatch):
    import backend.research as research
    monkeypatch.setattr(research, 'MAX_REPORT_BYTES', 16)
    write_json(tmp_path / 'artifacts/screw-evaluation.json', measured_report())
    assert workspace_report(tmp_path / 'data', tmp_path / 'artifacts', 'screw')['evaluation'] is None


def test_research_api_rejects_paths_and_is_category_scoped(tmp_path, monkeypatch):
    import backend.api as api
    monkeypatch.setattr(api, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(api, 'ARTIFACTS', tmp_path / 'artifacts')
    write_json(tmp_path / 'artifacts/screw-evaluation.json', measured_report())
    client = TestClient(app)
    assert client.get('/research', params={'category': '../../etc/passwd'}).status_code == 422
    response = client.get('/research', params={'category': 'screw'})
    assert response.status_code == 200
    assert response.json()['evaluation']['test_images'] == 2
    assert client.get('/research', params={'category': 'bottle'}).json()['evaluation'] is None
