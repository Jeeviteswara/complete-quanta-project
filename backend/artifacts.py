import hashlib
from pathlib import Path


def artifact_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_comparison(comparison, spatial):
    if getattr(comparison, 'category', None) != spatial.category:
        raise ValueError('Quantum comparison category is missing or does not match. Retrain the comparison.')
    expected = getattr(comparison, 'spatial_sha256', None)
    if not expected or expected != getattr(spatial, 'artifact_sha256', None):
        raise ValueError('Quantum comparison was not trained against this spatial artifact. Retrain the comparison.')
