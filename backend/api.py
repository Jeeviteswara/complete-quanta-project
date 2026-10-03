import logging
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend.artifacts import validate_comparison
from backend.dataset import CATEGORIES

app = FastAPI(title='QuantumInspect', version='0.1.0', description='Local research inference. No fabricated predictions or physical-defect guarantees.')
ARTIFACTS = Path(__file__).resolve().parent.parent / 'artifacts'
DATA = Path(__file__).resolve().parent.parent / 'data'
MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000
INFERENCE_LOCK = Lock()
logger = logging.getLogger('quantuminspect')


@lru_cache(maxsize=1)
def spatial_model(category, modified):
    from backend.spatial import SpatialModel
    return SpatialModel.load(ARTIFACTS / f'{category}.pt', expected_category=category)


@lru_cache(maxsize=1)
def quantum_model(category, modified):
    # Artifacts are locally produced, trusted Python objects; never accept uploaded pickles.
    return joblib.load(ARTIFACTS / f'{category}-quantum.joblib')


@app.get('/health')
def health():
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'PyTorch not installed'
    models = [category for category in CATEGORIES if (ARTIFACTS / f'{category}.pt').is_file()]
    quantum = [category for category in CATEGORIES if (ARTIFACTS / f'{category}-quantum.joblib').is_file()]
    return dict(connected=True, models=models, quantum_models=quantum, device=device,
                message='Inference artifacts available.' if models else 'Train category models before running an inspection.')


@app.get('/research')
def research(category: str = 'screw'):
    from backend.research import workspace_report
    if category not in CATEGORIES:
        raise HTTPException(422, 'Select a valid MVTec product category.')
    return workspace_report(DATA, ARTIFACTS, category)


def infer(image, category, include_quantum):
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(429, 'Another inspection is in progress. Try again shortly.')
    try:
        started = time.perf_counter()
        path = ARTIFACTS / f'{category}.pt'
        model = spatial_model(category, path.stat().st_mtime_ns)
        result, _ = model.predict(image)
        q_path = ARTIFACTS / f'{category}-quantum.joblib'
        if include_quantum and q_path.is_file():
            try:
                comparison = quantum_model(category, q_path.stat().st_mtime_ns)
                validate_comparison(comparison, model)
                output = comparison.predict(model.embedding(image)[None])
                result['quantum'] = dict(prediction='DEFECTIVE' if int(output['quantum'][0]) else 'NORMAL', classical_prediction='DEFECTIVE' if int(output['classical'][0]) else 'NORMAL', margin=float(output['quantum_margin'][0]))
            except Exception:
                logger.exception('Optional quantum comparison failed; preserving spatial evidence')
                result['quantum'] = None
                result['warning'] += ' Quantum comparison unavailable or incompatible. Retrain it against the current spatial model; spatial results are unchanged.'
        elif include_quantum:
            result['warning'] += ' Quantum comparison not trained for this category.'
        result['elapsed_ms'] = (time.perf_counter() - started) * 1000
        return result
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('Local inference failed')
        raise HTTPException(503, 'The local model could not run. Check its artifact and Python dependencies.') from error
    finally:
        INFERENCE_LOCK.release()


@app.post('/inspect')
async def inspect(image: UploadFile = File(...), category: str = Form(...), quantum: bool = Form(True)):
    if category not in CATEGORIES:
        raise HTTPException(422, 'Select a valid MVTec product category.')
    if image.content_type not in {'image/png', 'image/jpeg', 'image/webp'}:
        raise HTTPException(415, 'Only PNG, JPEG, and WebP images are accepted.')
    content = await image.read(MAX_BYTES + 1)
    await image.close()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, 'Image exceeds the 10 MB limit.')
    try:
        with Image.open(BytesIO(content)) as source:
            if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise HTTPException(415, 'Unsupported image encoding.')
            if source.width * source.height > 25_000_000:
                raise HTTPException(413, 'Image exceeds the 25 megapixel limit.')
            decoded = ImageOps.exif_transpose(source).convert('RGB')
            decoded.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning, ValueError) as error:
        raise HTTPException(422, 'The image is corrupt, invalid, or too large.') from error
    if not (ARTIFACTS / f'{category}.pt').is_file():
        raise HTTPException(503, f'No trained {category} model is available. Train the local spatial model first; no prediction or boxes were generated.')
    return await run_in_threadpool(infer, decoded, category, quantum)
