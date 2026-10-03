# QuantumInspect

An evidence-first manufacturing inspection research workspace with a Next.js/React/TypeScript frontend, a local Python/FastAPI service, a PyTorch spatial anomaly baseline, and a real Qiskit quantum-kernel classifier.

## Current status — read this first

**This is a runnable research implementation, not a validated defect detector.** No MVTec dataset was supplied to the development workspace. There is no trained MVTec artifact, accuracy figure, or demonstrated quantum advantage. The initial bolt is an AI-generated illustration with manually placed boxes, explicitly labeled in the UI. Its boxes are not algorithm outputs and must not be used for evaluation.

Unsupervised anomaly detection cannot establish that an unusual region is a scratch rather than a reflection. The baseline returns **REVIEW** for retained anomaly regions, and **NORMAL** for no retained regions. NORMAL is not a quality certification. It never asserts physical damage based solely on an image-level or quantum prediction. Ground-truth labels are used only by evaluation, never by inference.

The Evaluation workspace displays actual aggregate metrics from locally generated category reports. No bundled evaluation report or measured MVTec accuracy is supplied.

## Architecture

- `components/`: responsive inspection UI, upload, category selection, original-image/box/heatmap views, region coordinates, JSON export, research panels.
- `app/api/`: same-origin Next.js proxy for the local Python service.
- `backend/dataset.py`: file/mask inspection, canonical image hashes, seeded normal train/validation split, official test preservation.
- `backend/spatial.py`: frozen ImageNet ResNet-18 layer-2/layer-3 features, normal patch memory, nearest-neighbor scoring, validation-only threshold calibration.
- `backend/localization.py`: connected components, original-coordinate boxes, mask metrics, one-to-one box matching.
- `backend/quantum.py`: genuine Qiskit ZZ feature-map states and fidelity-kernel SVM; matched classical RBF SVM.
- `backend/train_quantum.py`: independent labeled data, train-only PCA/scaling, validation-only model selection.
- `backend/evaluate.py`: held-out classification and localization reports, optional paired quantum/classical evaluation.
- `backend/api.py`: local inference, health, and research endpoints; fails explicitly when artifacts are missing.
- `backend/research.py`: bounded, validated JSON report reading for the live dataset, model, and evaluation screens. Returns aggregate summaries only, without per-image local paths.
- `backend/tests/`: unit tests plus a random-image pipeline smoke test. These are not MVTec experiments.

DINOv2 was replaced with the smaller ResNet-18 baseline. PyTorch and the quantum model are retained. This baseline is deliberately simple, not a claim to reproduce PatchCore's full published method.

## Windows 11 installation

Obtain the project through the connected GitHub repository or the v0 shadcn installation command. The following commands are for **your Windows workstation**, not the v0 browser preview.

Prerequisites: Python 3.11 or 3.12, Node.js 22+, pnpm matching `package.json`, and a current NVIDIA driver. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to select a Windows/Pip/Python/CUDA build compatible with your driver. An 8 GB RTX 4060 Laptop GPU is the target, not a benchmarked guarantee.

From the project directory, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

Install `torch` and `torchvision` using the CUDA command from the PyTorch selector, substituting `.\.venv\Scripts\python.exe -m pip` for `pip`. For example, **only if your driver supports the selected CUDA 12.8 runtime**:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For a CPU-only installation, use `https://download.pytorch.org/whl/cpu` instead. GPU training must report `True` in the CUDA check; do not assume installing a CUDA toolkit alone enables PyTorch GPU support.

```powershell
pnpm install --frozen-lockfile
```

Start the local Python service in one PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Start the frontend in another:

```powershell
pnpm dev
```

Open `http://localhost:3000`. API documentation is at `http://127.0.0.1:8000/docs`.

The API runs without model artifacts and truthfully reports that training is required. Uploads are kept only in request memory; inspection history and accounts are not implemented. Do not expose this unauthenticated local research service publicly.

## Stage 1: inspect MVTec AD

Download [MVTec AD from its official provider](https://www.mvtec.com/company/research/datasets/mvtec-ad), respecting **CC BY-NC-SA 4.0**. Commercial use needs appropriate licensing. This project does not redistribute the dataset.

Expected layout:

```text
data/mvtec_ad/screw/
  train/good/*.png
  test/good/*.png
  test/<defect_type>/*.png
  ground_truth/<defect_type>/*_mask.png
```

```powershell
.\.venv\Scripts\python.exe -m backend.dataset --root data/mvtec_ad --category screw --output data/screw-manifest.json --seed 42
```

The inspector checks image decoding, missing/empty masks, mask/image dimensions, and cross-split duplicate decoded-pixel hashes. Source-identical normal images stay together. About 20% of the official normal training sources become validation sources. The official test split remains untouched. Fitting and evaluation revalidate hashes to reject changed images.

Hashes cannot detect every semantically duplicated crop or transformed variant. Source/lot provenance must also be audited manually. If you generate synthetic variants, split source groups **before** generating them. A training-data synthetic generator is not included; the UI illustration is not a training sample.

## Stage 2: train the spatial baseline

```powershell
.\.venv\Scripts\python.exe -m backend.spatial --manifest data/screw-manifest.json --category screw --output artifacts --device cuda --memory-size 8192 --seed 42
```

The initial run downloads official pretrained ImageNet ResNet-18 weights. Inference uses the extractor stored in the locally produced artifact and needs no weight download.

Configuration: 256×256 model input, one image at a time, frozen features, 384-dimensional 32×32 patch maps, capped 8,192-patch bank, 128-query distance chunks, FP32, no data-loader workers. Thresholds use only normal validation: pixel quantile 0.995, and the 0.95 quantile of per-image maximum scores with the `higher` estimator. This is not a formal false-positive-rate guarantee, especially with few validation images. Real operating points need larger independent calibration data.

Scores are resized directly to the decoded original image dimensions. No center-crop transform is used. Connected components above the fixed pixel threshold produce boxes; components under 32 original pixels are discarded. An image score below the validation-calibrated image threshold suppresses all candidate regions. A high image score cannot create a missing region. Two touching anomalies can merge; high-resolution tiling or supervised masks may be necessary for finer damage.

Artifacts: `artifacts/screw.pt` and `artifacts/screw-training.json`. Train a separate artifact for each category. Restart the Python service after code changes; model artifacts are reloaded when their modification times change.

## Stage 3: train the quantum comparison

**MVTec's official training images are all normal.** Do not transfer its defective test images to training. Supply independently collected labeled normal/defective images and independent source-group validation data. Real lighting/reflection hard negatives are important. Suitable supervised defect masks were not provided, so supervised segmentation training is not implemented in this baseline.

Use a JSON manifest with an `images` array. Every row needs `path` (absolute), `category`, `label` (0 normal or 1 defective), `split` (`train` or `val`), `group` (source/physical-part identifier), and optional `mask`. For example, create your own records in this shape, pointing to real files:

```json
{
  "images": [
    {"path":"C:/inspection/labeled/normal-a.png","category":"screw","label":0,"split":"train","group":"part-a","mask":null},
    {"path":"C:/inspection/labeled/damaged-b.png","category":"screw","label":1,"split":"train","group":"part-b","mask":null}
  ]
}
```

This two-row illustration is **not sufficient training data**. The script requires at least eight training samples and both classes in train and validation. For this local experiment, supply a predeclared stratified subset of at most 256 training examples to **both** classifiers. All crops and variants of a physical source must share its group and split. Never derive labeled training images from official test images, including transformed variants that evade exact hashing.

```powershell
.\.venv\Scripts\python.exe -m backend.train_quantum --mvtec-manifest data/screw-manifest.json --labeled-manifest data/labeled-screw.json --category screw --artifacts artifacts --device cuda --seed 42
```

Both classifiers share image embeddings, fitted train-only StandardScaler → PCA(4) → angle scaling, train/validation examples, and the same three C values (0.1, 1, 10). C is selected using validation balanced accuracy; neither is refitted on validation. Classical RBF gamma is fixed to `scale`; the quantum circuit is fixed to four qubits, two ZZ repetitions, linear entanglement. These fixed architecture choices must be declared before examining test results.

The quantum fidelity kernel is built with Qiskit's actual `Statevector` simulation and trains a precomputed-kernel SVM. This is **quantum-circuit simulation on a classical CPU**, not quantum hardware or a demonstrated computational advantage. Neither classifier creates localization boxes. No calibrated probabilities are produced: decision margins and anomaly distances are scores, not confidence percentages.

Artifacts: `screw-quantum.joblib` and `screw-quantum-validation.json`. Joblib artifacts execute Python during deserialization: only load artifacts you produced and trust. No API accepts artifact uploads.

## Stage 4: evaluate on the untouched test split

```powershell
.\.venv\Scripts\python.exe -m backend.evaluate --manifest data/screw-manifest.json --category screw --artifacts artifacts --output artifacts/screw-evaluation.json --device cuda
```

The report measures:

- Region-backed image classification: precision, recall, F1, balanced accuracy, false-positive rate, missed-defect rate, confusion counts.
- Pixel localization: aggregate IoU/Dice plus defective-image macro IoU/Dice.
- Boxes: axis-aligned ground-truth component boxes; one-to-one maximum-cardinality matching at IoU ≥ 0.5, then IoU maximization; box precision/recall and mean matched IoU over **all** ground-truth boxes (unmatched boxes count as zero).
- Optional quantum/classical image classification, on the **same complete** official test set.
- Per-image predictions for manual localization review.

Undefined denominators are `null`, not misleading perfect or zero scores. REVIEW with a retained region counts as positive for anomaly benchmark scoring. This does not establish physical damage discrimination. Keep test thresholds frozen; do not repeatedly optimize against the test report. Preregister repeated seeds, use paired confidence intervals, and collect a fresh final holdout if iteration has exposed the test set. The current report generator does not calculate confidence intervals.

Manually review false positives, missed regions, and all lighting/reflection-related cases. Additional supervised segmentation requires separate defect masks and real harmless-artifact masks in training/validation; never use the official test masks for supervision.

## API contract

- `GET /health`: connected status, category artifact availability, quantum artifact availability, compute device.
- `GET /research?category=screw`: reads `data/screw-manifest.json`, `artifacts/screw-training.json`, and `artifacts/screw-evaluation.json`. Other categories use the same filename convention. Invalid reports produce explicit warnings; missing reports return null, not invented metrics. The dashboard refreshes every 30 seconds or on demand. Manifest counts are recorded metadata, not live image-integrity checks. Evaluations older than the current model/manifest are marked historical using file modification times; this is a useful warning, not cryptographic provenance.
- `POST /inspect`: multipart `image`, `category`, optional boolean `quantum`; returns original width/height, status, independent region boxes, raw anomaly score, threshold, heatmap PNG data URI, duration, optional quantum/classical image decisions, and limitations.
- Coordinates are decoded, EXIF-normalized original-image pixel coordinates; width/height are exclusive extents.
- Missing artifacts: 503, with no simulated prediction. Invalid category/data: 422; unsupported format: 415; oversize: 413; concurrent inference: 429.
- Next.js exposes these as `/api/status`, `/api/research`, and `/api/inspect` and enforces its multipart size limit before decoding. Inspection is enabled only when the selected category has a spatial artifact. Cancel stops the browser request; Python inference may finish in the background and hold the inference lock until completion.

Uploads are limited to PNG/JPEG/WebP, 10 MB and 25 megapixels. No user-specified backend URL or path is accepted. The default Python service binds only to loopback; public hosting needs authentication, quotas, transport security, and a separately secured inference service. The Vercel-hosted frontend does **not** include a GPU runtime; its loopback proxy must be replaced with an authenticated hosted service connection before real hosted inference is possible. Previewing or publishing the UI alone does not train a model.

## Automated checks

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
pnpm exec tsc --noEmit
pnpm install --frozen-lockfile
pnpm build
```

Tests cover empty maps, distinct/tiny/edge regions, original-coordinate boxes, matching, metrics, split hashes, missing masks, Qiskit kernel properties, actual classifier fitting on explicitly synthetic numeric fixtures, PyTorch feature shapes, fail-closed API behavior, and random-image training/evaluation plumbing. Fixture test metrics are intentionally not used as product accuracy claims.

## What is still required for real physical-defect detection

1. A licensed MVTec dataset and independently labeled real training/validation damage examples.
2. Actual category training, held-out evaluation, and human localization review.
3. Hard negatives covering shadows, specular highlights, texture, harmless machining marks, and changing illumination.
4. Separately annotated real defect segmentation data for supervised refinement; microscopic defects may need larger inputs/tiling.
5. Target-hardware VRAM, latency, and thermal testing on the Windows RTX 4060 Laptop GPU.
6. Independent validation for domain shift, defect types missing from MVTec, and the real production camera setup.

Until those stages are complete, use QuantumInspect only as a research workbench with human review—not automated manufacturing acceptance/rejection.
