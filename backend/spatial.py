import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

from backend.artifacts import artifact_hash
from backend.dataset import CATEGORIES, load_manifest
from backend.localization import extract_regions, heatmap_data_uri


class SpatialModel:
    def __init__(self, device=None, pretrained=True):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.extractor = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None).eval().to(self.device)
        self.extractor.requires_grad_(False)
        self.bank = None
        self.pixel_threshold = None
        self.image_threshold = None
        self.category = None
        self.training_hashes = []
        self.validation_hashes = []

    @torch.inference_mode()
    def features(self, image):
        image = ImageOps.exif_transpose(image).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
        tensor = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)
        model = self.extractor
        value = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
        value = model.layer1(value)
        lower = model.layer2(value)
        upper = F.interpolate(model.layer3(lower), size=lower.shape[-2:], mode='bilinear', align_corners=False)
        features = F.avg_pool2d(torch.cat([lower, upper], dim=1), 3, stride=1, padding=1)
        return F.normalize(features, dim=1)

    @torch.inference_mode()
    def embedding(self, image):
        return self.features(image).mean(dim=(2, 3)).squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def score_map(self, image):
        if self.bank is None:
            raise RuntimeError('A normal-image memory bank is required.')
        features = self.features(image)
        _, dimensions, height, width = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(-1, dimensions)
        distances = torch.cat([torch.cdist(chunk, self.bank).min(dim=1).values for chunk in patches.split(128)])
        resized = F.interpolate(distances.reshape(1, 1, height, width), size=(image.height, image.width), mode='bilinear', align_corners=False)
        return resized[0, 0].cpu().numpy()

    def predict(self, image, include_heatmap=True):
        started = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert('RGB')
        scores = self.score_map(image)
        score = float(scores.max())
        boxes, mask = extract_regions(scores, self.pixel_threshold)
        if score <= self.image_threshold:
            boxes, mask = [], np.zeros_like(scores, dtype=bool)
        # Unsupervised novelty alone cannot establish physical damage.
        status = 'REVIEW' if boxes else 'NORMAL'
        return dict(source='model', status=status, width=image.width, height=image.height, boxes=boxes,
                    score=score, threshold=self.pixel_threshold, elapsed_ms=(time.perf_counter() - started) * 1000,
                    heatmap=heatmap_data_uri(scores, self.pixel_threshold) if include_heatmap else None,
                    quantum=None, category=self.category,
                    warning='Unsupervised anomaly evidence, not a verified physical defect. NORMAL means no retained region, not a guarantee of quality.'), mask

    @classmethod
    def load(cls, path, device=None, expected_category=None):
        digest = artifact_hash(path)
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if artifact.get('version') != 1:
            raise ValueError('Unsupported spatial artifact.')
        if expected_category is not None and artifact.get('category') != expected_category:
            raise ValueError('Spatial artifact category does not match the requested category.')
        model = cls(device=device, pretrained=False)
        model.extractor.load_state_dict(artifact['extractor'])
        model.bank = artifact['bank'].to(model.device)
        model.pixel_threshold = artifact['pixel_threshold']
        model.image_threshold = artifact['image_threshold']
        model.category = artifact['category']
        model.artifact_sha256 = digest
        model.training_hashes = artifact['training_hashes']
        model.validation_hashes = artifact['validation_hashes']
        return model


def train(manifest, category, output, device=None, seed=42, memory_size=8192):
    if memory_size < 128 or memory_size > 32768:
        raise ValueError('Memory bank size must be between 128 and 32768 patches.')
    rows = load_manifest(manifest, category)
    training = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'val']
    if len(training) < 4 or len(validation) < 2 or any(row['label'] for row in training + validation):
        raise ValueError('Spatial fitting requires at least 4 normal training images and 2 independent normal validation images.')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialModel(device)
    model.category = category
    candidates = []
    per_image = max(1, min(128, memory_size // len(training)))
    for index, row in enumerate(training):
        with Image.open(row['path']) as image:
            features = model.features(image).permute(0, 2, 3, 1).reshape(-1, 384).cpu()
        chosen = rng.choice(len(features), size=min(per_image, len(features)), replace=False)
        candidates.append(features[chosen])
        print(f'Train features {index + 1}/{len(training)}', flush=True)
    bank = torch.cat(candidates)
    indices = rng.choice(len(bank), size=min(memory_size, len(bank)), replace=False)
    model.bank = bank[indices].to(model.device)
    calibration, maxima = [], []
    for row in validation:
        with Image.open(row['path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            scores = model.score_map(image)
        maxima.append(float(scores.max()))
        flat = scores.reshape(-1)
        calibration.append(flat[rng.choice(len(flat), size=min(10000, len(flat)), replace=False)])
    model.pixel_threshold = float(np.quantile(np.concatenate(calibration), .995))
    model.image_threshold = float(np.quantile(maxima, .95, method='higher'))
    model.training_hashes = [row['sha256'] for row in training]
    model.validation_hashes = [row['sha256'] for row in validation]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=1, category=category, extractor={k: v.cpu() for k, v in model.extractor.state_dict().items()},
                    bank=model.bank.cpu(), pixel_threshold=model.pixel_threshold, image_threshold=model.image_threshold,
                    training_hashes=model.training_hashes, validation_hashes=model.validation_hashes), output / f'{category}.pt')
    report = dict(status='trained_not_tested', category=category, seed=seed, training_images=len(training),
                  validation_images=len(validation), memory_patches=len(model.bank), device=str(model.device),
                  pixel_threshold=model.pixel_threshold, image_threshold=model.image_threshold,
                  threshold_protocol='Normal validation only; pixel q=.995, image-max q=.95 (higher). Not a guaranteed FPR.',
                  physical_defect_detection_validated=False)
    (output / f'{category}-training.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--memory-size', type=int, default=8192)
    args = parser.parse_args()
    train(args.manifest, args.category, args.output, args.device, args.seed, args.memory_size)
