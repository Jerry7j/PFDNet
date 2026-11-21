"""Build training targets from MAT or JPEG landmark samples."""

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms

from ..annotation.labelme import read_text_points
from .heatmaps import draw_umich_gaussian

POINT_COUNTS = {"hip": 1, "knee": 6, "ankle": 3}
TRAIN_IMAGE_SUFFIXES = {".mat", ".jpg", ".jpeg"}
INPUT_SIZE = 224
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])


def read_image(path, region):
    """Read a MAT sample or regular image and return RGB pixels."""
    if path.suffix.lower() == ".mat":
        image = loadmat(path)[f"{region}sample"]
    else:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)
    conversion = cv2.COLOR_GRAY2RGB if grayscale else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(image, conversion)


def image_tensor(rgb):
    # The original detector trained on RGB values in the 0-255 range.
    return IMAGE_TRANSFORM(rgb) * 255.0


def read_points(path, region, point_count=None):
    if path.suffix.lower() == ".txt":
        coordinates = read_text_points(path)
    else:
        coordinates = loadmat(path)[f"{region}label"]
    points = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    count = POINT_COUNTS[region] if point_count is None else point_count
    if points.size != count * 2 or not np.isfinite(points).all():
        raise ValueError(f"{path}: expected {count} finite (x, y) pairs")
    return points.reshape(count, 2)


def label_path_for(sample, label_dir, region):
    """Match JPEG text sidecars by stem, or pair with a regional MAT label."""
    text_label = label_dir / f"{sample.stem}.txt"
    is_mat = sample.suffix.lower() == ".mat"
    if not is_mat and text_label.is_file():
        return text_label

    # MAT files use the same sample ID under different image and label prefixes.
    prefix = f"{region}sample"
    if sample.stem.startswith(prefix):
        mat_label = label_dir / f"{region}label{sample.stem[len(prefix):]}.mat"
        if mat_label.is_file() or is_mat:
            return mat_label
    elif is_mat:
        raise ValueError(f"{sample.name}: expected a filename beginning with {prefix}")
    return text_label


class LandmarkDataset(Dataset):
    def __init__(self, root, region, split="train", *, images_dir=None, labels_dir=None, point_count=None):
        suffix = {"train": "", "val": "_val", "test": "_test"}[split]
        sample_dir = images_dir if images_dir is not None else root / f"{region}sample{suffix}"
        labels = labels_dir if labels_dir is not None else root / f"{region}labels{suffix}"
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {sample_dir}")
        samples = sorted(path for path in sample_dir.iterdir()
                         if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_SUFFIXES)
        if not samples:
            raise ValueError(f"No MAT or JPEG samples found in {sample_dir}")
        self.pairs = [(sample, label_path_for(sample, labels, region)) for sample in samples]
        for _, label in self.pairs:
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
        self.region = region
        self.point_count = point_count

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sample, label = self.pairs[index]
        rgb = read_image(sample, self.region)
        height, width = rgb.shape[:2]
        points = read_points(label, self.region, self.point_count)
        heatmaps = np.zeros((len(points), INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for heatmap, (x, y) in zip(heatmaps, points):
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(f"{label}: landmark lies outside its image")
            # Also accept points at the far edge of the MATLAB coordinate range.
            center = (min(x * INPUT_SIZE / width, INPUT_SIZE - 1),
                      min(y * INPUT_SIZE / height, INPUT_SIZE - 1))
            draw_umich_gaussian(heatmap, center, radius=10)
        return image_tensor(rgb), torch.from_numpy(heatmaps)
