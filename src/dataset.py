from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must have the same length.")

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        return image

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        return mask

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray):
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        return image, mask

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        image = np.transpose(image, (2, 0, 1))   # HWC -> CHW
        mask = np.expand_dims(mask, axis=0)       # HW -> 1HW

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def get_image_mask_paths(data_root: str):
    data_root = Path(data_root)
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    valid_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_suffixes
    ])

    mask_paths = sorted([
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_suffixes
    ])

    print(f"Found image files: {len(image_paths)}")
    print(f"Found mask files : {len(mask_paths)}")

    mask_map = {}
    for mask_path in mask_paths:
        # TCGA_xxx_1_mask.tif -> TCGA_xxx_1
        mask_key = mask_path.stem.replace("_mask", "")
        mask_map[mask_key] = mask_path

    matched_image_paths = []
    matched_mask_paths = []

    for image_path in image_paths:
        image_key = image_path.stem
        if image_key in mask_map:
            matched_image_paths.append(image_path)
            matched_mask_paths.append(mask_map[image_key])

    print(f"Matched pairs    : {len(matched_image_paths)}")

    if len(matched_image_paths) == 0:
        print("Sample image names:")
        for p in image_paths[:5]:
            print("  ", p.name)

        print("Sample mask names:")
        for p in mask_paths[:5]:
            print("  ", p.name)

        raise RuntimeError("No matched image-mask pairs found. Check file naming rules.")

    return matched_image_paths, matched_mask_paths