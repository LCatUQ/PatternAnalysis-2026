"""HipMRI 2D prostate slices: discovery, patient-level checks, loading."""

import glob
import math
import os
import random
import re
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CLASS_NAMES = ("background", "body", "bone", "bladder", "rectum", "prostate")
PROSTATE = 5
NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$")


@dataclass(frozen=True)
class SliceRef:
    image: str
    label: str
    case: str
    week: int
    index: int


def discover(root, split):
    image_dir = os.path.join(root, f"keras_slices_{split}")
    label_dir = os.path.join(root, f"keras_slices_seg_{split}")
    refs = []
    for path in sorted(glob.glob(os.path.join(image_dir, "*.nii.gz"))):
        name = os.path.basename(path)
        match = NAME_RE.match(name)
        if match is None:
            continue
        case, week, index = match.groups()
        label = os.path.join(label_dir, "seg_" + name[len("case_"):])
        if not os.path.exists(label):
            raise FileNotFoundError(label)
        refs.append(SliceRef(path, label, case, int(week), int(index)))
    refs.sort(key=lambda r: (r.case, r.week, r.index))
    return refs


def patient_ids(refs):
    return {ref.case for ref in refs}


def check_split(train, val, test):
    groups = {"train": patient_ids(train), "val": patient_ids(val), "test": patient_ids(test)}
    for left in groups:
        for right in groups:
            if left >= right:
                continue
            shared = groups[left] & groups[right]
            if shared:
                raise ValueError(f"patient leakage between {left} and {right}: {sorted(shared)}")
    return groups


def normalize(image):
    body = image > 0
    if body.any():
        values = image[body]
        mean, std = values.mean(), values.std()
    else:
        mean, std = image.mean(), image.std()
    return (image - mean) / (std + 1e-6)


class HipMRISlices(Dataset):
    def __init__(self, refs, augment=False, preload=True):
        self.refs = refs
        self.augment = augment
        self.images = None
        self.labels = None
        if preload and refs:
            height, width = nib.load(refs[0].image).shape[:2]
            self.images = np.empty((len(refs), height, width), dtype=np.float16)
            self.labels = np.empty((len(refs), height, width), dtype=np.uint8)
            for i, ref in enumerate(refs):
                image, mask = self._read(ref)
                if image.shape != (height, width):
                    raise ValueError(f"inconsistent shape in {ref.image}: {image.shape}")
                self.images[i] = image.astype(np.float16)
                self.labels[i] = mask

    @staticmethod
    def _read(ref):
        image = np.asanyarray(nib.load(ref.image).dataobj).astype(np.float32)
        mask = np.asanyarray(nib.load(ref.label).dataobj).astype(np.uint8)
        if image.shape != mask.shape:
            raise ValueError(f"image/label shape mismatch: {ref.image}")
        return image, mask

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        if self.images is None:
            image, mask = self._read(self.refs[index])
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask.astype(np.int64))
        else:
            image = torch.from_numpy(self.images[index].astype(np.float32)).unsqueeze(0)
            mask = torch.from_numpy(self.labels[index].astype(np.int64))

        if self.augment:
            image, mask = self._geometry(image, mask)
        image = normalize(image)
        if self.augment:
            image = image * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return image, mask

    @staticmethod
    def _geometry(image, mask):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if random.random() < 0.7:
            angle = math.radians(random.uniform(-10, 10))
            scale = random.uniform(0.95, 1.05)
            tx = random.uniform(-0.03, 0.03)
            ty = random.uniform(-0.03, 0.03)
            theta = torch.tensor(
                [[math.cos(angle) * scale, -math.sin(angle) * scale, tx],
                 [math.sin(angle) * scale, math.cos(angle) * scale, ty]],
                dtype=torch.float32,
            ).unsqueeze(0)
            grid = F.affine_grid(theta, (1,) + tuple(image.shape), align_corners=False)
            image = F.grid_sample(
                image.unsqueeze(0), grid, align_corners=False, padding_mode="border"
            )[0]
            mask = F.grid_sample(
                mask[None, None].float(), grid, mode="nearest",
                align_corners=False, padding_mode="border",
            )[0, 0].long()
        return image, mask


def make_datasets(root, limit=0):
    train_refs = discover(root, "train")
    val_refs = discover(root, "validate")
    test_refs = discover(root, "test")
    check_split(train_refs, val_refs, test_refs)

    if limit:
        step = max(1, len(train_refs) // limit)
        train_refs = train_refs[::step][:limit]

    return (
        HipMRISlices(train_refs, augment=True),
        HipMRISlices(val_refs, augment=False),
        HipMRISlices(test_refs, augment=False),
    )
