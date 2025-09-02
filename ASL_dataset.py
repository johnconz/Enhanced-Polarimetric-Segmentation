"""
asl_dataset.py

ASLFrameDataset: a PyTorch Dataset for loading frames from ASL files
(lazy or preload), computing Stokes-derived modalities and optional
enhanced S0 outputs using your existing ASL reader and helper_functions.

Usage:
    from datasets.asl_dataset import ASLFrameDataset, build_file_index

    file_index = build_file_index(data_paths, mask_paths)
    dataset = ASLFrameDataset(file_index, modalities=["s0","dolp","aop"],
                              preload=False, compute_enhanced=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4)

This file expects your project to expose:
 - ASL class (ASL.get_data(frames=[1-based index]) -> frame_data,...)
 - helper_functions (hf) with compute_stokes(...) and compute_enhanceds0(...)

"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset

# import your existing ASL reader and helper functions
# (these should be importable from your project)
from ASL import ASL
import helper_functions as hf


def _default_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a 2D numpy array to a torch tensor shaped [1, H, W]."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    t = torch.from_numpy(arr)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t.float()


class ASLFrameDataset(Dataset):
    """Dataset exposing single frames from ASL files as multimodal samples.

    file_index: list of dicts with keys:
        - "asl_path": str or Path to the .asl.hdr
        - "frame": int (0-based frame index)
        - "mask": np.ndarray (H,W) or None
        - "valid_pixels": np.ndarray (H,W) or None
        - "meta": optional dict
        - "label": optional int (target)

    modalities: list of modality names to produce, e.g. ["s0","s1","s2","dolp","aop","es0","s0e1","s0e2"]

    transforms: dict mapping modality -> callable(np.ndarray) -> torch.Tensor
        If a modality transform isn't provided the dataset will convert
        the np.ndarray to a torch tensor with shape [C,H,W] (C=1 for 2D arrays).

    preload: if True, loads and processes every sample into memory at init.
    compute_enhanced: if True, it will call hf.compute_enhanceds0 for the frame
    cache_dir: optional path where computed enhanced outputs will be saved/loaded as .npy to speed up repeated runs.
    """

    def __init__(
        self,
        file_index: List[Dict[str, Any]],
        modalities: List[str],
        transforms: Optional[Dict[str, Callable]] = None,
        preload: bool = False,
        compute_enhanced: bool = False,
        s0std: float = 3.0,
        dolp_max: float = 1.0,
        aop_max: float = 0.5,
        fusion_coeff: float = 0.5,
        aop_mode: int = 1,
        cache_dir: Optional[str] = None,
        asl_class=ASL,
    ) -> None:
        self.file_index = file_index
        self.modalities = modalities
        self.transforms = transforms or {}
        self.preload = preload
        self.compute_enhanced = compute_enhanced
        self.s0std = s0std
        self.dolp_max = dolp_max
        self.aop_max = aop_max
        self.fusion_coeff = fusion_coeff
        self.aop_mode = int(aop_mode)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache: Dict[int, Dict[str, Any]] = {}
        self.ASL = asl_class

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.preload:
            self._preload_all()

    def _preload_all(self) -> None:
        for i in range(len(self.file_index)):
            self._cache[i] = self._load_and_process(i)

    def __len__(self) -> int:
        return len(self.file_index)

    def __getitem__(self, idx: int):
        if self.preload and idx in self._cache:
            sample = self._cache[idx]
        else:
            sample = self._load_and_process(idx)
            if self.preload:
                self._cache[idx] = sample

        # Convert modalities to tensors and apply transforms
        out_modalities: Dict[str, torch.Tensor] = {}
        for mod, arr in sample["modalities"].items():
            if mod in self.transforms:
                out_modalities[mod] = self.transforms[mod](arr)
            else:
                out_modalities[mod] = _default_to_tensor(arr)

        label = sample.get("label", -1)
        mask = sample.get("mask", None)
        meta = sample.get("meta", None)

        return out_modalities, label, mask, meta

    def _cache_paths_for(self, asl_path: str, frame_idx: int) -> Dict[str, Path]:
        base = Path(asl_path).stem.replace(".", "_")
        stem = f"{base}_frame{frame_idx}"
        return {
            "es0": self.cache_dir / f"{stem}_es0.npy",
            "s0e1": self.cache_dir / f"{stem}_s0e1.npy",
            "s0e2": self.cache_dir / f"{stem}_s0e2.npy",
        }

    def _load_and_process(self, idx: int) -> Dict[str, Any]:
        entry = self.file_index[idx]
        asl_path = str(entry["asl_path"]) if "asl_path" in entry else entry.get("path")
        frame_idx = int(entry["frame"])
        mask = entry.get("mask", None)
        valid_pixels = entry.get("valid_pixels", None)
        meta = entry.get("meta", {})
        label = entry.get("label", -1)

        # Read single frame using your ASL reader (ASL.get_data expects 1-based frames)
        asl = self.ASL(asl_path)
        frame_data, _, _ = asl.get_data(frames=[frame_idx + 1])
        frame_data = frame_data.astype(np.float32)

        # Compute Stokes using helper function (expected shape (H, W, 5))
        stokes = hf.compute_stokes(frame_data)
        H, W = stokes.shape[0], stokes.shape[1]

        modalities_out: Dict[str, np.ndarray] = {}

        # basic channels
        if "s0" in self.modalities:
            modalities_out["s0"] = stokes[:, :, 0].astype(np.float32)
        if "s1" in self.modalities:
            modalities_out["s1"] = stokes[:, :, 1].astype(np.float32)
        if "s2" in self.modalities:
            modalities_out["s2"] = stokes[:, :, 2].astype(np.float32)

        # derived DoLP / AoP (normalize as in your prepare_data)
        if any(m in self.modalities for m in ("dolp", "aop")):
            s0 = stokes[:, :, 0].astype(np.float32)
            s1 = stokes[:, :, 1].astype(np.float32)
            s2 = stokes[:, :, 2].astype(np.float32)
            s1n = s1 / (s0 + 1e-8)
            s2n = s2 / (s0 + 1e-8)
            dolp = np.sqrt(s1n**2 + s2n**2)
            aop = 0.5 * np.arctan2(s2n, s1n)
            if "dolp" in self.modalities:
                modalities_out["dolp"] = np.clip(dolp, 0, 1).astype(np.float32)
            if "aop" in self.modalities:
                # keep radians in [-pi/2, pi/2]
                modalities_out["aop"] = aop.astype(np.float32)

        # Optionally compute enhanced outputs (calls your helper)
        if self.compute_enhanced:
            # hf.compute_enhanceds0 expects S as (H,W,5) or (H,W,5,num_frames)
            # here we pass a single-frame S
            S_stack = stokes.copy()

            # caching support
            es0 = None
            s0e1 = None
            s0e2 = None
            if self.cache_dir is not None:
                cpaths = self._cache_paths_for(asl_path, frame_idx)
                if cpaths["es0"].exists() and cpaths["s0e1"].exists() and cpaths["s0e2"].exists():
                    es0 = np.load(cpaths["es0"])
                    s0e1 = np.load(cpaths["s0e1"])
                    s0e2 = np.load(cpaths["s0e2"])
            if es0 is None:
                es0, s0e1_all, s0e2_all = hf.compute_enhanceds0(
                    S_stack,
                    self.s0std,
                    self.dolp_max,
                    self.aop_max,
                    self.fusion_coeff,
                    valid_pixels if valid_pixels is not None else np.ones((H, W)),
                    asl,
                    self.aop_mode,
                )
                # hf returns es0 (2D) and s0e1/s0e2 shaped (H,W,num_frames) or (H,W)
                # extract first frame if necessary
                if isinstance(s0e1_all, np.ndarray) and s0e1_all.ndim == 3:
                    s0e1 = s0e1_all[:, :, 0]
                else:
                    s0e1 = s0e1_all
                if isinstance(s0e2_all, np.ndarray) and s0e2_all.ndim == 3:
                    s0e2 = s0e2_all[:, :, 0]
                else:
                    s0e2 = s0e2_all

                if self.cache_dir is not None:
                    np.save(self.cache_dir / f"{Path(asl_path).stem}_frame{frame_idx}_es0.npy", es0)
                    np.save(self.cache_dir / f"{Path(asl_path).stem}_frame{frame_idx}_s0e1.npy", s0e1)
                    np.save(self.cache_dir / f"{Path(asl_path).stem}_frame{frame_idx}_s0e2.npy", s0e2)

            modalities_out["es0"] = np.asarray(es0, dtype=np.float32)
            modalities_out["s0e1"] = np.asarray(s0e1, dtype=np.float32)
            modalities_out["s0e2"] = np.asarray(s0e2, dtype=np.float32)

        sample = {
            "modalities": modalities_out,
            "mask": mask,
            "valid_pixels": valid_pixels,
            "meta": meta,
            "label": label,
        }
        return sample


# Utility to build file index from directories (mirrors your prepare_data logic)
def build_file_index(data_paths: List[Path], mask_paths: List[Path]) -> List[Dict[str, Any]]:
    """Construct file_index list from matching ASL header files and mask npz files.

    data_paths and mask_paths must be parallel-ordered lists (zip-able).
    Each returned dict contains 'asl_path', 'frame', 'mask', 'valid_pixels', 'meta'.
    """
    file_index: List[Dict[str, Any]] = []
    for data_file, mask_file in zip(data_paths, mask_paths):
        asl = ASL(data_file)
        hdr = asl.get_header()
        num_frames = int(hdr.required["frames"]) if "frames" in hdr.required else 1

        with np.load(mask_file) as npz:
            mask_array = npz["relabeled_masks"]  # (H,W,azimuths)
            valid_pixels_stack = npz["valid_pixels"]  # (H,W,num_frames)

        for frame_idx in range(num_frames):
            az_idx = frame_idx % mask_array.shape[-1]
            mask_for_frame = mask_array[:, :, az_idx]
            valid_for_frame = valid_pixels_stack[:, :, frame_idx]
            file_index.append(
                {
                    "asl_path": str(data_file),
                    "frame": int(frame_idx),
                    "mask": mask_for_frame.astype(np.uint8),
                    "valid_pixels": valid_for_frame.astype(np.uint8),
                    "meta": {"scene": data_file.name, "frame": frame_idx},
                }
            )
    return file_index


if __name__ == "__main__":
    # quick demo: build index from folders and iterate dataset
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/connor/MATLAB/data")
    parser.add_argument("--mask_dir", type=str, default="/home/connor/Thesis/updated_masks")
    args = parser.parse_args()

    data_files = sorted(Path(args.data_dir).glob("*.asl.hdr"))
    mask_files = sorted(Path(args.mask_dir).glob("*.npz"))

    file_index = build_file_index(data_files, mask_files)
    print(f"Built file_index with {len(file_index)} entries")

    dataset = ASLFrameDataset(
        file_index=file_index,
        modalities=["s0", "dolp", "aop", "es0"],
        preload=False,
        compute_enhanced=True,
        cache_dir="./cache_asl",
    )

    # example: iterate first 5 samples
    for i in range(min(5, len(dataset))):
        mod_dict, label, mask, meta = dataset[i]
        print(i, meta, {k: v.shape for k, v in mod_dict.items()})
