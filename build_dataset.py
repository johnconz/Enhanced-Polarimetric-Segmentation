# ----------------------------------------------------------------------->
# Author: Connor Prikkel
# Applied Sensing Lab, University of Dayton
# 9/2/2025
# ----------------------------------------------------------------------->

from pathlib import Path
from ASL import ASL
import argparse
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
import torch
from typing import Callable, Dict, List, Optional, Any
import helper_functions as hf
from torch.utils.data import Dataset

# Define constants
MIN_INTENSITY = 1
MAX_INTENSITY = 4094.0

# For computing enhanced Stokes parameters
S0_STD = 3
DOLP_MAX = 1.0
AOP_MAX = 0.5
FUSION_COEFFICIENT = 0.5

def _default_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a 2D numpy array to a torch tensor shaped [1, H, W]."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    t = torch.from_numpy(arr)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t.float()


# Path to ASL data dir
files = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Vision Mamba for multiple datasets."
    )
    parser.add_argument(
        "--aop_rotate",
        action="store_true",
        help="Perform aop rotations on each frame.",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Apply logarithmic scaling to s0.",
    )
    parser.add_argument(
        "--raw_scale",
        action="store_true",
        help="Apply min-max scaling on raw intensity data.",
    )
    parser.add_argument(
        "--min_max",
        action="store_true",
        help="Apply min-max normalization to s0.",
    )
    parser.add_argument(
        "--hist_shift",
        action="store_true",
        help="Apply histogram shifting to s0.",
    )
    return parser.parse_args()

def match_mask_to_frame(mask_array, valid_pixels, stride=8):
    """
    Build a dictionary mapping each data frame to its corresponding 
    mask frame and valid_pixels array.
    """

    mapping = {}
    num_frames = valid_pixels.shape[-1]

    for frame_idx in range(num_frames):

        # frame 0 -> mask[0], frame 1 -> mask[1], etc.
        # The azimuth repeats every 'stride' frames
        azimuth_idx = frame_idx % stride
        mapping[frame_idx] = {
            "mask": mask_array[:, :, azimuth_idx],
            "valid_pixels": valid_pixels[:, :, frame_idx]
        }

    return mapping

class ASLFrameDataset(Dataset):
    """
    Process all ASL files in the given directory.

    Loads data header, mask array, valid_pixels;
    builds a mapping, and computes stokes + enhanced parameters per frame. Utilizes lazy loading.
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
    """
    Construct a file_index list from ASL headers and mask npz files.
    Uses modular indexing like match_mask_to_frame to handle mismatched frame counts.
    """
    file_index: List[Dict[str, Any]] = []

    for data_file, mask_file in zip(data_paths, mask_paths):
        asl = ASL(data_file)
        hdr = asl.get_header()
        num_frames = int(hdr.required.get("frames", 1))  # total frames in ASL

        with np.load(mask_file) as npz:
            mask_array = npz["relabeled_masks"]        # (H, W, azimuths)
            valid_pixels_stack = npz["valid_pixels"]   # (H, W, num_valid_frames)

        # Build mapping using modular indexing
        mapping = match_mask_to_frame(mask_array, valid_pixels_stack)

        for frame_idx in range(num_frames):
            file_index.append(
                {
                    "asl_path": str(data_file),
                    "frame": frame_idx,
                    "mask": mapping[frame_idx]["mask"].astype(np.uint8),
                    "valid_pixels": mapping[frame_idx]["valid_pixels"].astype(np.uint8),
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

