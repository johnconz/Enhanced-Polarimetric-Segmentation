import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import hashlib
from collections import OrderedDict
import helper_functions as hf
from ASL import ASL

MIN_INTENSITY = 1
MAX_INTENSITY = 4094.0

S0_STD = 3
DOLP_MAX = 1.0
AOP_MAX = 0.5
FUSION_COEFFICIENT = 0.5

class MultiModalASLDataset(Dataset):
    def __init__(self,
                 asl_files,
                 mask_files,
                 modalities=("s0", "dolp", "aop"),
                 aop_mode: int = 1,
                 compute_enhanced: bool = False,
                 raw_scale: bool = False,
                 min_max: bool = False,
                 debug: bool = False,
                 stack_modalities: bool = False,
                 cache_dir: str = "/home/connor/Thesis/cache",
                 enable_disk_cache: bool = True,
                 enable_ram_cache: bool = False,
                 max_ram_cache_size: int = 175):
        """
        Hybrid dataset with LRU RAM cache + optional disk cache.

        max_ram_cache_size: max number of frames to keep in RAM cache
        """
        self.asl_files = list(asl_files)
        self.mask_files = list(mask_files)
        self.modalities = modalities
        self.aop_mode = aop_mode
        self.compute_enhanced = compute_enhanced
        self.raw_scale = raw_scale
        self.min_max = min_max
        self.debug = debug
        self.stack_modalities = stack_modalities
        self.enable_disk_cache = enable_disk_cache
        self.enable_ram_cache = enable_ram_cache
        self.max_ram_cache_size = max_ram_cache_size

        self.cache_dir = Path(cache_dir)
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # LRU RAM cache: OrderedDict to evict oldest items
        self._ram_cache = OrderedDict()

        self.index_map = []
        self._asl_headers = [ASL(f) for f in self.asl_files]

        for file_idx, (asl_obj, mask_file) in enumerate(zip(self._asl_headers, self.mask_files)):
            hdr = asl_obj.get_header()
            with np.load(mask_file) as npz:
                mask_array = npz["relabeled_masks"]
                valid_pixels = npz["valid_pixels"]

            num_masks = mask_array.shape[-1]
            num_valid_pixels = valid_pixels.shape[-1]
            num_frames = hdr.required["frames"]

            for frame_idx in range(num_frames):
                az_idx = frame_idx % num_masks
                vp_idx = frame_idx % num_valid_pixels
                self.index_map.append((
                    file_idx,
                    frame_idx,
                    mask_array[:, :, az_idx],
                    valid_pixels[:, :, vp_idx]
                ))

    def __len__(self):
        return len(self.index_map)

    def _get_cache_key(self, file_idx, frame_idx):
        """Unique key for RAM cache."""
        return f"{self.asl_files[file_idx]}_{frame_idx}_{'_'.join(self.modalities)}_{self.raw_scale}_{self.min_max}_{self.compute_enhanced}"

    def _get_cache_path(self, cache_key):
        """Disk cache filename."""
        hashed = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.pt"

    def _compute_modalities(self, frame_data, valid_pixels, asl_obj):
        frame_data = frame_data.astype(np.float32)
        if self.raw_scale:
            frame_data = (frame_data - MIN_INTENSITY) / (MAX_INTENSITY - MIN_INTENSITY + 1e-8)

        S = hf.compute_stokes(frame_data)
        s0, s1, s2 = S[:, :, 0], S[:, :, 1], S[:, :, 2]

        if self.min_max:
            s0 = (s0 - np.min(s0)) / (np.max(s0) - np.min(s0) + 1e-8)
        elif not self.raw_scale:
            s0_log = np.log1p(s0)
            s0 = (s0_log - np.min(s0_log)) / (np.max(s0_log) - np.min(s0_log) + 1e-8)

        output = {}
        if "s0" in self.modalities:
            output["s0"] = s0
        if "s1" in self.modalities:
            output["s1"] = s1 / (s0 + 1e-8)
        if "s2" in self.modalities:
            output["s2"] = s2 / (s0 + 1e-8)
        if "dolp" in self.modalities:
            output["dolp"] = np.clip(np.sqrt(s1**2 + s2**2), 0, 1)
        if "aop" in self.modalities:
            aop = 0.5 * np.arctan2(s2, s1)
            output["aop"] = (aop + np.pi / 2) / np.pi

        if self.compute_enhanced:
            es0, shape_enhancement, shape_contrast_enhancement = hf.compute_enhanceds0(
                S, s0std=S0_STD, dolp_max=DOLP_MAX, aop_max=AOP_MAX,
                fusion_coefficient=FUSION_COEFFICIENT,
                valid_pixels=valid_pixels,
                hdr=asl_obj,
                aop_mode=self.aop_mode
            )
            if "enhanced_s0" in self.modalities:
                output["enhanced_s0"] = es0
            if "shape_enhancement" in self.modalities:
                output["shape_enhancement"] = shape_enhancement.squeeze(-1)
            if "shape_contrast_enhancement" in self.modalities:
                output["shape_contrast_enhancement"] = shape_contrast_enhancement.squeeze(-1)

        for k in output:
            output[k] = torch.from_numpy(output[k]).unsqueeze(0).float()

        if self.stack_modalities:
            return torch.cat([output[k] for k in self.modalities if k in output], dim=0)

        return output

    def __getitem__(self, idx):
        file_idx, frame_idx, mask_np, valid_np = self.index_map[idx]
        cache_key = self._get_cache_key(file_idx, frame_idx)
        cache_path = self._get_cache_path(cache_key)

        # 1. Try RAM cache first
        if self.enable_ram_cache and cache_key in self._ram_cache:
            # Move to end to mark as recently used
            self._ram_cache.move_to_end(cache_key)
            return self._ram_cache[cache_key]

        # 2. Try disk cache
        if self.enable_disk_cache and cache_path.exists():
            result = torch.load(cache_path)
            if self.enable_ram_cache:
                self._ram_cache[cache_key] = result
                # enforce max size
                if len(self._ram_cache) > self.max_ram_cache_size:
                    self._ram_cache.popitem(last=False)
            return result

        # 3. Compute from scratch
        asl_obj = self._asl_headers[file_idx]
        frame_data, _, _ = asl_obj.get_data(frames=[frame_idx + 1])
        modalities = self._compute_modalities(frame_data, valid_np, asl_obj)
        mask = torch.from_numpy(mask_np).long()
        valid_pixels = torch.from_numpy(valid_np).bool()

        result = (modalities, mask, valid_pixels)

        # Save to RAM cache
        if self.enable_ram_cache:
            self._ram_cache[cache_key] = result
            # enforce max size
            if len(self._ram_cache) > self.max_ram_cache_size:
                self._ram_cache.popitem(last=False)

        # Save to disk cache
        if self.enable_disk_cache:
            torch.save(result, cache_path)

        return result
