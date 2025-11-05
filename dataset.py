import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import OrderedDict
import helper_functions as hf
from ASL import ASL
import random
from filelock import FileLock # For safe multi-process caching
import os

MIN_INTENSITY = 1
MAX_INTENSITY = 4094.0

S0_STD = 3
DOLP_MAX = 1.0
AOP_MAX = 0.5
FUSION_COEFFICIENT = 0.5


# -------------------------
# CutMix Augmentation Class
# -------------------------
class CutMixSegmentation:
    def __init__(self, dataset, probability=0.5, rare_classes=None, min_size=32, max_size=128):
        """
        CutMix augmentation for segmentation.

        Args:
            dataset: reference to the dataset (to sample second random image/mask)
            probability: chance of applying CutMix
            rare_classes: list of class IDs considered "rare"
            min_size, max_size: bounds for random patch size
        """
        self.dataset = dataset
        self.probability = probability
        self.rare_classes = rare_classes if rare_classes else []
        self.min_size = min_size
        self.max_size = max_size

    def contains_rare_class(self, mask):
        """Check if mask contains at least one rare class."""
        if not self.rare_classes:
            return False
        return any((mask == c).any() for c in self.rare_classes)

    def __call__(self, img1, mask1):
        # Skip augmentation most of the time
        if random.random() > self.probability or not self.rare_classes:
            return img1, mask1, None, None

        # DO NOT NEED TO CHECK IF img1/mask1 CONTAINS RARE CLASS
        #if not self.contains_rare_class(mask1):
        #    return img1, mask1

        # Try to find a second image/mask with rare class
        max_attempts = 100
        for _ in range(max_attempts):
            idx2 = random.randint(0, len(self.dataset) - 1)
            img2, mask2, *_ = self.dataset[idx2]

            if self.contains_rare_class(mask2):
                print(f"[CutMix] Pasting from index {idx2}.")
                break
        else:
            # If no suitable partner found, skip CutMix
            return img1, mask1, None, None

        # Convert to tensors if not already
        if not isinstance(img1, torch.Tensor):
            img1 = torch.from_numpy(img1)
        if not isinstance(mask1, torch.Tensor):
            mask1 = torch.from_numpy(mask1)
        if not isinstance(img2, torch.Tensor):
            img2 = torch.from_numpy(img2)
        if not isinstance(mask2, torch.Tensor):
            mask2 = torch.from_numpy(mask2)

        #mask1before = mask1.clone()

        # Random patch size and position
        H, W = mask1.shape[-2:]
        cut_h = random.randint(self.min_size, min(self.max_size, H))
        cut_w = random.randint(self.min_size, min(self.max_size, W))

        # Choose rare class pixels
        rare_mask = torch.zeros_like(mask2, dtype=torch.bool)
        for c in self.rare_classes:
            rare_mask |= (mask2 == c)

        rare_pixels = rare_mask.nonzero(as_tuple=False)
        if rare_pixels.numel() == 0:
            return img1, mask1, None, None
        cy, cx = rare_pixels[random.randint(0, len(rare_pixels) - 1)].tolist()

        # Compute patch bounds
        cy_h = max(0, cy - cut_h // 2)
        cx_w = max(0, cx - cut_w // 2)
        cy_h2 = min(H, cy_h + cut_h)
        cx_w2 = min(H, cx_w + cut_w)
        cut_h, cut_w = cy_h2 - cy_h, cx_w2 - cx_w

        # Pick random spot to paste cut patch
        py = random.randint(0, H - cut_h)
        px = random.randint(0, W - cut_w)

        # Cut and paste patch from img2/mask2 into img1/mask1
        img1[..., py:py+cut_h, px:px+cut_w] = img2[..., cy_h:cy_h2, cx_w:cx_w2]
        mask1[..., py:py+cut_h, px:px+cut_w] = mask2[..., cy_h:cy_h2, cx_w:cx_w2]

        return img1, mask1, mask2, (px, py, cut_w, cut_h)


class MultiModalASLDataset(Dataset):
    def __init__(self,
                 asl_files,
                 mask_files,
                 modalities=("s0", "dolp", "aop"),
                 aop_mode=1,
                 compute_enhanced=False,
                 raw_scale=False,
                 min_max=False,
                 debug=False,
                 stack_modalities=False,
                 cache_dir="/home/connor/Thesis/cache",
                 enable_disk_cache=False,
                 enable_ram_cache=False,
                 max_ram_cache_size=175,
                 cutmix_aug=None,
                 cutmix_active=False,
                 visualize_cutmix=False):
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
        self.cutmix_aug = cutmix_aug
        self.cutmix_active = cutmix_active
        self.visualize_cutmix = visualize_cutmix

        self.cache_dir = Path(cache_dir)
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._ram_cache = OrderedDict()
        self._mask_arrays = []
        self._valid_arrays = []
        self.index_map = []
        self._asl_headers = [ASL(f) for f in self.asl_files]

        # Build index map
        for file_idx, (asl_obj, mask_file) in enumerate(zip(self._asl_headers, self.mask_files)):
            hdr = asl_obj.get_header()
            with np.load(mask_file) as npz:
                mask_array = npz["relabeled_masks"]
                valid_pixels = npz["valid_pixels"]

            self._mask_arrays.append(mask_array)
            self._valid_arrays.append(valid_pixels)

            num_masks = mask_array.shape[-1]
            num_valid_pixels = valid_pixels.shape[-1]
            num_frames = hdr.required["frames"]

            for frame_idx in range(num_frames):
                az_idx = frame_idx % num_masks
                vp_idx = frame_idx % num_valid_pixels
                self.index_map.append((file_idx, frame_idx, az_idx, vp_idx))

            # Precompute CutMix coords per mask if CutMix is active
            if self.cutmix_active and self.cutmix_aug is not None:
                # Use last frame as representative
                mask_tensor = torch.from_numpy(mask_array[:, :, 0]).long()
                self.cutmix_aug.precompute_coords(mask_tensor)

    def __len__(self):
        return len(self.index_map)

    def _get_cache_key(self, file_idx, frame_idx):
        return f"{self.asl_files[file_idx]}_{frame_idx}_{'_'.join(self.modalities)}_{self.raw_scale}_{self.min_max}_{self.compute_enhanced}"

    def _get_cache_path(self, cache_key):
        import hashlib
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
            es0, shape_enh, shape_contr = hf.compute_enhanceds0(
                S, s0std=S0_STD, dolp_max=DOLP_MAX, aop_max=AOP_MAX,
                fusion_coefficient=FUSION_COEFFICIENT,
                valid_pixels=valid_pixels,
                hdr=asl_obj,
                aop_mode=self.aop_mode
            )
            if "enhanced_s0" in self.modalities: 
                output["enhanced_s0"] = es0
            if "shape_enhancement" in self.modalities: 
                output["shape_enhancement"] = shape_enh.squeeze(-1)
            if "shape_contrast_enhancement" in self.modalities: 
                output["shape_contrast_enhancement"] = shape_contr.squeeze(-1)

        for k in output: 
            output[k] = torch.from_numpy(output[k]).unsqueeze(0).float()
        if self.stack_modalities: 
            return torch.cat([output[k] for k in self.modalities if k in output], dim=0)
        return output

    def __getitem__(self, idx):
        try:
            file_idx, frame_idx, az_idx, vp_idx = self.index_map[idx]
        except IndexError:
            return None
        mask_array = self._mask_arrays[file_idx]
        valid_array = self._valid_arrays[file_idx]

        mask_np = mask_array[:, :, az_idx]
        valid_np = valid_array[:, :, vp_idx]

        cache_key = self._get_cache_key(file_idx, frame_idx)
        cache_path = self._get_cache_path(cache_key)
        lock_path = str(cache_path) + ".lock"

        # RAM cache
        if self.enable_ram_cache and cache_key in self._ram_cache:
            self._ram_cache.move_to_end(cache_key)
            return self._ram_cache[cache_key]

        # Disk cache: safe load with FileLock
        if self.enable_disk_cache and cache_path.exists():
            # Acquire lock for read (other processes will also use the same FileLock)
            lock = FileLock(lock_path)
            with lock:
                # Force load on CPU - avoids CUDA device differences across workers
                result = torch.load(cache_path, map_location="cpu")

            # Optionally populate RAM cache
            if self.enable_ram_cache:
                self._ram_cache[cache_key] = result
                if len(self._ram_cache) > self.max_ram_cache_size:
                    self._ram_cache.popitem(last=False)
            return result

        # Compute modalities (heavy work may happen here)
        asl_obj = self._asl_headers[file_idx]
        frame_data, _, _ = asl_obj.get_data(frames=[frame_idx + 1])
        modalities = self._compute_modalities(frame_data, valid_np, asl_obj)
        mask = torch.from_numpy(mask_np).long()
        valid_pixels = torch.from_numpy(valid_np).bool()
        result = (modalities, mask, valid_pixels)

        # Cache to RAM/Disk safely
        if self.enable_ram_cache:
            self._ram_cache[cache_key] = result
            if len(self._ram_cache) > self.max_ram_cache_size:
                self._ram_cache.popitem(last=False)

        if self.enable_disk_cache:
            tmp_path = str(cache_path) + ".tmp"
            lock = FileLock(lock_path)
            with lock:
                # save to a tmp file first, then atomically replace
                torch.save(result, tmp_path)
                # os.replace is atomic on most OSes
                os.replace(tmp_path, cache_path)

        # Apply CutMix
        if self.cutmix_active and self.cutmix_aug is not None:
            if self.stack_modalities:
                img1 = modalities
            else:
                img1 = torch.cat([modalities[k] for k in self.modalities], dim=0)

            # Save copy of og mask for visualization
            mask_before = mask.clone()
            new_img, new_mask, cutmask, box = self.cutmix_aug(img1, mask)

            # Visualize CutMix result
            if self.visualize_cutmix:
                if cutmask is not None:
                    hf.visualize_cutmix(mask_before, new_mask, cutmask, box=box)
                    #self.visualize_cutmix = False  # Only visualize once
                else:
                    print("[CutMix] No CutMix applied for this sample.") 
            if self.stack_modalities:
                result = (new_img, new_mask, valid_pixels)
            else:
                new_modalities = {k: new_img[i].unsqueeze(0) for i, k in enumerate(self.modalities)}
                result = (new_modalities, new_mask, valid_pixels)

        return result

