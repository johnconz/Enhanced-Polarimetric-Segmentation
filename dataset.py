import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import hashlib
from collections import OrderedDict
import helper_functions as hf
from ASL import ASL
import random  # <-- added

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
    def __init__(self, probability=0.5, rare_classes=None, min_size=32, max_size=128):
        """
        Args:
            probability: chance to apply CutMix
            rare_classes: list of underrepresented class indices
            min_size, max_size: size of patch to cut
        """
        self.probability = probability
        self.rare_classes = rare_classes if rare_classes is not None else []
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img1, mask1, img2, mask2):
        if random.random() > self.probability or not self.rare_classes:
            return img1, mask1

        H, W = mask1.shape
        cls = random.choice(self.rare_classes)

        coords = (mask2 == cls).nonzero(as_tuple=False)
        if len(coords) == 0:
            return img1, mask1

        y, x = coords[random.randint(0, len(coords) - 1)]
        size = random.randint(self.min_size, self.max_size)

        y1, y2 = max(0, y - size // 2), min(H, y + size // 2)
        x1, x2 = max(0, x - size // 2), min(W, x + size // 2)

        patch_img = img2[:, y1:y2, x1:x2]
        patch_mask = mask2[y1:y2, x1:x2]

        target_mask_crop = mask1[y1:y2, x1:x2]
        paste_area = (target_mask_crop == 0)  # assume 0=background

        new_img = img1.clone()
        new_mask = mask1.clone()
        new_img[:, y1:y2, x1:x2][:, paste_area] = patch_img[:, paste_area]
        new_mask[y1:y2, x1:x2][paste_area] = patch_mask[paste_area]

        return new_img, new_mask

# -------------------------
# Dataset Class
# -------------------------
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
                 max_ram_cache_size: int = 175,
                 cutmix_aug=None,
                 cutmix_active=False):   # <-- new args
        """
        Hybrid dataset with LRU RAM cache + optional disk cache.
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
        self.cutmix_aug = cutmix_aug
        self.cutmix_active = cutmix_active

        self.cache_dir = Path(cache_dir)
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        return f"{self.asl_files[file_idx]}_{frame_idx}_{'_'.join(self.modalities)}_{self.raw_scale}_{self.min_max}_{self.compute_enhanced}"

    def _get_cache_path(self, cache_key):
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

        # RAM cache
        if self.enable_ram_cache and cache_key in self._ram_cache:
            self._ram_cache.move_to_end(cache_key)
            return self._ram_cache[cache_key]

        # Disk cache
        if self.enable_disk_cache and cache_path.exists():
            result = torch.load(cache_path)
            if self.enable_ram_cache:
                self._ram_cache[cache_key] = result
                if len(self._ram_cache) > self.max_ram_cache_size:
                    self._ram_cache.popitem(last=False)
            return result

        # Compute
        asl_obj = self._asl_headers[file_idx]
        frame_data, _, _ = asl_obj.get_data(frames=[frame_idx + 1])
        modalities = self._compute_modalities(frame_data, valid_np, asl_obj)
        mask = torch.from_numpy(mask_np).long()
        valid_pixels = torch.from_numpy(valid_np).bool()
        result = (modalities, mask, valid_pixels)

        # Caching
        if self.enable_ram_cache:
            self._ram_cache[cache_key] = result
            if len(self._ram_cache) > self.max_ram_cache_size:
                self._ram_cache.popitem(last=False)
        if self.enable_disk_cache:
            torch.save(result, cache_path)

        # -------------------------
        # Apply CutMix augmentation
        # -------------------------
        if self.cutmix_active and self.cutmix_aug is not None:
            j = random.randint(0, len(self.index_map) - 1)
            file_idx2, frame_idx2, mask_np2, valid_np2 = self.index_map[j]
            asl_obj2 = self._asl_headers[file_idx2]
            frame_data2, _, _ = asl_obj2.get_data(frames=[frame_idx2 + 1])
            modalities2 = self._compute_modalities(frame_data2, valid_np2, asl_obj2)
            mask2 = torch.from_numpy(mask_np2).long()

            if self.stack_modalities:
                img1 = modalities
                img2 = modalities2
            else:
                img1 = torch.cat([modalities[k] for k in self.modalities], dim=0)
                img2 = torch.cat([modalities2[k] for k in self.modalities], dim=0)

            new_img, new_mask = self.cutmix_aug(img1, mask, img2, mask2)

            if self.stack_modalities:
                result = (new_img, new_mask, valid_pixels)
            else:
                new_modalities = {k: new_img[i].unsqueeze(0) for i, k in enumerate(self.modalities)}
                result = (new_modalities, new_mask, valid_pixels)

        return result
