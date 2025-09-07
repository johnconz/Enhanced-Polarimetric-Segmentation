# ----------------------------------------------------------------------->
# Author: Connor Prikkel
# Applied Sensing Lab, University of Dayton
# 9/6/2025
# ----------------------------------------------------------------------->

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from ASL import ASL
import helper_functions as hf
import argparse

# Define constants
MIN_INTENSITY = 1
MAX_INTENSITY = 4094.0

# For computing enhanced Stokes parameters
S0_STD = 3
DOLP_MAX = 1.0
AOP_MAX = 0.5
FUSION_COEFFICIENT = 0.5

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Vision Mamba for multiple datasets."
    )
    parser.add_argument(
        "--s0",
        action="store_true",
        help="Compute s0.",
    )
    parser.add_argument(
        "--s1",
        action="store_true",
        help="Compute s1.",
    )
    parser.add_argument(
        "--s2",
        action="store_true",
        help="Compute s2.",
    )
    parser.add_argument(
        "--dolp",
        action="store_true",
        help="Compute DoLP.",
    )
    parser.add_argument(
        "--aop",
        action="store_true",
        help="Compute AoP.",
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
        "--aop_rotate",
        action="store_true",
        help="Apply aop rotations to s0.",
    )
    parser.add_argument(
        "--enhanced_mixtures",
        action="store_true",
        help="Save 'mixtures' relating to enhanced s0."
    )
    parser.add_argument(
        "--hist_shift",
        action="store_true",
        help="Apply histogram shifting to s0.",
    )
    parser.add_argument(
        "--stack_modalities",
        action="store_true",
        help="Stack output modalities -> represent as a tensor.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add debug print statements to see intermediate output."
    )
    return parser.parse_args()

class MultiModalASLDataset(Dataset):
    """
    PyTorch-ready dataset for ASL polarimetric data with masks and multiple input modalities.
    Uses lazy loading: frames are read only on demand.
    """

    def __init__(self, 
                 asl_files, 
                 mask_files, 
                 modalities=("s0", "dolp", "aop"), 
                 aop_mode: int = 1, 
                 compute_enhanced: bool = False,
                 raw_scale: bool = False,
                 min_max: bool = False,
                 debug: bool = False,
                 stack_modalities: bool = False):
        """
        Parameters
        ----------
        asl_files : list of Path
            List of .asl.hdr files
        mask_files : list of Path
            List of .npz mask files (with relabeled_masks, valid_pixels)
        modalities : tuple
            Any combination of "s0", "s1", "s2", "dolp", "aop", "enhanced_s0"
        aop_mode : str
            AoP rotation mode if enhanced_s0 is computed
        compute_enhanced : bool
            Whether to compute enhanced s0 maps (s0e1, s0e2)
        raw_scale : bool
            If True, min–max normalize raw data to [0,1] before computing Stokes
        min_max : bool
            If True, apply min–max normalization to s0
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

        # Pre-build index mapping (file_idx, frame_idx → mask slice)
        self.index_map = []  # list of (file_idx, frame_idx, mask, valid_pixels)
        self._asl_headers = [ASL(f) for f in self.asl_files]

        for file_idx, (asl_obj, mask_file) in enumerate(zip(self._asl_headers, self.mask_files)):
            hdr = asl_obj.get_header()
            with np.load(mask_file) as npz:
                mask_array = npz["relabeled_masks"]    # (H,W,8)
                valid_pixels = npz["valid_pixels"]     # (H,W,num_frames)

            num_masks = mask_array.shape[-1]
            num_valid_pixels = valid_pixels.shape[-1]
            num_frames = hdr.required["frames"]

            if self.debug:
                print(f"[DEBUG] File {asl_obj.path.name}: {num_frames} frames, "
                      f"{num_masks} masks, {num_valid_pixels} valid slices")

            # Handle mapping of masks/valid_pixels to their respective frames
            # Repeats every 8 frames
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

    def _load_frame_modalities(self, frame_data, valid_pixels, asl_obj, frame_idx):
        """
        Compute requested modalities from raw frame data.
        """
        frame_data = frame_data.astype(np.float32)

        # Scale raw data if requested
        if self.raw_scale:
            frame_data = (frame_data - MIN_INTENSITY) / (MAX_INTENSITY - MIN_INTENSITY + 1e-8)

        # Compute Stokes parameters
        S = hf.compute_stokes(frame_data)  # (H, W, 5): s0, s1, s2, dolp, aop
        s0 = S[:, :, 0]
        s1 = S[:, :, 1]
        s2 = S[:, :, 2]

        # Normalize s0
        if self.min_max:
            s0 = (s0 - np.min(s0)) / (np.max(s0) - np.min(s0) + 1e-8)
        elif self.raw_scale:
            pass  # already in [0,1]
        else:
            # Use log scaling
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
            dolp = np.sqrt(s1**2 + s2**2)
            
            # Clip DoLP to [0, 1]
            dolp_norm = np.clip(dolp, 0, 1)
            output["dolp"] = dolp_norm
        if "aop" in self.modalities:
            aop = 0.5 * np.arctan2(s2, s1)

            # Normalize AoP from [-pi/2, pi/2] to [0, 1]
            aop_norm = (aop + np.pi / 2) / np.pi
            output["aop"] = aop_norm

        if self.compute_enhanced:
            es0, s0e1, s0e2 = hf.compute_enhanceds0(
                S, s0std=S0_STD, dolp_max=DOLP_MAX, aop_max=AOP_MAX,
                fusion_coefficient=FUSION_COEFFICIENT,
                valid_pixels=valid_pixels,
                hdr=asl_obj,
                aop_mode=self.aop_mode
            )

            output["enhanced_s0"] = es0

            if "s0e1" in self.modalities:

                # To remove extra 4th dimension
                output["s0e1"] = s0e1.squeeze(-1)
            
            if "s0e2" in self.modalities:
                output["s0e2"] = s0e2.squeeze(-1)

        # Convert to torch tensors with (C,H,W) format (C=1 per modality)
        for k in output:
            arr = output[k]
            output[k] = torch.from_numpy(arr).unsqueeze(0).float()

        if self.debug:
            print(f'[DEBUG] Frame {frame_idx}: modalities {list[output.keys()]}, shapes: {[output[k].shape for k in output]}')

        # Stack modalities and represent as a (C, H, W) tensor
        if self.stack_modalities:
            tensor = torch.cat([output[k] for k in self.modalities if k in output], dim=0)
            if self.debug:
                print(f'[DEBUG] Stacked tensor shape: {tensor.shape}')

            return tensor

        return output

    def __getitem__(self, idx):
        file_idx, frame_idx, mask_np, valid_np = self.index_map[idx]
        asl_obj = self._asl_headers[file_idx]

        # 1-indexed frame access
        frame_data, _, _ = asl_obj.get_data(frames=[frame_idx+1])
        modalities_dict = self._load_frame_modalities(frame_data, valid_np, asl_obj, frame_idx)

        # Convert mask and valid pixels to torch tensors
        mask = torch.from_numpy(mask_np).long()
        valid_pixels = torch.from_numpy(valid_np).bool()

        return modalities_dict, mask, valid_pixels
    
# if __name__ == "__main__":
#     # Read input from user + initialize placeholder vars
#     args = parse_args()
#     raw_scale = False
#     min_max = False
#     debug = False
#     stack_modalities = False

#     # Initialize list of target modalities
#     modalities = []

#     data_dir = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
#     mask_dir = Path('/home/connor/Thesis/updated_masks').glob('*.npz')

#     if args.raw_scale:
#         raw_scale = True

#     if args.min_max:
#         min_max = True

#     if args.stack_modalities:
#         stack_modalities = True

#     if args.hist_shift:
#         # Bool to track whether to compute enhanced param.
#         compute_enhanced = True
#         aop_mode = 2
#         modalities.append("enhanced_s0")

#     if args.aop_rotate:
#         compute_enhanced = True
#         aop_mode = 1
#         modalities.append("enhanced_s0")

#     if args.s0:
#         modalities.append("s0")
#     if args.s1:
#         modalities.append("s1")
#     if args.s2:
#         modalities.append('s2')
#     if args.dolp:
#         modalities.append("dolp")
#     if args.aop:
#         modalities.append("aop")
#     if args.enhanced_mixtures:
#         modalities.append("s0e1")
#         modalities.append("s0e2")

#     if args.debug:
#         debug = True

#     # Create a dataset
#     dataset = MultiModalASLDataset(
#         data_dir,
#         mask_dir,
#         modalities=modalities,
#         aop_mode= aop_mode,
#         compute_enhanced=compute_enhanced,
#         raw_scale=raw_scale,
#         min_max=min_max,
#         debug=debug,
#         stack_modalities=stack_modalities
#     )

#     # Get first sample of dataset
#     x, mask, valid = dataset[0] # x is [modalities, H, W]

    