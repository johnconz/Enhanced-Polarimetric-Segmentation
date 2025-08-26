# ----------------------------------------------------------------------->
# Author: Connor Prikkel
# Applied Sensing Lab, University of Dayton
# 8/8/2025
# ----------------------------------------------------------------------->

from pathlib import Path
from ASL import ASL
import argparse
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from typing import Dict, List

# Define constants
MIN_INTENSITY = 1
MAX_INTENSITY = 4094.0
COMPUTE_ENHANCED = False  # Flag for enhanced DoLP/AoP computation

# For computing enhanced Stokes parameters
S0_STD = 3
DOLP_MAX = 1.0
AOP_MAX = 0.5
FUSION_COEFFICIENT = 0.5

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
        azimuth_idx = frame_idx % stride
        mapping[frame_idx] = {
            "mask": mask_array[:, :, azimuth_idx],
            "valid_pixels": valid_pixels[:, :, frame_idx]
        }

    return mapping


def process_dataset(
    data_dir: List[Path],
    mask_dir: List[Path],
    valid_pix_files: List[Path],
    aop_mode: str = "deg"
):
    """
    Process all ASL files in the given directory.

    Loads data header, mask array, valid_pixels;
    builds a mapping, and computes stokes + enhanced parameters per frame.
    """

    for data_file, mask_file, vp_file in zip(data_dir, mask_dir, valid_pix_files):
        print(f"Proccesing: {data_file.name}")

        # Load header
        c_hdr = ASL(data_file)

        # Load mask and valid pixels
        mask_arr = np.load(mask_file) # (H, W, 8)
        with np.load(vp_file) as npz:
            valid_pixels = npz["valid_pixels"] # (H, W, num_frames)

        # Build mapping
        mapping = match_mask_to_frame(mask_arr, valid_pixels)

        frame_data, _, _ = c_hdr.get_data()
        print(f"Frame array shape: {frame_data}")

        # For each frame
        # for frame_idx in len(frame_data):

        #     # Load frame data
        #     frame_data, _, _ = c_hdr.get_data(frames=[frame_data])


        #     # Get valid pixels + mask for this frame
        #     valid_pixels = mapping[frame_idx]["valid_pixels"]
        #     mask = mapping[frame_idx]["mask"]

        #     # Need to typecast to float for Stokes computation
        #     frame_data = frame_data.astype(np.float32)
            
        #     # Assuming min = 1, max = 4094.0
        #     if args.raw_scale:
        #         raw_norm = (frame_data - MIN_INTENSITY) / (MAX_INTENSITY - MIN_INTENSITY + 1e-8)
        #         stokes = hf.compute_stokes(raw_norm)
        #     else:

        #         # Compute Stokes parameters for each frame
        #         stokes = hf.compute_stokes(frame_data)

        #     s0 = stokes[:, :, 0]
        #     s1 = stokes[:, :, 1]
        #     s2 = stokes[:, :, 2]

        #     # Normalize s0 to [0, 1]
        #     # OPTION 1: Use min-max normalization
        #     if args.min_max:
        #         s0_norm = (s0 - np.min(s0)) / (np.max(s0) - np.min(s0) + 1e-8)
        #     elif args.raw_scale:
        #         s0_norm = s0
        #     else:
        #         # Use logarithmic scaling first (default)
        #         s0_log = np.log1p(s0) # log(1 + s0)
        #         s0_norm = (s0_log - np.min(s0_log)) / (np.max(s0_log) - np.min(s0_log) + 1e-8)

        #     # Normalize s1 and s2 by s0
        #     s1 = s1 / (s0 + 1e-8)
        #     s2 = s2 / (s0 + 1e-8)

        #     # Compute AoP and DoLP from s1 and s2
        #     dolp = np.sqrt(s1**2 + s2**2)
        #     aop = 0.5 * np.arctan2(s2, s1)

        #     # Normalize AoP from [-pi/2, pi/2] to [0, 1]
        #     aop_norm = (aop + np.pi / 2) / np.pi

        #     # Clip DoLP to [0, 1]
        #     dolp_norm = np.clip(dolp, 0, 1)

        #     print(f"Normalized s0: {s0_norm}")
        #     print(f"Normalized DoLP: {dolp_norm}")
        #     print(f"Normalized AoP: {aop_norm}")

        #     # Check if computing custom parameters
        #     if args.hist_shift or args.aop_rotate:
        #         COMPUTE_ENHANCED = True

        #         if args.hist_shift:
        #             aop_mode = 2
        #         if args.aop_rotate:
        #             aop_mode = 1

        #     if COMPUTE_ENHANCED:
                
        #         exit()
        #         # Compute enhanced parameters
        #         #es0, s0e1, s0e2 = hf.compute_enhanceds0(stokes, S0_STD, DOLP_MAX, AOP_MAX, FUSION_COEFFICIENT, valid_pixels, c_hdr, aop_mode)




if __name__ == "__main__":
    args = parse_args()
    data_dir = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
    mask_dir = Path('/home/connor/Thesis/updated_masks').glob('*.npz')
    process_dataset(data_dir, mask_dir)