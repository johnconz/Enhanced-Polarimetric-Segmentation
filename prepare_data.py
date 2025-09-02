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
import sys

# Define constants
MIN_INTENSITY = 1
MAX_INTENSITY = 4094.0

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
        # The azimuth repeats every 'stride' frames
        azimuth_idx = frame_idx % stride
        mapping[frame_idx] = {
            "mask": mask_array[:, :, azimuth_idx],
            "valid_pixels": valid_pixels[:, :, frame_idx]
        }

    return mapping


def process_dataset(
    data_dir: List[Path],
    npy_dir: List[Path],
    aop_mode: str = "deg",
    compute_enhanced: bool = False,
):
    """
    Process all ASL files in the given directory.

    Loads data header, mask array, valid_pixels;
    builds a mapping, and computes stokes + enhanced parameters per frame.
    """

    for data_file, npy_file in zip(data_dir, npy_dir):
        print(f"Proccesing: {data_file.name}")

        # Load header
        c_hdr = ASL(data_file)

        # Load mask + valid_pixels array
        with np.load(npy_file) as npz:
            mask_array = npz["relabeled_masks"]    # (H, W, 8)
            valid_pixels = npz["valid_pixels"]     # (H, W, num_frames)

        # Build mapping for this dataset
        mapping = match_mask_to_frame(mask_array, valid_pixels)

        # Total data points = instances x frames
        hdr = c_hdr.get_header()
        num_frames = hdr.required["frames"]

        # For each frame
        for frame_idx in range(num_frames):

            # Load frame data
            # (Shift by 1 as python is 1-index based)
            frame_data, _, _ = c_hdr.get_data(frames=[frame_idx + 1])

            # Get valid pixels + mask for this frame
            valid_pixels = mapping[frame_idx]["valid_pixels"]
            mask = mapping[frame_idx]["mask"]

            # Need to typecast to float for Stokes computation
            frame_data = frame_data.astype(np.float32)
            
            # Assuming min = 1, max = 4094.0
            if args.raw_scale:
                raw_norm = (frame_data - MIN_INTENSITY) / (MAX_INTENSITY - MIN_INTENSITY + 1e-8)
                stokes = hf.compute_stokes(raw_norm)
            else:
                # Compute Stokes parameters for each frame
                stokes = hf.compute_stokes(frame_data)

            s0 = stokes[:, :, 0]
            s1 = stokes[:, :, 1]
            s2 = stokes[:, :, 2]

            # Normalize s0 to [0, 1]
            # OPTION 1: Use min-max normalization
            if args.min_max:
                s0_norm = (s0 - np.min(s0)) / (np.max(s0) - np.min(s0) + 1e-8)
            elif args.raw_scale:
                s0_norm = s0
            else:
                # Use logarithmic scaling first (default)
                s0_log = np.log1p(s0) # log(1 + s0)
                s0_norm = (s0_log - np.min(s0_log)) / (np.max(s0_log) - np.min(s0_log) + 1e-8)

            # Normalize s1 and s2 by s0
            s1 = s1 / (s0 + 1e-8)
            s2 = s2 / (s0 + 1e-8)

            # Compute AoP and DoLP from s1 and s2
            dolp = np.sqrt(s1**2 + s2**2)
            aop = 0.5 * np.arctan2(s2, s1)

            # Normalize AoP from [-pi/2, pi/2] to [0, 1]
            aop_norm = (aop + np.pi / 2) / np.pi

            # Clip DoLP to [0, 1]
            dolp_norm = np.clip(dolp, 0, 1)

            print(f"Normalized s0: {s0_norm}")
            print(f"Normalized DoLP: {dolp_norm}")
            print(f"Normalized AoP: {aop_norm}")

            if compute_enhanced:

                # Compute enhanced parameters
                es0, s0e1, s0e2 = hf.compute_enhanceds0(stokes, S0_STD, DOLP_MAX, AOP_MAX, FUSION_COEFFICIENT, valid_pixels, c_hdr, aop_mode)
                
                # DEBUG PRINT STATEMENTS #
                #print(f"Enhanced s0: {es0}")
                #print(f"Enhanced mixture 1: {np.nonzero(s0e1)}")
                #print(f"Enhanced mixture 2: {np.nonzero(s0e2)}")

if __name__ == "__main__":
    args = parse_args()
    data_dir = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
    mask_dir = Path('/home/connor/Thesis/updated_masks').glob('*.npz')

    if args.hist_shift:
        # Bool to track whether to compute enhanced param.
        compute_enhanced = True
        aop_mode = 2

    if args.aop_rotate:
        compute_enhanced = True
        aop_mode = 1

    process_dataset(data_dir, mask_dir, aop_mode, compute_enhanced)