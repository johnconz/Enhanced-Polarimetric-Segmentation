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

# Define constants
MIN_INTENSITY = 1
MAX_INTENSITY = 4094.0

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

def process_dataset(asl_dir):
    """
    Process all ASL files in the given directory.
    """
    for file in asl_dir:

        c_hdr = ASL(file)

        # FOR DEBUG PURPOSES
        #data, part, hdr = c_hdr.get_data()
        print(f"Processing header: {c_hdr}")

        # Get frames for each turntable position
        azimuth_frames = c_hdr.get_frames_by_azimuth()

        # For each azimuth position
        for azimuth, frames in azimuth_frames.items():

            # For each frame
            for frame_idx in frames:

                frame_data, _, _ = c_hdr.get_data(frames=[frame_idx])

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

                if args.aop_rotate:
                    

                    # Rotate the frame data by the AoP
                    rotated_frame = c_hdr.rotate_frame(frame_data, aop)
                    print(f"Rotated frame at azimuth {azimuth}, index {frame_idx}")



if __name__ == "__main__":

    args = parse_args()
    files = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
    process_dataset(files)