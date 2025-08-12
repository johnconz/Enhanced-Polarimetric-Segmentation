# ----------------------------------------------------------------------->
# Author: Connor Prikkel
# Applied Sensing Lab, University of Dayton
# 8/8/2025
# ----------------------------------------------------------------------->

from pathlib import Path
from ASL import ASL

# Path to ASL data dir
files = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')

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


if __name__ == "__main__":
    files = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
    process_dataset(files)