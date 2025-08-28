# ----------------------------------------------------------------------->
# Author: Connor Prikkel, Shaik Nordin Abouzahara, Bradley M. Ratliff
# Applied Sensing Lab & Vision Lab, University of Dayton
# 8/8/2025
# ----------------------------------------------------------------------->

from pathlib import Path
from ASL import ASL
import matplotlib.pyplot as plt


"""
TEST ON HEADER
"""
x = ASL(Path('/home/connor/MATLAB/data/scenario05_sensorElevation01_Mono Polarized.asl.hdr'))
data, part, hdr = x.get_data()

print('='*20)
print('Printing data: ')
print(data)
print('Printing data shape: ')
print(data.shape)

# Visualize a frame
plt.figure(figsize=(10, 7))
plt.imshow(data[:, :, 2], cmap="gray")
plt.axis(False)
plt.show()


# ----------------------------------------------------------------------->
# Additional Functionality Testing
#
# NOTE: This function is an example extension to the ASL class
# (not part of the original MATLAB code).
# It demonstrates how new methods can be easily added for custom workflows.
# ----------------------------------------------------------------------->

"""
GET ALL FRAMES FOR A TURNTABLE POSITION
"""

# Utilizes 'turntable device azimuth' metadata (in degrees).
pos_1_frames = x.get_frames_for_turntable_pos(135)
print("Frames with turntable pos 1:", pos_1_frames)

# Just load those frames
data_pos_1, part_pos_1, hdr_pos_1 = x.get_data(frames=pos_1_frames)

print('='*20)
print('Printing data: ')
print(data_pos_1)
print('Printing data shape: ')
print(data_pos_1.shape)



