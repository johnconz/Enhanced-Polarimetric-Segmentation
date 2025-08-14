# ----------------------------------------------------------------------->
# Authors: Connor Prikkel
# Applied Sensing Lab & Vision Lab, University of Dayton
# 8/8/2025
#
# A variety of helper functions for Stokes and custom parameter calculations.
# ----------------------------------------------------------------------->

import math
from skimage.restoration import denoise_bilateral # for 'imbilatfilt" MATLAB equivalent
from skimage import data
import numpy as np
from scipy.ndimage import binary_dilation

def disk_structure(radius):
    """Create a disk-shaped structuring element."""
    L = int(np.ceil(2*radius + 1))
    Y, X = np.ogrid[:L, :L]
    center = radius
    selem = (X - center)**2 + (Y - center)**2 <= radius**2
    return selem


def enhancedDoLP(self, S: np.ndarray) -> np.ndarray:
    """
    Compute the enhanced Degree of Linear Polarization (DoLP) from Stokes parameters.
    NOTE: This function is a conversion of the MATLAB function of the same name.
    """
    #Compute enhanced DoLP
    DoLP = S[:,:,4]
    uDoLP = np.sqrt(S[:,:,2]**2 + S[:,:,3]**2)
    uDoLP = uDoLP - min(uDoLP[:])
    dolpmu = np.mean(DoLP)
    udolpmu = np.mean(uDoLP)

    eDoLP = np.max(S[:,:,4], uDoLP*(1.1*dolpmu/udolpmu))
    eDoLP = eDoLP - np.min(eDoLP[:])

    return eDoLP

def compute_stokes(self, I: np.ndarray) -> np.ndarray:
    """
    Compute the Stokes parameters from the raw polarimetric intensity data
    where I is a stack of intensity vectors [0, 45, 90, 135].
    """

    # Initialize stokes array
    H, W, _ = I.shape
    S = np.zeros((H, W, 5), dtype=I.dtype)

    S[:, :, 0] = 0.5 * np.sum(I, axis=2) # S0
    S[:, :, 1] = I[:, :, 0] - I[:, :, 2] # S1
    S[:, :, 2] = I[:, :, 1] - I[:, :, 3] # S2

    # To avoid division by zero, add a small epsilon
    epsilon = 1e-12
    denom = S[:, :, 0] + epsilon
    S[:, :, 3] = np.sqrt((S[:, :, 1] / denom)**2 + (S[:, :, 2] / denom)**2) # DoLP
    S[:, :, 4] = 0.5 * np.arctan2(S[:, :, 2], S[:, :, 1]) # AoLP

    return S

def computeSAoPC(self, aop: np.ndarray, msize: int, method: float) -> np.ndarray:
    """
    Compute the SAoPC (Spatially Adaptive Orientation of Polarization Coherence) from enhanced AoP.
    aop: AoP array
    msize: size of the neighborhood (should be odd)
    method: 0- unweighted neighborhood, 1- Gaussian weighted neighborhood
    NOTE: This function is a conversion of the MATLAB function of the same name.
    """
    
    # Compute first step of S-AoPC
    ca = np.cos(2*aop)
    sa = np.sin(2*aop)

    # Construct neighborhood
    if(np.mod(msize, 2) == 0): 
        msize = msize + 1 #Make odd if even
    radius = (msize - 1) / 2 + 1 
    selem = disk_structure(radius) # Create a disk-shaped structuring element
    mask = se.Neighborhood
    if(msize==3)
        mask = ones(3);
    elseif(msize==5) %Remove corners as a special case
        mask(1,1) = 0;
        mask(1,5) = 0;
        mask(5,1) = 0;
        mask(5,5) = 0;
    end

    %Apply Gaussian weights to neighborhood if enabled
    if(method==1)    
        G = fspecial('gaussian',msize,1+msize/2);
        mask = G.*mask;
    end

    mask = mask./sum(mask(:));
    cam = imfilter(ca,mask,'symmetric');
    sam = imfilter(sa,mask,'symmetric');
    pcm = real(sqrt( cam.^2 + sam.^2));

    %Check PCM for NaN's and remove
    pcm(isnan(pcm(:))) = 0;

def compute_enhanceds0(self, S, s0std, dolpMax, aopMax, fusionCoefficient,
                    validPixels, hdr, aopMode):
    """
    Scale s0 and denoise s1/s2, compute updated Stokes + derivative products.
    S is a Stokes array of shape (H, W, 5).
    aopMode: 1 = geometric rotation, 2 = histogram shifting.
    NOTE: This function is a conversion of the MATLAB function of the same name.
    """

    # Shape check
    if S.ndim != 4:
        raise ValueError(f"S must be 4D (H, W, 5, num_frames), got {S.ndim}D")
    H, W, C, num_frames = S.shape
    if C != 5:
        raise ValueError(f"Third dimension of S must be 5 (s0, s1, s2, dolp, aop), got {C}")

    print(f"[INFO] S.shape = {S.shape}  → num_frames = {num_frames}")

    # Initialize enhanced Stokes products
    s0e1 = np.zeros((H, W, num_frames), dtype=np.float64)
    s0e2 = np.zeros((H, W, num_frames), dtype=np.float64)

    # Intensity range (for sigma_color scaling)
    S_min, S_max = S.min(), S.max()
    intensity_range = S_max - S_min


    # For all frames
    for k in range(num_frames):
        SS = np.zeros((H, W, 5), dtype=np.float64)
        SS[:, :, 0] = S[:, :, 0, k]

        # MATLAB imbilatfilt replacement
        SS[:, :, 1] = denoise_bilateral(
            S[:, :, 1, k].astype(np.float64),
            sigma_color=100 / intensity_range,  # scale to match MATLAB DegreeOfSmoothing
            sigma_spatial=5,                    # adjust to match MATLAB SpatialSigma
            channel_axis=None
        )

        SS[:, :, 2] = denoise_bilateral(
            S[:, :, 2, k].astype(np.float64),
            sigma_color=100 / intensity_range,
            sigma_spatial=5,
            channel_axis=None
        )

        # Compute DoLP and AoP
        SS[:, :, 3] = np.sqrt((SS[:, :, 1] / SS[:, :, 0])**2 +
                            (SS[:, :, 2] / SS[:, :, 0])**2)
        SS[:, :, 4] = 0.5 * np.arctan2(SS[:, :, 2], SS[:, :, 1])

        edolp = ASL.enhancedDoLP(S)
        if aopMode == 1:
            eaop = rotateAoP(SS[:, :, 1], SS[:, :, 2], hdr, k)  # TODO: implement
        elif aopMode == 2:
            eaop = circshiftAoPHistMax(SS[:, :, 4], validPixels)  # TODO: implement
        else:
            raise ValueError("Invalid aopMode. Use 1 or 2.")

        saopc = computeSAoPC(eaop, 3, 0)  # TODO: implement

        s0 = imgscale(SS[:, :, 0], s0std)  # TODO: implement
        aopmap = imgscale(periodicAoP(eaop), -np.sqrt(2), np.sqrt(2))  # TODO: implement
        dolpmap = imgscale(edolp, 0, dolpMax)

        mixture1 = imgscale(imgscale(fusionCoefficient * s0 +
                        (1 - fusionCoefficient) * np.maximum(s0, dolpmap)) ** (1 - dolpmap))
        mixture2 = imgscale(imgscale(fusionCoefficient * s0 +
                        (1 - fusionCoefficient) * np.maximum(s0,
                        np.maximum(dolpmap, aopMax * aopmap))) **
                        (1 - np.maximum(dolpmap, aopMax * aopmap)))

        s0e1[:, :, k] = ((1 - saopc) * s0 + saopc * mixture1) * validPixels
        s0e2[:, :, k] = ((1 - saopc) * s0 + saopc * mixture2) * validPixels

    return s0, s0e1, s0e2

