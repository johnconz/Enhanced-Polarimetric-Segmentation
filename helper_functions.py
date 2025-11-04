# ----------------------------------------------------------------------->
# Authors: Connor Prikkel
# Applied Sensing Lab & Vision Lab, University of Dayton
# 8/8/2025
#
# A variety of helper functions for Stokes and custom parameter calculations.
# ----------------------------------------------------------------------->

#from skimage.restoration import denoise_bilateral # for 'imbilatfilt" MATLAB equivalent
import cv2
import numpy as np
import torch
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Limit OpenCV to 1 thread per worker process
cv2.setNumThreads(1)

# Visualize predicted vs true test mask
def visualize_masks(pred_mask, true_mask=None, color_map=None, alpha=0.5, title=None):
    """
    Visualize predicted vs. ground truth masks with color mapping.

    Args:
        pred_mask (np.ndarray or torch.Tensor): Predicted mask (H×W) with class indices.
        true_mask (np.ndarray or torch.Tensor, optional): Ground truth mask (H×W).
        color_map (dict or list, optional): Maps class index → RGB tuple (0–255).
        alpha (float): Transparency for overlay.
        title (str): Optional title for the plot.
    """
    # Convert tensors to numpy
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()
    if true_mask is not None and torch.is_tensor(true_mask):
        true_mask = true_mask.detach().cpu().numpy()

    # Default colormap (up to 10 classes)
    if color_map is None:
        color_map = {
            0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
            4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
            7: (128, 128, 128), 8: (255, 165, 0), 9: (255, 255, 255)
        }

    def map_colors(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for k, v in color_map.items():
            rgb[mask == k] = v
        return rgb

    pred_rgb = map_colors(pred_mask)

    plt.figure(figsize=(10, 5))

    if true_mask is not None:
        true_rgb = map_colors(true_mask)
        # Show side-by-side
        plt.subplot(1, 2, 1)
        plt.imshow(true_rgb)
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_rgb)
        plt.title("Prediction")
        plt.axis('off')
    else:
        plt.imshow(pred_rgb)
        plt.axis('off')

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_cutmix(mask1_before, mask1_after, mask2, color_map=None, box=None):
    """
    Visualize the result of CutMix showing the swapped region.
    mask1, mask2: (H×W) arrays or tensors
    lam: float (mix coefficient)
    box: (cx, cy, w, h) tuple indicating patch location
    """
    if torch.is_tensor(mask1_before):
        mask1_before = mask1_before.cpu().numpy()
    if torch.is_tensor(mask1_after):
        mask1_after = mask1_after.cpu().numpy()
    if torch.is_tensor(mask2):
        mask2 = mask2.cpu().numpy()

    if color_map is None:
        color_map = {
            0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
            4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
            7: (128, 128, 128), 8: (255, 165, 0), 9: (255, 255, 255)
        }

    def map_colors(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for k, v in color_map.items():
            rgb[mask == k] = v
        return rgb

    mask1_before_rgb = map_colors(mask1_before)
    mask1_after_rgb = map_colors(mask1_after)
    mask2_rgb = map_colors(mask2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(mask1_before_rgb)
    plt.title("Original Mask (Before CutMix)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask2_rgb)
    plt.title("Cut Patch From Second Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask1_after_rgb)
    plt.title("Updated Mask (After CutMix)")

    # --- Draw rectangle for the cut region ---
    if box is not None:
        cx, cy, w, h = box
        rect = plt.Rectangle((cx, cy), w, h, linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(cx, cy - 5, 'Cut region', color='red', fontsize=10, backgroundcolor='white')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def disk_structure(radius):
    """Create a disk-shaped structuring element."""
    L = int(np.ceil(2*radius + 1))
    Y, X = np.ogrid[:L, :L]
    center = radius
    selem = (X - center)**2 + (Y - center)**2 <= radius**2
    return selem

def compute_periodic_aop(aop: np.ndarray, plot_mapping=False) -> np.ndarray:
    """
    This fn. takes a set of AoP images and applies a periodic mapping to remove display discontinuities.
    NOTE: This is a conversion of the MATLAB function of the same name.
    """
    # Parameters
    ph = -3 * np.pi / 4

    # Mapping function
    paop = np.cos(2*aop + ph) + np.sin(2*aop + ph)

    if plot_mapping:
        # Define theta values
        th = np.linspace(-np.pi/2, np.pi/2, 1000)

        # Plot
        plt.figure(figsize=(7, 5))
        plt.plot(th, paop, label=r'$\cos(2\theta - \tfrac{3\pi}{4}) + \sin(2\theta - \tfrac{3\pi}{4})$')
        plt.xlabel(r'$\theta$', fontsize=12)
        plt.ylabel(r'$\cos(2\theta - \tfrac{3\pi}{4}) + \sin(2\theta - \tfrac{3\pi}{4})$', fontsize=12)
        plt.title('AoP Periodic Mapping Function', fontsize=14)
        plt.axis([-np.pi/2, np.pi/2, -np.sqrt(2), np.sqrt(2)])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    
    return paop

def circshift_aop_histmax(aop: np.ndarray, valid_pixels=None) -> np.ndarray:

    # If not given, assume all pixels are valid
    if valid_pixels is None:
        valid_pixels = np.ones_like(aop, dtype=int)
        
    idx = valid_pixels == 1

    v = np.linspace(-np.pi/2, np.pi/2, 180)

    # Histogram (counts and bin edges)
    N, edges = np.histogram(aop[idx], bins=v)

    # Bin centers
    b = 0.5 * (edges[:-1] + edges[1:])

    # Shift = center of most populated bin
    shift = b[np.argmax(N)]

    # Shift AoP
    aop_shifted = aop - shift

    if shift > 0:
        # Wrap values that go below -π/2
        rows, cols = np.where(aop - shift < -np.pi/2)
        aop_shifted[rows, cols] = aop[rows, cols] + np.pi - shift
    else:
        # Wrap values that go above +π/2
        rows, cols = np.where(aop - shift > np.pi/2)
        aop_shifted[rows, cols] = aop[rows, cols] - np.pi - shift

    return aop_shifted


def compute_enhanced_dolp(S: np.ndarray) -> np.ndarray:
    """
    Compute the enhanced Degree of Linear Polarization (DoLP) from Stokes parameters.
    NOTE: This function is a conversion of the MATLAB function of the same name.
    """
    #Compute enhanced DoLP
    dolp = S[:,:,4]
    udolp = np.sqrt(S[:,:,2]**2 + S[:,:,3]**2)
    udolp = udolp - np.min(udolp[:])
    dolpmu = np.mean(dolp)
    udolpmu = np.mean(udolp)

    edolp = np.maximum(S[:,:,4], udolp*(1.1*dolpmu/udolpmu))
    edolp = edolp - np.min(edolp[:])

    return edolp

def compute_stokes(I: np.ndarray) -> np.ndarray:
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

def compute_saopc(aop: np.ndarray, msize: int, method: float) -> np.ndarray:
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
    selem = disk_structure(radius) # Create a disk-shaped structuring element (as np array)
    mask = selem.copy()
    if msize == 3:
        mask = np.ones((3, 3), dtype=float)

    # Remove corners as a special case
    elif msize == 5: 
        mask[0, 0] = 0
        mask[0, -1] = 0
        mask[-1, 0] = 0
        mask[-1, -1] = 0

    # Apply Gaussian weights to neighborhood if enabled
    if method == 1:

        # Gaussian kernel with stddev ~ (1 + msize / 2), same as fspecial    
        g = gaussian_filter(np.zeros((msize, msize)), sigma = 1 + msize / 2)

        # Apply impulse
        g[msize // 2, msize // 2] = 1
        G = gaussian_filter(g, sigma = 1 + msize / 2)
        mask = G * mask

    # Normalize mask
    mask = mask / mask.sum()

    # Apply filtering
    cam = convolve2d(ca, mask, mode='same', boundary='symm')
    sam = convolve2d(sa, mask, mode='same', boundary='symm')
    pcm = np.real(np.sqrt(cam**2 + sam**2))

    # Check PCM for NaN's and remove
    pcm[np.isnan(pcm)] = 0

    return pcm

def imgscale(img, scaling_a = None, scaling_b = None, return_params = False):
    """
    Scale a given image based on certain input parameters.

    Inputs
    -------
    scaling_a:
    - if None, min-max scale
    - if given & scaling_b are None, interpreted as standard deviation multiplier for statistical scaling.
    - if given & scaling is also given, interpreted as lower bound for absolute scaling.

    scaling_b:
    - upper bound for absolute scaling

    Returns
    -------
    out:
    - ndarray, scaled image in range [0, 1], dtype=float64.
    
    (optional) alpha_l:
    - lower bound used for scaling.
    
    (optional) alpha_h:
    - upper bound used for scaling.
    """

    img = np.asarray(img, dtype=float)  # convert to double
    M, N = img.shape[:2]
    #Z = 1 if img.ndim == 2 else img.shape[2]

    # Min-Max Scaling
    if scaling_a is None and scaling_b is None:
        alpha_l = np.min(img)
        alpha_h = np.max(img)

    # Statistical Scaling
    elif scaling_a is not None and scaling_b is None:
        mu = np.mean(img)
        sd = np.std(img)
        alpha_l = mu - scaling_a * sd
        alpha_h = mu + scaling_a * sd

    # Absolute Scaling
    elif scaling_a is not None and scaling_b is not None:
        alpha_l = scaling_a
        alpha_h = scaling_b

    else:
        raise ValueError("Invalid parameter combination for imgscale.")

    # Saturate values
    img = np.clip(img, alpha_l, alpha_h)

    # Linear scaling to [0,1]
    out = (img - alpha_l) / (alpha_h - alpha_l)

    if return_params:
        return out, alpha_h, alpha_l
    else:
        return out
    

def compute_enhanceds0(S, s0std, dolp_max, aop_max, fusion_coefficient,
                    valid_pixels, hdr, aop_mode):
    """
    Scale s0 and denoise s1/s2, compute updated Stokes + derivative products.
    S is a Stokes array of shape (H, W, 5).
    aop_mode: 1 = geometric rotation, 2 = histogram shifting.
    NOTE: This function is a conversion of the MATLAB function of the same name.
    """

    # Handle 3D (single frame) by promoting to 4D
    if S.ndim == 3:
        S = S[:, :, :, np.newaxis]
    if S.ndim != 4:
        raise ValueError(f"S must be 4D (H, W, 5, num_frames), got {S.ndim}D")

    H, W, C, num_frames = S.shape
    if C != 5:
        raise ValueError(f"Third dimension of S must be 5 (s0, s1, s2, dolp, aop), got {C}")

    #print(f"[INFO] S.shape = {S.shape}  → num_frames = {num_frames}")

    # Initialize outputs
    shape_enhancement = np.zeros((H, W, num_frames), dtype=np.float64)
    shape_contrast_enhancement = np.zeros((H, W, num_frames), dtype=np.float64)

    for k in range(num_frames):
        SS = np.zeros((H, W, 5), dtype=np.float64)

        # Scale s0 and denoise s1/s2
        # Single threaded and SLOWER than MATLAB imbilatfilt
        SS[:, :, 0] = S[:, :, 0, k]
        #SS[:, :, 1] = denoise_bilateral(S[:, :, 1, k].astype(np.float64),
        #                                sigma_color=100,
        #                                sigma_spatial=2,
        #                                channel_axis=None)
        #SS[:, :, 2] = denoise_bilateral(S[:, :, 2, k].astype(np.float64),
        #                                sigma_color=100,
        #                                sigma_spatial=2,
        #                                channel_axis=None)

        # Try OpenCV bilateral filter (faster, multi-threaded)
        SS[:, :, 1] = cv2.bilateralFilter(S[:, :, 1, k].astype(np.float32),
                                          d=5, sigmaColor=100, sigmaSpace=2)
        SS[:, :, 2] = cv2.bilateralFilter(S[:, :, 2, k].astype(np.float32),
                                          d=5, sigmaColor=100, sigmaSpace=2)

        # Derived Stokes products
        #SS[:, :, 3] = np.sqrt((SS[:, :, 1] / SS[:, :, 0])**2 +
        #                        (SS[:, :, 2] / SS[:, :, 0])**2)
        #SS[:, :, 4] = 0.5 * np.arctan2(SS[:, :, 2], SS[:, :, 1])
        
        # Avoid division by zero
        S2_by_S0 = np.divide(SS[:, :, 2], SS[:, :, 0], out=np.zeros_like(SS[:, :, 2]), where=SS[:, :, 0]!=0)
        S1_by_S0 = np.divide(SS[:, :, 1], SS[:, :, 0], out=np.zeros_like(SS[:, :, 1]), where=SS[:, :, 0]!=0)

        SS[:, :, 3] = np.sqrt(S1_by_S0**2 + S2_by_S0**2)
        SS[:, :, 4] = 0.5 * np.arctan2(SS[:, :, 2], SS[:, :, 1])

        # Enhanced DoLP
        edolp = compute_enhanced_dolp(S[:, :, :, k])

        # Select AoP mode
        if aop_mode == 1:
            eaop = hdr.rotate_aop(SS[:, :, 1], SS[:, :, 2], k)
        elif aop_mode == 2:
            eaop = circshift_aop_histmax(SS[:, :, 4], valid_pixels)
        else:
            raise ValueError("Invalid aop_mode. Use 1 or 2.")

        saopc = compute_saopc(eaop, 3, 0)

        # Scale s0 and enhancement maps
        es0 = imgscale(SS[:, :, 0], s0std)
        aopmap = imgscale(compute_periodic_aop(eaop), -np.sqrt(2), np.sqrt(2))
        dolpmap = imgscale(edolp, 0, dolp_max)

        # Blend s0 and enhanced images
        mixture1 = imgscale(fusion_coefficient * es0 +
                            (1 - fusion_coefficient) * np.maximum(es0, dolpmap))
        mixture1 = imgscale(mixture1 ** (1 - dolpmap))

        mixture2 = imgscale(fusion_coefficient * es0 +
                            (1 - fusion_coefficient) *
                            np.maximum(es0, np.maximum(dolpmap, aop_max * aopmap)))
        mixture2 = imgscale(mixture2 ** (1 - np.maximum(dolpmap, aop_max * aopmap)))

        # Discount noisy polarimetric pixels
        shape_enhancement[:, :, k] = ((1 - saopc) * es0 + saopc * mixture1) * valid_pixels
        shape_contrast_enhancement[:, :, k] = ((1 - saopc) * es0 + saopc * mixture2) * valid_pixels

    return es0, shape_enhancement, shape_contrast_enhancement



