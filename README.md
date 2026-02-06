# Geometrically-Enhanced Polarimetric Image Fusion and Segmentation

Code, data processing pipeline, and experiments accompanying my Master's thesis:  
**Geometrically-Enhanced Polarimetric Image Fusion as an Alternative to RGB Imaging for Object Segmentation.**

## Summary

Polarization is a property of light describing the orientation of its electric field relative to the light's path; polarimetric imaging captures this by measuring how light is reflected off objects. 

This thesis involved the creation of a custom polarimetric dataset and Python lazy loading framework to support piecewise data preprocessing. It also puts forth unique polarization products for training and evaluation of the data on a state-of-the-art state-space object segmentation model. 

### Thesis

The full thesis is available here:

- **OhioLINK ETD**:  
  → http://rave.ohiolink.edu/etdc/view?acc_num=dayton1764191303578868

Please reference this document for further background on polarimetric imaging and object segmentation and an exhaustive review of the this work's dataset, metholodgy, and results.

---

## Polarimetric Dataset

![GIF showing s0, DoLP, and AoP images for a variety of turntable positions for one scene view.](https://ecommons.udayton.edu/assets/md5images/5f7155100f2294084edc88194e2a09dc.gif)

The polarimetric dataset used in this work is publically available at:

- **Dataset Archive**:  
  → https://doi.org/10.26890/vfeb3620

The dataset utilizes a custom `.ASL` file format. It includes 4608 Stokes vector images comprising 8 different scenarios, 128 unique scene views each with 36 different illumination geometries. 

For more information about the dataset alongside MATLAB source code for using it, see the above link.

### Using the Dataset

**Step 1**: Download the dataset (Polarimetric Data: Scenario 01 ... Scenario 08) from https://doi.org/10.26890/vfeb3620.

**Step 2**: 
* If using **MATLAB**: install the `ASL File Reader` and `Utility functions` from https://ecommons.udayton.edu/appliedsensinglab_sourcecode/.
* If using **Python**: clone the `data-reader` branch of this repository by:

```bash
git clone --branch data-reader --single-branch https://github.com/johnconz/Enhanced-Polarimetric-Segmentation.git
cd Enhanced-Polarimetric-Segmenation
```
--

## Project Structure

This repository uses two separate branches to isolate stable thesis code utilizing a custom environment with many packages from the base Python framework and reader.

- `main` – Final thesis code
- `data-reader` – Standalone Python polarimetric data reader and lazy loading framework

## Results

The dataset was evaluated on a variation of the **Ultralight VM-UNet** architecture:

* Paper: https://doi.org/10.48550/arXiv.2403.20035
* GitHub: https://github.com/wurenkai/UltraLight-VM-UNet

... under two paradigms: RGB-like 3-channel fusions and variable modality tensors.

It was found that through both methods, the custom products **outperformed** traditional polarimetric ones and **achieved effective per-class IoU**. 

These results emphasize polarimetric imaging as a promising path forward in place of traditional RGB data for certain tasks. I plan to complete further experimentation and directly compare polarimetric versus visible-band training and evaluation, including possible testing with other sensing bands like SWIR.


