# Geometrically-Enhanced Polarimetric Image Fusion

Code, data processing pipeline, and experiments accompanying my Master's thesis:  
**"Geometrically-Enhanced Polarimetric Image Fusion as an Alternative to RGB Imaging for Object Segmentation"**

Polarization is a property of light describing the orientation of its electric field relative to the light's path; polarimetric imaging captures this by measuring how light is reflected off objects. This thesis involved the creation of a custom polarimetric dataset and Python lazy loading framework to support piecewise data preprocessing. It also puts forth unique polarization products for training and evaluation of the data on a state-of-the-art state-space object segmentation model. 

---

## Thesis

The full thesis is available here:

- **OhioLINK ETD**:  
  → http://rave.ohiolink.edu/etdc/view?acc_num=dayton1764191303578868

Please reference this document for further background on polarimetric imaging and object segmentation and an exhaustive review of the dataset created for this project, as well as the metholodgy and results of this work.

---

## Polarimetric Dataset

The polarimetric dataset used in this work is publically available at:

- **Dataset repository / archive**:  
  → https://doi.org/10.26890/vfeb3620

The dataset utilizes a custom file format: **'.ASL'**--- and 4608 Stokes vector images comprising 8 different scenarios, 128 unique scene views each with 36 different illumination geometries. For more information about the dataset alongside MATLAB source code for using it, see the above link.

---

## Branches

This repository uses separate branches to isolate stable thesis code from development work.

- `main` – Final, thesis-aligned code
- `data-reader` – Standalone Python polarimetric data reader
- `dev` – Ongoing experiments and prototyping

---

## Cloning the Repository

To clone the repository and check out the **Python polarimetric data reader** branch:

```bash
git clone --branch data-reader --single-branch https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME


