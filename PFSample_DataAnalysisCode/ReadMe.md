# Metasurface Spectral Analysis Pipeline

MATLAB pipeline for processing and analyzing hyperspectral transmittance data from metasurface-based biosensing experiments.

---

## What this repo does

This project takes raw spectral measurements from metasurface sensors and turns them into:

- Clean, averaged spectra per sample  
- Feature-enhanced signals (via second derivative)  
- Low-dimensional representations (via PCA)  


---

## Data format

Each sample consists of:

- 20 metasurfaces
- 11 × 11 spatial measurements per metasurface
- Spectral range: 950–1800 cm⁻¹
- Resolution: 2 cm⁻¹ (~426 points)

After processing → each sample becomes:

20 × 426 matrix (metasurface × wavelength)

---

## Pipeline overview

1. Load & average
   - Spatial averaging across each metasurface

2. Preprocessing
   - Extract spectral region of interest
   - (Optional) normalization

3. Feature enhancement
   - Savitzky–Golay 2nd derivative  
     - Polynomial order: 3  
     - Window: 15  
   - Suppresses baseline, enhances peaks

4. Analysis
   - Principal Component Analysis (PCA)
   - Visualize clustering across samples

---

## Repository structure

├── analyzeData.mlx <br>
    ├── load_and_average_sample.m <br>
    ├── process_one_sample.m <br>
    ├── select_main_island.m <br>
    ├── sg_second_derivative.m <br>
---

## How to use

### Run full pipeline
Open in MATLAB:

matlab analyzeData.mlx 

### Process one sample
matlab data = process_one_sample(sample_path); 

---

## Dataset (used in study)

- 20 ovarian cancer samples  
- 2 non-cancer controls  

> Note: Small and imbalanced dataset — results are exploratory.

---

## Example output

- PCA score plots  
- Cluster visualization  
- Enhanced spectral features  

(Add your PCA screenshot here later)

---

## Why this matters

Metasurface sensors produce rich spectral data, but raw signals are noisy and high-dimensional.

This pipeline:
- reduces noise
- highlights subtle spectral features
- enables pattern discovery across samples


