# Metasurface Spectral Analysis Pipeline

MATLAB pipeline for processing and analyzing hyperspectral transmittance data from metasurface-based biosensing experiments.

---

## What this repo does

This project takes raw spectral measurements from metasurface sensors and turns them into:

- Clean, averaged spectra per sample that contains only the resonance region of the metasurface
- Feature-enhanced signals (via second derivative)  
- Low-dimensional representations (via PCA)  

The transmittance spectra from the metasurfaces are first loaded


  <img src="figures/transmittance" width="500"/>



Then we only select the part of the resonance (by thresholding and keeping only the top 70%) 


  <img src="figures/selected_transmittance" width="500"/>



Subsequently we perform second derivative and the data can be used for classification of patient samples



  <img src="figures/2ndDerivative" width="500"/>



---

## Data format

Each sample consists of:

- 20 metasurfaces
- 11 × 11 spatial measurements per metasurface (region determined manually for now)
- Spectral range: 950–1800 cm⁻¹
- Resolution: 2 cm⁻¹ (~426 points)

After processing → each sample becomes:

20 × 426 matrix (metasurface × wavelength) only in the resonance region of the metasurface

---

## Pipeline overview

1. Load & average
   - Spatial averaging (11x11) across each metasurface

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


## Example output

- PCA score plots  
- Cluster visualization  
- Enhanced spectral features  


