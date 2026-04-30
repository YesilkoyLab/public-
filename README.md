This repository contains the data processing and analysis pipeline used for CaF₂ substrate–based mid-infrared (mid-IR) hyperspectral imaging in the fix/freeze (FF vs. FFPE) tissue study.

The code is designed to preprocess and analyze spectral data acquired from a QCL-based MIRSI system, with a focus on improving spectral quality and enabling reproducible comparison between fresh frozen (FF) and formalin-fixed paraffin-embedded (FFPE) tissues.

Key Features
CaF₂ background correction for substrate-induced spectral artifacts
Automated tissue masking using Otsu thresholding
Rubberband baseline correction for spectral normalization
Spectral filtering and quality control (SNR / peak-based selection)
Second-derivative processing (Savitzky–Golay) for enhanced band resolution
Extraction and visualization of mean ± standard deviation spectra
Support for downstream statistical analysis (LDA, PCA, clustering)
Application

This pipeline was developed to:

Reduce substrate and preparation-induced variability
Identify fixation-related spectral features (e.g., ~1026 cm⁻¹ band)
Enable robust biochemical comparison between FF and FFPE tissues
Reproducibility

All preprocessing steps and parameters are explicitly defined to ensure reproducibility of spectral analysis workflows used in the associated manuscript.


Zheng T, Adi W, Campagnola PJ, Yesilkoy F. Fix or Freeze? Spectral Differences Arising from Tissue Preparation in Chemical Imaging. bioRxiv [Preprint]. 2025 Nov 19:2025.11.19.689284. doi: 10.1101/2025.11.19.689284. PMID: 41332745; PMCID: PMC12667749.
