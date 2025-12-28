# 🌱 UniCrop User Manual

A unified pipeline for remote sensing feature extraction and machine learning yield modelling.

---

# Table of Contents
1. Introduction  
2. System Overview  
3. Repository Structure  
4. Installation  
5. Input Data Requirements  
6. Running UniCrop  
7. Remote Sensing Pipelines  
8. ML Modelling Pipeline  
9. Outputs  
10. Troubleshooting  
11. Extending UniCrop  
12. Citation  

---

# 1. Introduction

UniCrop is designed to support large-scale, reproducible crop yield modelling using:
- NASA POWER weather data  
- Sentinel-2 optical indices  
- MODIS vegetation and energy balance products  
- SRTM terrain features  
- Machine‑learning benchmarking  

UniCrop separates *data downloading* from *modelling* to avoid unnecessary recomputation.

---

# 2. System Overview

UniCrop runs in two phases:

### ▶ First Run
- Detects **no folder** named after the CSV  
- Downloads remote sensing features  
- Saves them  
- **Stops**

### ▶ Second Run
- Detects the folder exists  
- Skips downloading  
- Runs machine learning modelling  

This ensures efficient processing for large datasets.

---

# 3. Repository Structure

```
UniCrop/
│
├── pipeline.py
├── config.py
├── unicrop_main.py
├── datasets/
│   └── Rice_Crop_Data_challenge.csv
├── model_results/
├── <dataset_name>/
│   ├── nasa_daily.csv
│   ├── S2_MODIS_timeseries.csv
│   ├── srtm_stats.csv
└── docs/
    └── USER_MANUAL.md
```

---

# 4. Installation

## A) Conda (recommended)

```bash
conda create -n unicrop python=3.10
conda activate unicrop
pip install -r requirements.txt
```

## B) Earth Engine Auth
```bash
earthengine authenticate
```

---

# 5. Input Data Requirements

Your CSV must match:

**Rice_Crop_Data_challenge.csv**

Required columns (case-insensitive):

| Column | Purpose |
|--------|---------|
| latitude, longitude | Spatial location |
| harvest date | Center of download window |
| source dataset | e.g., NASA, S2, MOD13Q1, SRTM |
| api parameter | Which variables to fetch |
| variable | Additional info / derived features |

---

# 6. Running UniCrop

## First Run — Download Stage

```bash
python unicrop_main.py --data Rice_Crop_Data_challenge.csv
```

Creates folder:

```
Rice_Crop_Data_challenge/
```

Pipeline stops here.

---

## Second Run — Modelling Stage

```bash
python unicrop_main.py --data Rice_Crop_Data_challenge.csv
```

Steps:

1. Load merged dataset  
2. Clean / preprocess  
3. Feature engineering  
4. Feature selection  
5. Cross‑validated ML benchmarking  
6. Save results  

Outputs saved into `model_results/`

---

# 7. Remote Sensing Pipelines

## NASA POWER
- T2M, RH2M, WS2M, etc.
- Derived indices: DTR, VPD, dew point

## Sentinel‑2 Harmonized
- NDVI, EVI, LAI, SAVI, NDRE, CIredge  
- Cloud masking using SCL  
- 25‑day nearest-window fallback

## MODIS
### MOD13Q1  
NDVI, EVI, VCI

### MOD15A2H  
LAI, FPAR

### MOD16A2  
ET (Evapotranspiration)

### MOD17A2H  
GPP, NPP

## SRTM
- Elevation  
- Slope  
- Aspect  
- Hillshade  

---

# 8. ML Modelling Pipeline

UniCrop uses your student’s advanced modelling code.

### Includes:
- Missing-data imputation  
- Robust scaling  
- Polynomial and interaction features  
- Seasonal and spatial features  
- Feature selection:
  - basic  
  - mutual information  
  - model-based  
  - combined  
- Models:
  - XGBoost  
  - LightGBM  
  - Random Forest  
  - ElasticNet (optional)  
- GroupKFold cross-validation  
- Automatic result saving  

Results saved in `model_results/`.

---

# 9. Outputs

| Output | Description |
|--------|-------------|
| `<dataset>/nasa_daily.csv` | NASA daily features |
| `<dataset>/S2_MODIS_timeseries.csv` | Optical + MODIS |
| `<dataset>/srtm_stats.csv` | Terrain |
| `merged_final.csv` | Final modelling dataset |
| `model_results/metrics.json` | R², MAE, RMSE for each model |
| `model_results/*_feature_importances.csv` | Feature importances |

---

# 10. Troubleshooting

### ❌ Earth Engine auth failures
Run:
```bash
earthengine authenticate
```

### ❌ NaN errors in PolynomialFeatures
Ensure missing values imputed before feature_engineering (fixed in pipeline).

### ❌ No SRTM or S2 data
Check spelling in `source dataset` column.

### ❌ Unicode / encoding errors
Use UTF‑8 CSV files.

---

# 11. Extending UniCrop

You can add:

- More satellite products  
- Additional feature selectors  
- Deep learning models  
- Parquet/Feather export  
- API endpoints  

The modular `pipeline.py` makes extension easy.

---

# 12. Citation

```
Karakus, O., et al. (2025). UniCrop: Unified Remote Sensing and Machine Learning
Pipeline for Crop Yield Modelling. Manuscript in preparation.
```

A final citation will be provided upon publication.

