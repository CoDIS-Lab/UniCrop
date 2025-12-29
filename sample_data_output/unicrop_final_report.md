
# UniCrop Prediction Model Report

## Abstract
This work presents UniCrop, a universal and model-independent model that integrates meteorological, satellite, SAR, soil, and topographic data. It emphasizes robust preprocessing (winsorization), transparent diagnostics (EDA, SHAP), a model-agnostic feature selection (composite relevance + mRMR), and a constrained ensemble.

## Acknowledgements
We thank the UniCrop contributors and partners who provided data access and agronomic guidance.

## Executive Summary
This report summarizes a production-hardened, generalizable base with strict schema enforcement at inference, family-aware imputations, and drift checks (PSI). The pipeline produces complete QC figures and reporting.

## Data Overview
- **Dataset Size**: 531 samples × 152 features
- **Target Variable**: fp_yield (kg / ha)
- **Cross-Validation Strategy**: 5-fold KFold (shuffle)

## Data Quality Assessment

### Target Distribution
- Mean Yield: 9270.75 kg/ha
- Standard Deviation: 3644.49 kg/ha
- Skewness: -0.747
- Range: 0 - 15147 kg/ha

### Missing Data Analysis
- Vegetation indices showed notable missing values with seasonal patterning (see figures).
- Applied family-aware imputation strategies.

## Baseline Implementation (Model Development)
- Outlier handling via winsorization; robust scaling.
- Family-aware imputers (meteorology: iterative imputer; soil: median). **VI uses KNN imputation with context (district/season) and median fallback.**
- Model-independent feature selection selecting exactly 15 features.
- Correlation-based screening and basic significance checks.
- SLSQP non-negative ensemble retained only if it beats the best single model.

### Statistical Screening
- Initial features: 151
- Selected (stable) features: 15
- Dropped features: 38

### Engineered Features
- GDD_base10
- heat_days_35
- chill_nights_15
- EVI_amplitude
- VV_VH_ratio
- VV_texture
- interaction_ALLSKY_SFC_SW_DWN_mean_RH2M_min
- interaction_ALLSKY_SFC_SW_DWN_mean_DTR_mean
- interaction_ALLSKY_SFC_SW_DWN_mean_T2M_MAX_mean
- interaction_RH2M_min_DTR_mean
- interaction_RH2M_min_T2M_MAX_mean
- interaction_DTR_mean_T2M_MAX_mean

### Final Selected 15 Features
- DTR_mean
- Fpar_500m_mean
- RVI_max
- ocd_5-15cm_mean_min
- slope_stdDev
- DEW POINT TEMPERATURE_min
- bdod_0-5cm_mean_mean
- interaction_ALLSKY_SFC_SW_DWN_mean_T2M_MAX_mean
- ocd_0-5cm_mean_max
- ALLSKY_SFC_SW_DWN_mean
- fp_Harvest Area
- nitrogen_0-5cm_mean_min
- RH2M_max
- RH2M_mean
- DTR_min

## Cross-Validation Results
| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|----|------|
| LightGBM | 1243.0463 | 931.5209 | 0.8834 | 10.05% |
| RandomForest | 1310.4038 | 928.1376 | 0.8705 | 10.01% |
| SVM | 2450.1911 | 1764.1476 | 0.5472 | 19.03% |
| ElasticNet | 2463.0636 | 1876.6553 | 0.5424 | 20.24% |
| Ensemble | 1253.1716 | 917.9587 | 0.8815 | 9.89% |


### Best Model Performance
- **Best Model**: LightGBM
- **RMSE**: 1243.0463 kg/ha
- **R²**: 0.8834


## Recommendations
- Maintain meteorological data quality (VPD, temperature, solar).
- Improve VI coverage during SA season.
- Monitor PSI on top features and re-train if PSI > 0.2 or performance degrades >10%.

---
*Report generated on 2025-12-28 23:18:29*
