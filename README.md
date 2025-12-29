# 🌾 UniCrop: A Universal Data Pipeline for Crop Yield Modelling

UniCrop is a **configuration-driven, universal data pipeline** designed to automate the construction of **analysis-ready environmental datasets** for crop yield modelling.  
Given field locations, dates, and a declarative feature specification, UniCrop automatically retrieves, harmonises, engineers, and selects predictors from **multi-source satellite, climate, soil, and topographic data**.

UniCrop focuses on **data engineering and reproducibility**, rather than proposing new machine-learning algorithms, enabling scalable and transparent crop yield modelling across regions and crops.

---

## 📌 Key Features

- **Universal & reusable** pipeline configurable for different crops, regions, and time windows
- **Multi-source data integration**:
  - Sentinel-2 (optical remote sensing)
  - Sentinel-1 (SAR backscatter)
  - MODIS vegetation products
  - ERA5-Land climate reanalysis
  - NASA POWER agro-climatology
  - SoilGrids soil properties
  - SRTM topography
- **Automated data harmonisation**:
  - Temporal alignment
  - Spatial aggregation
  - Provenance tracking
- **Agronomic feature engineering**:
  - Growing Degree Days (GDD)
  - Vegetation dynamics
  - SAR texture metrics
  - Soil–climate interaction features
- **Statistical feature selection**:
  - Near-zero variance filtering
  - High-correlation pruning
  - Minimum Redundancy Maximum Relevance (mRMR)
- **Baseline modelling & interpretability**:
  - LightGBM, Random Forest, SVR, ElasticNet
  - Constrained ensemble modelling
  - SHAP-based interpretability

---

## 🧠 Design Philosophy

UniCrop separates **data specification** from **data implementation**.

All required environmental variables are defined in a human-readable **feature mapping file**, allowing users to adapt the pipeline to new crops or regions **without modifying code**. This design promotes portability, reproducibility, and scalability.

---

## 📂 Repository Structure
```
unicrop/
│
├── unicrop_main.py # Main pipeline execution script
├── requirements.txt # Python package details
├── requirements_optional.txt # Optional package imports
├── README - FOR NEW DATA USAGE.md
│
├── source_codes/
│ ├── pipeline.py # Data acquisition and harmonisation
│ ├── modeller.py # Feature engineering, selection, modelling
│ ├── config.py # Pipeline and model configuration
│ ├── paths.py # Folder details for data, sources, etc.
│ └── sources.py # Additional source codes
│
├── data/
│ └── sample_data.csv
│
├── source_files/
│ ├── cleaned_feature_mapping.csv # Declarative feature specification
│ ├── cleaned_input_table.csv
│ ├── unicrop_feature_mapping.csv
│ └── fetch_plan.csv
│
├── sample_data_output/
│ ├── unicrop_master_timeseries.csv
│ ├── unicrop_columns_manifest.csv
│ ├── unicrop_model_artifacts1.pkl
│ ├── unicrop_final_report.md
│ ├── unicrop_figures/ # Folder storing figures saved from sample_data.csv modelling
│ │ └── ...
│
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- Python ≥ 3.9  
- Google Earth Engine account (for satellite data access)

Install dependencies:

```bash
pip install -r requirements.txt
```

Authenticate Google Earth Engine (once):

```bash
earthengine authenticate
```

---

### 2️⃣ Configure Features

Edit `unicrop_feature_mapping.csv` to define:
 - variable names
 - data sources
 - API parameters
 - optional derivation rules

Each row corresponds to one environmental variable.

---

### 3️⃣ Run the Pipeline

```bash
python unicrop_main.py
```

This will:
 - Downloading Stage (runs only ONCE for a new dataset)
   - Clean and validate field-level input data
   - Generate an automated fetch plan
   - Download and harmonise multi-source environmental data
   - Engineer agronomic features
 - Modelling Stage
   - Perform statistical screening and mRMR feature selection
   - Train baseline models and ensemble
   - Export modelling artefacts and visualisation data

*Currently, the folders include downloaded data for the `sample_data.csv`. When users run the script above, it will bypass **the Downloading Stage** above, and only run **the Modelling Stage** for performance and prediction outputs.*

---

## 📊 Outputs

Key outputs include:
 - `unicrop_master_timeseries.csv` --> Harmonised multi-source dataset before feature selection
 - `unicrop_model_artifacts1.pkl` --> Trained models, selected features, feature families, ensemble weights
 - `unicrop_final_report.md` --> Summary of modelling results

---

## 📈 Case Study

### Public Crop Yield Case Study (Spain – Maize)

For the open-source release on GitHub, UniCrop is demonstrated using a **publicly available maize yield dataset from Spain**, sourced from the Wageningen University & Research (WUR) AI sample data repository:

🔗 https://github.com/WUR-AI/sample_data/tree/main

The dataset contains annual maize yield observations aggregated at the regional level, along with location identifiers that can be linked to geographic coordinates. To align with UniCrop’s temporal modelling assumptions and satellite data availability, we **subsample the dataset to include harvest years from 2010 onwards**. The processed data used in this repository is provided in the `data/` directory.

### Purpose of the Case Study

This case study demonstrates that:

- UniCrop can be executed entirely using **public, non-proprietary agricultural datasets**
- Annual (year-level) harvest information can be integrated using UniCrop’s **date-anchoring strategy**
- Automated data pipelines produce **consistent and interpretable environmental predictors** from NASA POWER, Sentinel-2, MODIS, and SRTM
- The resulting features support **robust baseline yield modelling** without manual data engineering

### Scope and Limitations

The Spain maize example is intended as a **methodological demonstration**, not as a claim of state-of-the-art crop yield prediction performance. Model accuracy depends on data availability, spatial resolution, and management information, which may be limited in public datasets.

Nevertheless, the case study highlights UniCrop’s key strengths:

- Reproducible data acquisition  
- Transparent feature construction  
- Modular modelling and benchmarking  
- Suitability for comparative and exploratory crop-yield analysis  

---

## 📄 Related Publication and Citation

If you use UniCrop in your research, please cite:

**UniCrop: A Universal, Multi-Source Data Engineering Pipeline for Scalable Crop Yield Prediction**  
E. Khidirova, & O. Karakus, *arXiv preprint*, 2025.

### BibTeX
```bibtex
@article{karakus2025unicrop,
  title   = {UniCrop: A Universal, Multi-Source Data Engineering Pipeline for ScalableCrop Yield Prediction},
  author  = {Khidirova, Emiliya, and Karakus, Oktay},
  journal = {arXiv preprint arXiv:250X.XXXXX},
  year    = {2025}
}
```

---

## ⚠️ Scope and Limitations

 - UniCrop does not propose new machine-learning algorithms
 - Model performance depends on input data quality
 - Satellite data availability may vary by region and season
 - UniCrop is intended as a data engineering foundation for downstream modelling and analysis.

---

## 🤝 Contributing

Contributions are welcome, particularly:
 - additional feature mappings
 - support for new data sources
 - enhancements to feature engineering modules

Please open an issue or submit a pull request.

---

## 📬 Contact

Oktay Karakus

Cardiff University

✉️ karakuso@cardiff.ac.uk

---

## 👩‍💻 Development and Contributions

This codebase was developed by **Emiliya Khidirova** as part of her **MSc dissertation at Cardiff University (2025)**.

- **All core coding, implementation, and pipeline development** were carried out by *Emiliya Khidirova*.
- The study was **supervised by Dr. Oktay Karakus**, who provided research guidance, conceptual oversight, and feedback.
- Dr. Karakus also contributed **minor cosmetic refinements** to the final published data products and code structure in preparation for public release.

This repository reflects the original MSc research work, released in the interest of transparency, reproducibility, and community reuse.

---

## 🏁 License

This project is released under the MIT License.


