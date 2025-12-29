# README – FOR NEW DATA USAGE

This document explains how to use **UniCrop** with a **new user-provided dataset**, including how to configure paths, settings, data downloading, and modelling.

---

## 1. Input Data Requirements

Users must provide a **CSV dataset** that contains, at minimum:

- **Latitude** (decimal degrees)
- **Longitude** (decimal degrees)
- **Yield value**  
  - Can be in `kg/ha`, `t/ha`, or any other continuous numeric unit  
  - The unit must be consistent across the dataset  

Optional but strongly recommended columns:

- **Harvest date or harvest year**
- **Season identifier** (e.g. `season`, `year`)
- **Spatial identifier** (e.g. `district`, `region`, `field_id`)

UniCrop does **not** require manual feature engineering in the input data. All environmental and remote-sensing features are generated automatically.

---

## 2. Step 1 – Configure Paths (`paths.py`)

The **first file to configure is `paths.py`**.

This file defines where UniCrop reads input data from and where outputs are written.

### Key principle
> **Only setting the input filename is sufficient** — UniCrop will automatically create the required folder structure.

Example:
```python
filename = "maize_ES_with_latlon.csv"
```

UniCrop will automatically create other required folder structures. 

---

## 3. Configure UniCrop (`config.py`)

The **second file to configure is `config.py`**. This file controls both the **modelling behaviour** and the **data download pipelines** used by UniCrop.

Only a small number of variables need to be set for a new dataset to work.

---

### 3.1 Model Configuration (`ModelConfig` class)

The `ModelConfig` class defines how UniCrop performs feature selection and modelling.  
For a new dataset, the following variables **must be set correctly**:

- **`target_col`**  
  Name of the yield variable in the dataset.  
  The target must be numeric and represent yield in `kg/ha`, `t/ha`, or another consistent unit.

- **`min_features`**  
  Minimum number of features required for model training.  
  If fewer features are available after preprocessing, modelling is skipped.

- **`cat_cols`**  
  List of categorical columns used for encoding and grouping (e.g. season, district).

- **`season_col`**  
  Column used for temporal grouping (e.g. harvest year or season).

- **`district_col`**  
  Column used for spatial grouping (e.g. district, region, field ID).

Example:
```python
class ModelConfig:
    target_col = "yield_kg_ha"
    min_features = 10
    cat_cols = ["season", "district"]
    season_col = "season"
    district_col = "district"
```

---

### 3.2 UniCrop Configuration (`UniCropConfig` class)

The `UniCropConfig` class defines **global settings** used by UniCrop’s data acquisition and feature construction pipelines. These settings control how external data sources are accessed and how raw input columns are prioritised.

For a new dataset, the following variables are the most important to configure:

- **`project_id`**  
  Google Earth Engine project ID used to initialise GEE.  
  This is required for downloading data from Sentinel-2, MODIS, and SRTM.  
  Users must replace this value with their own valid GEE project ID.

- **`priority_columns`**  
  A list of column names from the raw input dataset that UniCrop should always preserve and propagate through the pipeline.  
  These typically include:
  - latitude  
  - longitude  
  - target yield variable  
  - season or harvest year  
  - spatial grouping variable (e.g. district or region)

Example:
```python
class UniCropConfig:
    project_id = "your-gee-project-id"
    priority_columns = [
        "latitude",
        "longitude",
        "yield_kg_ha",
        "season",
        "district"
    ]
```

Once these variables are set, UniCrop has all the information required to initialise external data sources and begin data downloading.

---

## 4. Running UniCrop

UniCrop is executed using a single command:
```python
python unicrop_main.py
```

The pipeline automatically determines whether data must be downloaded or whether modelling can proceed.

---

### 4.1 First Execution – Data Download

When `python unicrop_main.py` is executed for the **first time** on a new dataset, UniCrop performs a check to determine whether data has already been downloaded.

- UniCrop looks for a dataset-specific output folder (named after the input CSV file, without the `.csv` extension)
- If this folder **does not exist**, UniCrop assumes that no external data has been downloaded yet

In this case, UniCrop:

- Initiates the **data downloading stage only**
- Retrieves environmental and remote-sensing data from:
  - NASA POWER (meteorological variables)
  - Sentinel-2 (spectral indices)
  - MODIS (vegetation, evapotranspiration, productivity)
  - Sentinel-1 (radar reflection for different polarisations)
  - SRTM (terrain variables)
  - ERA5 (Additional Meteorological variables)
  - SoilGrids (Soil-related features)
- Stores all downloaded and processed features in the dataset-specific folder

After the download stage completes, the program **terminates automatically**.  
No feature selection or modelling is performed during this first execution.

This design ensures a clear separation between **data acquisition** and **model experimentation**.

---

### 4.2 Subsequent Executions – Modelling and Benchmarking

If `unicrop_main.py` is executed again and UniCrop detects that the dataset-specific folder **already exists**, it assumes that all required external data has been downloaded.

In this case, UniCrop:

- Skips the data downloading stage entirely
- Loads the previously downloaded features
- Performs feature validation, preprocessing, and selection
- Trains baseline yield prediction models
- Benchmarks model performance using cross-validation and robustness analyses

This allows users to:

- Re-run modelling experiments multiple times
- Adjust modelling parameters without re-downloading data
- Focus on evaluation, diagnostics, and interpretation

---

## 5. Customising Data Download Sources

Users may customise parameters of existing data sources by editing the relevant configuration classes in `config.py`.

Examples of configurable parameters include:

- Temporal aggregation windows
- Date tolerances around harvest
- Cloud coverage thresholds (Sentinel-2)
- Spatial buffer sizes
- Lists of requested variables

These changes affect **future data downloads only**.  
If parameters are changed after data has already been downloaded, users should delete the dataset-specific folder to force a fresh download.

---

## 6. Extending UniCrop with New Features or Data Sources

### 6.1 Adding New Features from Existing Sources

To add additional variables from existing data sources:

1. Update the relevant configuration sections in `config.py`
2. Register the new feature definitions in:
   ```text
   source_files/unicrop_feature_mapping.csv
   ```

No changes to unicrop_main.py are required.

---

### 6.2 Adding a New Data Source (Advanced)

Advanced users can extend UniCrop with new data sources (e.g. Sentinel-3) by:
 - Defining a new configuration class in config.py
 - Implementing the corresponding data pipeline logic in pipeline.py
 - Registering feature mappings in:
   ```python
   source_files/unicrop_feature_mapping.csv
   ```

This modular architecture allows UniCrop to be extended without altering the core workflow.

---

## 7. Summary Workflow

  1. Prepare a dataset with latitude, longitude, and yield
  2. Set dataset filename in paths.py
  3. Configure ModelConfig and UniCropConfig in config.py
  4. Run: python unicrop_main.py
     ```text
     ├─ First run  → data download
     └─ Subsequent runs → modelling and benchmarking
     ```

UniCrop is designed to support reproducible, extensible, and transparent crop-yield analysis using both public and private datasets.
