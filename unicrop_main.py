# Example script to run the UniCrop tool
# This script performs preprocessing and then runs the automated pipeline.
# Assumptions:
# - You have 'Crop_Yield_Data_challenge_2.csv' and 'unicrop_feature_mapping.csv' in the current directory.
# - The config.py and pipeline.py files are in the same directory or in PYTHONPATH.
# - Google Earth Engine account is set up for authentication.

import os
import re
import pandas as pd

INPUT_PATH = "Rice_Crop_Data_challenge.csv"
MAP_PATH   = "unicrop_feature_mapping.csv"
OUTPUT_PATH = "./"+INPUT_PATH.split(".")[0]+"_output"

OUT_CLEANED_INPUT   = "cleaned_input_table.csv"
OUT_CLEANED_MAPPING = "cleaned_feature_mapping.csv"
OUT_FETCH_PLAN      = "fetch_plan.csv"

WINDOW_MONTHS = None      # ← change to 1, 2, ... (calendar months, not rolling days)
WINDOW_MODE   = "last"    # "last" → last N calendar months; "first" → first N months

# -------- Helpers
def _clean_text(x):
    if pd.isna(x): return None
    return (str(x).strip()
            .replace("\n", " ").replace("\r", " ")
            .replace("’", "'").replace("‘", "'")
            .replace("–", "-").replace("—", "-").replace("\t", " "))

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out

def _find_first(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None

def _find_harvest_date_col(df: pd.DataFrame) -> str | None:
    col = _find_first(df, ["date", "harvest"])
    if col: return col
    for name in ["Date of harvest", "Harvest date", "date_of_harvest", "harvest_date"]:
        for c in df.columns:
            if c.strip().lower() == name.lower():
                return c
    for c in df.columns:
        if c.strip().lower() == "date":
            return c
    return None

def _clean_frequency(freq):
    if pd.isna(freq): return None
    t = _clean_text(freq).lower()
    if t in {"static","once","one-time","0","n/a"}: return 0
    if t == "none": return None
    if t in {"daily","daily/seasonal"}: return 1
    if t == "weekly": return 7
    m = re.search(r"(\d+)\s*(?:to|–|-|—)\s*(\d+)", t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return round((lo + hi) / 2)
    m = re.match(r"^(\d+)", t.replace("days","").replace("day","").replace("d","").replace("~","").strip())
    if m: return int(m.group(1))
    return None

# -------- 1) Load files
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input not found: {INPUT_PATH}")
if not os.path.exists(MAP_PATH):
    raise FileNotFoundError(f"Feature mapping not found: {MAP_PATH}")

df_in  = _normalize_cols(pd.read_csv(INPUT_PATH))
df_map = _normalize_cols(pd.read_csv(MAP_PATH))

# clean text values (column-wise; avoids applymap warnings)
df_in  = df_in.apply(lambda col: col.map(_clean_text))
df_map = df_map.apply(lambda col: col.map(_clean_text))

# -------- 2) Input: detect/standardize keys BUT KEEP ALL OTHER COLUMNS
harv_col = _find_harvest_date_col(df_in)
if harv_col is None:
    raise ValueError("Could not find a harvest date column (e.g., 'Date of harvest').")

# parse harvest date IN-PLACE (we keep the original column)
df_in[harv_col] = pd.to_datetime(df_in[harv_col], errors="coerce", dayfirst=True)

# locate lat / lon
lat_col = _find_first(df_in, ["lat"])
lon_col = _find_first(df_in, ["lon"])
if lat_col is None or lon_col is None:
    raise ValueError("Could not find latitude/longitude columns (looked for 'lat'/'lon').")

# add standardized key copies (keep originals too)
df_in["latitude"]  = pd.to_numeric(df_in[lat_col], errors="coerce")
df_in["longitude"] = pd.to_numeric(df_in[lon_col], errors="coerce")

# drop any exact 'date' column (if present) that isn't the harvest column (keeps table clean)
drop_exact_date_cols = [c for c in df_in.columns if c.strip().lower() == "date" and c != harv_col]
df_in.drop(columns=drop_exact_date_cols, inplace=True, errors="ignore")

# filter out rows without valid coordinates, but KEEP ALL original columns
df_in = df_in.dropna(subset=["latitude","longitude"]).copy()

# -------- 3) Keep LAST/FIRST N CALENDAR MONTHS + chronological order (optional)
if df_in[harv_col].notna().any() and isinstance(WINDOW_MONTHS, int) and WINDOW_MONTHS > 0:
    df_in = df_in.sort_values(harv_col)
    months = df_in[harv_col].dt.to_period("M")
    uniq_months = months.sort_values().unique()
    if WINDOW_MODE.lower() == "first":
        keep_months = set(uniq_months[:WINDOW_MONTHS])
    else:
        keep_months = set(uniq_months[-WINDOW_MONTHS:])
    before_rows = len(df_in)
    before_uniq_dates = df_in[harv_col].nunique()
    df_in = df_in[months.isin(keep_months)].copy()
    after_rows = len(df_in)
    after_uniq_dates = df_in[harv_col].nunique()
    print(f"[calendar-month filter] mode={WINDOW_MODE}, months={WINDOW_MONTHS} "
          f"| rows: {before_rows} → {after_rows}, unique dates: {before_uniq_dates} → {after_uniq_dates}")

# de-dup by keys but KEEP all other columns (first occurrence wins)
df_in = df_in.drop_duplicates(subset=[harv_col, "latitude", "longitude"])
df_in_sorted = df_in.sort_values([harv_col, "latitude", "longitude"]).reset_index(drop=True)
df_in_sorted.to_csv(OUT_CLEANED_INPUT, index=False)

# -------- 4) Mapping: normalize headers; KEEP ALL API variants
rename_map = {
    "Key Variable": "variable",
    "Variable": "variable",
    "API Parameter": "api parameter",
    "API": "api parameter",
    "Band": "api parameter",
    "Source Dataset": "source dataset",
    "Dataset (GEE/NASA/Other)": "dataset",
    "Frequency": "frequency",
    "Detailed Notes (Calculation / Derivation)": "detailed_notes",
    "Detailed Notes": "detailed_notes",
}
df_map.rename(columns={k:v for k,v in rename_map.items() if k in df_map.columns}, inplace=True)

for c in ["variable","api parameter","source dataset","dataset","frequency","detailed_notes"]:
    if c not in df_map.columns:
        df_map[c] = pd.NA

df_map["frequency"] = df_map["frequency"].apply(_clean_frequency)

dedup_keys = [c for c in ["variable","api parameter","source dataset","dataset","frequency","detailed_notes"]
              if c in df_map.columns]
df_map = df_map.drop_duplicates(subset=dedup_keys)
df_map.to_csv(OUT_CLEANED_MAPPING, index=False)

# -------- 5) Build FETCH PLAN (cartesian: input × mapping)
# Keep track of the original input columns so we can preserve them in the final order
input_cols_all = list(df_in_sorted.columns)

fetch_plan = (
    df_in_sorted.assign(_k=1)
    .merge(df_map.assign(_k=1), on="_k", how="left")
    .drop(columns="_k")
)

# drop exact duplicates only if the entire tuple is identical
fp_dedup_keys = [harv_col, "latitude", "longitude", "variable", "api parameter", "frequency", "detailed_notes"]
for opt in ["source dataset", "dataset"]:
    if opt in fetch_plan.columns:
        fp_dedup_keys.append(opt)
fetch_plan = fetch_plan.drop_duplicates(subset=fp_dedup_keys)

# store harvest date as ISO yyyy-mm-dd
fetch_plan[harv_col] = pd.to_datetime(fetch_plan[harv_col], errors="coerce").dt.strftime("%Y-%m-%d")

# -------- Column order:
# keys → ALL original input columns (except the standardized keys if duplicated) → mapping columns
mapping_cols = [c for c in ["variable","api parameter","source dataset","dataset","frequency","detailed_notes"]
                if c in fetch_plan.columns]

# ensure keys (using standardized names) are first
key_cols = [harv_col, "latitude", "longitude"]

# keep input columns (excluding key columns to avoid duplicates)
other_input_cols = [c for c in input_cols_all if c not in key_cols]

final_cols = [c for c in key_cols if c in fetch_plan.columns] \
             + [c for c in other_input_cols if c in fetch_plan.columns] \
             + mapping_cols

# add any columns we didn't account for (just in case)
final_cols += [c for c in fetch_plan.columns if c not in final_cols]

fetch_plan = fetch_plan[final_cols]

# write
fetch_plan.to_csv(OUT_FETCH_PLAN, index=False)

# -------- 6) Sanity summary
print("✅ Preprocessing complete.")
print(f"• Cleaned input  → {OUT_CLEANED_INPUT}")
print(f"• Cleaned map    → {OUT_CLEANED_MAPPING}")
print(f"• Fetch plan     → {OUT_FETCH_PLAN}")
print(f"Rows: {len(fetch_plan):,} | Cols: {len(fetch_plan.columns):,}")

# product check: (# unique date/loc) × (# mapping rows)
N_in = df_in_sorted.drop_duplicates(subset=[harv_col, "latitude", "longitude"]).shape[0]
M_map = df_map[dedup_keys].drop_duplicates().shape[0] if dedup_keys else 0
print(f"N_in (unique date/loc): {N_in}")
print(f"M_map (unique mapping rows): {M_map}")
print(f"Expected rows (N_in × M_map): {N_in * M_map}")

# Import the UniCropPipeline class (assuming it's in pipeline.py)
from pipeline import UniCropPipeline
from modeller import UniCropModeler
from modeller import ModelConfig

data_filepath = OUTPUT_PATH+"/unicrop_master_timeseries.csv"

if not os.path.exists(data_filepath):
    print(f"Error: Data file not found at '{data_filepath}'")
    print("Data will be downloaded!...")
    # Instantiate and run data downloading step
    unicrop = UniCropPipeline(project_id='glass-arcade-366520')  # Replace with your GEE project ID
    unicrop.config.output_dir = OUTPUT_PATH
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_df = unicrop.run_all(
        fetch_plan_path="fetch_plan.csv",
        f_name_suffix="trial",
        master_timeseries_csv="unicrop_master_timeseries.csv",
        columns_manifest_csv="unicrop_columns_manifest.csv"
    )
else:
    print("🚀 Starting UniCrop Complete Modeling")
    print("=" * 60)

    config = ModelConfig()
    modeler = UniCropModeler(config)

    try:
        df = modeler.load_and_validate_data(data_filepath)
        _ = modeler.comprehensive_eda(df)
        df_proc = modeler.advanced_preprocessing(df)
        df_screened, _ = modeler.statistical_screening(df_proc)
        df_eng = modeler.feature_engineering(df_screened)
        _ = modeler.train_baseline_models(df_eng)
        _ = modeler.ensemble_models(df_eng)
        _ = modeler.model_interpretability(df_eng, modeler.selected_features)

        # Export PKL + write predict_crop_yield.py
        modeler.generate_prediction_function(df_eng, modeler.selected_features)

        # Final report
        _ = modeler.generate_final_report()

        print("\n🎉 UniCrop Modeling Complete!")
        # ------------------------------------------------------------
        # SAVE VISUALISATION DATA FOR NOTEBOOK (Figures E, F, H, I)
        # ------------------------------------------------------------
        import json
        import pickle
        import numpy as np

        print("\n💾 Saving visualisation data...")

        df_vis = df_eng.copy()

        # ---------------------------------------------
        # 1. Detect target column
        # ---------------------------------------------
        target_col = None
        for c in df_vis.columns:
            if "yield" in c.lower():
                target_col = c
                break

        if target_col is None:
            raise ValueError("❌ Could not detect target (yield) column.")

        y_true = df_vis[target_col].values

        # ---------------------------------------------
        # 2. Restore coordinates
        # ---------------------------------------------
        lat_candidates = [c for c in df.columns if "lat" in c.lower()]
        lon_candidates = [c for c in df.columns if "lon" in c.lower()]

        lat_col = lat_candidates[0]
        lon_col = lon_candidates[0]

        df_vis[lat_col] = df[lat_col]
        df_vis[lon_col] = df[lon_col]

        # ---------------------------------------------
        # 3. Restore season as categorical variable
        # ---------------------------------------------
        season_candidates = [c for c in df.columns if "season" in c.lower()]
        season_col = None

        if season_candidates:
            season_flag = season_candidates[0]

            # Step 1: Extract raw values
            raw = df[season_flag]

            # Step 2: Convert possible string booleans to real booleans
            raw_bool = raw.map({
                True: True,
                False: False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                1: True,
                0: False
            })

            # If no conversion happened, fallback to heuristic:
            if raw_bool.isna().all():
                # values might be integers or mixed
                raw_bool = raw.apply(lambda x: True if x in [1, "1", "WS"] else False)

            # Step 3: Convert booleans into readable seasons
            df_vis["season"] = raw_bool.map({
                True: "Winter–Spring",
                False: "Summer–Autumn"
            })

            season_col = "season"

        # ---------------------------------------------
        # 4. Restore district as categorical variable
        # ---------------------------------------------
        district_cols = [c for c in df.columns if "district" in c.lower()]
        district_col = None

        if district_cols:
            def get_district(row):
                for c in district_cols:
                    if row[c] is True:
                        return c.replace("fp_District_", "")
                return "Other"


            df_vis["district"] = df.apply(get_district, axis=1)
            district_col = "district"

        # ---------------------------------------------
        # 5. Rebuild ENSEMBLE predictions
        # ---------------------------------------------
        print("🔧 Rebuilding ensemble predictions...")
        base_models = {}
        for k in ['LightGBM_full', 'RandomForest_full', 'SVM_full', 'ElasticNet_full']:
            if k in modeler.models:
                base_models[k.replace('_full', '')] = modeler.models[k]
        model_dict = base_models
        weights = modeler.artifacts["ensemble_weights"]

        # Ensure arrays line up
        X_vis = df_vis[modeler.selected_features].values

        preds = {}

        for mname in ["LightGBM", "RandomForest", "SVM", "ElasticNet"]:
            preds[mname] = model_dict[mname].predict(X_vis)

        # Weighted ensemble
        y_pred = (
                weights["LightGBM"] * preds["LightGBM"] +
                weights["RandomForest"] * preds["RandomForest"] +
                weights["SVM"] * preds["SVM"] +
                weights["ElasticNet"] * preds["ElasticNet"]
        )

        df_vis["y_pred"] = y_pred
        df_vis["residuals"] = y_pred - y_true

        # ---------------------------------------------
        # 6. Build dictionary for saving
        # ---------------------------------------------
        visual_dict = {
            "df": df_vis,
            "y_true": y_true,
            "y_pred": y_pred,
            "residuals": df_vis["residuals"].values,
            "selected_features": modeler.selected_features,
            "feature_families": modeler.artifacts["eda_results"]["feature_families"],
            "lat_col": lat_col,
            "lon_col": lon_col,
            "season_col": season_col,
            "district_col": district_col,
            "target_column": target_col,
            "best_model_name": "Ensemble",
            "ensemble_weights": weights,
        }

        # ---------------------------------------------
        # 7. Save PKL / CSV / JSON
        # ---------------------------------------------
        VIS_PATH = "unicrop_visualisation_data.pkl"
        CSV_PATH = "unicrop_visualisation_data.csv"
        META_PATH = "unicrop_visualisation_metadata.json"

        with open(VIS_PATH, "wb") as f:
            pickle.dump(visual_dict, f)

        df_vis.to_csv(CSV_PATH, index=False)

        metadata = {
            "lat_column": lat_col,
            "lon_column": lon_col,
            "target_column": target_col,
            "prediction_column": "y_pred",
            "season_column": season_col,
            "district_column": district_col,
            "selected_features": modeler.selected_features,
            "feature_families": modeler.artifacts["eda_results"]["feature_families"],
            "ensemble_weights": weights
        }

        with open(META_PATH, "w") as f:
            json.dump(metadata, f, indent=4)

        print("🎉 Visualisation data saved successfully!")
        print(f"   • {VIS_PATH}")
        print(f"   • {CSV_PATH}")
        print(f"   • {META_PATH}")

        print("=" * 60)
        print("Generated files:")
        print("  📊 Figures: unicrop_figures1/ (01a.., 02a.., 03_*, 04a.., 05a.., 05e, 06a.., 07.., 08.., 09*)")
        print("  🤖 Artifacts: unicrop_model_artifacts1.pkl")
        print("  🔮 Prediction function: predict_crop_yield.py")
        print("  📋 Final report: unicrop_final_report.md")

        # Quick summary
        if 'final_comparison' in modeler.artifacts:
            all_metrics = modeler.artifacts['final_comparison']
            best_model_name = min(all_metrics.keys(), key=lambda k: all_metrics[k]['RMSE'])
            best_metrics = all_metrics[best_model_name]
            print("\n📈 Pipeline Summary:")
            print(f"    Best Model: {best_model_name}")
            print(f"    Selected Features: {len(modeler.selected_features)}")
            print(f"    Final RMSE: {best_metrics.get('RMSE', 'N/A'):.4f}")

            print("\n📊 Final Model Comparison:")
            print(pd.DataFrame(all_metrics).T.round(4))
    except Exception as e:
        print(f"❌ Model failed with error: {str(e)}")
        import traceback
        traceback.print_exc()