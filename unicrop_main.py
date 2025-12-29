import os
import re
import pandas as pd

from source_codes.paths import INPUT_PATH, OUTPUT_PATH, MAP_PATH, OUT_CLEANED_INPUT, OUT_CLEANED_MAPPING, OUT_FETCH_PLAN, FIGURES_PATH

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
# # filter out rows without valid coordinates, but KEEP ALL original columns
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
from source_codes.pipeline import UniCropPipeline
from source_codes.modeller import UniCropModeler
from source_codes.config import ModelConfig

data_filepath = OUTPUT_PATH+"/unicrop_master_timeseries.csv"

if not os.path.exists(data_filepath):
    print(f"Error: Data file not found at '{data_filepath}'")
    print("Data will be downloaded!...")
    # Instantiate and run data downloading step
    unicrop = UniCropPipeline()  # Replace with your GEE project ID
    unicrop.config.output_dir = OUTPUT_PATH
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_df = unicrop.run_all(
        fetch_plan_path=OUT_FETCH_PLAN,
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

        print("=" * 60)
        print("Generated files:")
        print("  📊 Figures: unicrop_figures/ (01a.., 02a.., 03_*, 04a.., 05a.., 05e, 06a.., 07.., 08.., 09*)")
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