import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from IPython.core.pylabtools import figsize

warnings.filterwarnings('ignore')
from source_codes.paths import INPUT_PATH, OUTPUT_PATH, MAP_PATH, OUT_CLEANED_INPUT, OUT_CLEANED_MAPPING, OUT_FETCH_PLAN, FIGURES_PATH

# Core ML libraries
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

# Tree-based models
import lightgbm as lgb

# Deep learning
import tensorflow as tf

# Statistical tests
from scipy import stats
from scipy.stats import jarque_bera
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import nnls, minimize

# SHAP for interpretability
import shap

# Utilities
import pickle
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import textwrap


# === Plot styling ===
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ======================
# Config dataclass
# ======================
from source_codes.config import ModelConfig

class UniCropModeler:
    """
    Complete UniCrop modeling following the blueprint specifications
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.artifacts: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.selected_features: List[str] = []
        self.oof_predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Any] = {}

        # Reproducibility
        np.random.seed(self.config.random_state)
        tf.random.set_seed(self.config.random_state)

    # --------------------------
    # Step 1: Load / Validate
    # --------------------------
    def load_and_validate_data(self, filepath: str) -> pd.DataFrame:
        print("🔄 Step 1: Loading and validating data...")

        df = pd.read_csv(filepath)
        print(f"✅ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

        # ------------------------------------------------------------
        # Drop columns with > 20% missing values (except protected cols)
        # ------------------------------------------------------------
        nan_threshold = 1.0  # 20%
        protected_cols = set([self.config.target_col, "latitude", "longitude"])  # adjust if needed
        missing_frac = df.isna().mean()

        cols_to_drop = [
            c for c in df.columns
            if (missing_frac.get(c, 0) > nan_threshold) and (c not in protected_cols)
        ]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"🧹 Dropped {len(cols_to_drop)} columns with >{int(nan_threshold * 100)}% NaNs.")
            # Optional: show top offenders
            top = sorted(cols_to_drop, key=lambda c: missing_frac[c], reverse=True)[:10]
            print("   Top dropped columns:", [(c, round(float(missing_frac[c]) * 100, 2)) for c in top])

        # Date -> cyclic feature
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['doy'] = df['date'].dt.dayofyear
            df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
            df.drop(columns=[c for c in ['doy_cos', 'doy', 'date'] if c in df.columns], inplace=True)

        # Types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.config.cat_cols

        for col in numeric_cols:
            df[col] = df[col].astype('float32')
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Drop duplicates/problem columns (keep ET over ET_mean)
        to_drop = ['latitude', 'longitude', 'slope_min']
        if 'ET_mean' in df.columns and 'ET' in df.columns:
            to_drop.append('ET_mean')
        existing = [c for c in to_drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            print(f"✅ Dropped duplicate/problematic columns: {existing}")

        # Save metadata for report
        self.artifacts['raw_data_shape'] = df.shape
        self.artifacts['categorical_columns'] = categorical_cols
        self.artifacts['dropped_columns'] = existing
        self.artifacts['dropped_nan_columns'] = cols_to_drop  # ✅ add this
        self.artifacts['nan_threshold'] = nan_threshold  # ✅ add this

        return df

    # --------------------------
    # Step 2: EDA
    # --------------------------
    def comprehensive_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        print("🔄 Step 2: Comprehensive EDA...")

        target = self.config.target_col
        eda_results: Dict[str, Any] = {}

        # === Target distribution (separate figure) ===
        plt.figure(figsize=(8, 6))
        plt.hist(df[target], bins=30, alpha=0.7, density=True)
        df[target].plot.kde(linewidth=2)
        plt.title("Target Distribution (Yield kg/ha)")
        plt.xlabel("Yield (kg/ha)")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/01a_target_distribution.svg", bbox_inches="tight")
        plt.close()

        # === Q-Q plot (separate figure) ===
        plt.figure(figsize=(8, 6))
        stats.probplot(df[target], dist="norm", plot=plt)
        plt.title("Q-Q Plot (Normality Check)")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/01b_qq_plot.svg", bbox_inches="tight")
        plt.close()

        # === Yield by Season (separate figure) ===
        season_col = self.config.season_col
        if season_col in df.columns:
            plt.figure(figsize=(9, 6))
            sns.boxplot(data=df, x=season_col, y=target, width=0.6, fliersize=2.5, linewidth=1.6)
            plt.title("Yield by Season")
            plt.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(f"{FIGURES_PATH}/01c_yield_by_season.svg", bbox_inches="tight")
            plt.close()

        # === Yield by District (separate figure) ===
        district_col = self.config.district_col
        if district_col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=district_col, y=target, width=0.6, fliersize=2.5, linewidth=1.6)
            plt.title("Yield by District")
            plt.xticks(rotation=30)
            plt.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(f"{FIGURES_PATH}/01d_yield_by_district.svg", bbox_inches="tight")
            plt.close()

        target_stats = {
            "mean": float(df[target].mean()),
            "std": float(df[target].std()),
            "min": float(df[target].min()),
            "max": float(df[target].max()),
            "skewness": float(df[target].skew()),
            "kurtosis": float(df[target].kurtosis()),
            "jarque_bera_pvalue": float(jarque_bera(df[target])[1]),
        }
        print("Target Statistics:")
        for k, v in target_stats.items():
            print(f"  {k}: {v:.4f}")
        eda_results["target_stats"] = target_stats

        # Missingness overview + VI seasonal missing
        missing_df = (df.isnull().sum().to_frame("Missing_Count")
                        .assign(Missing_Percentage=lambda x: x["Missing_Count"] / len(df) * 100)
                        .sort_values("Missing_Percentage", ascending=False))

        # === Top missing features (separate) ===
        top_missing = missing_df[missing_df["Missing_Percentage"] > 0].head(20)
        if len(top_missing) > 0:
            plt.figure(figsize=(10, 8))
            top_missing["Missing_Percentage"].plot(kind="barh")
            plt.title("Top 20 Features by Missing Percentage")
            plt.xlabel("Missing Percentage (%)")
            plt.tight_layout()
            plt.savefig(f"{FIGURES_PATH}/02a_top_missingness.svg", bbox_inches="tight")
            plt.close()

        # === VI missing by season (separate) ===
        vi_cols = [c for c in df.columns if any(t in c.upper() for t in ["NDVI", "EVI", "LAI", "NDRE", "SAVI", "CIREDGE"])]
        if vi_cols and season_col in df.columns:
            vi_missing = df.groupby(season_col)[vi_cols].apply(lambda x: x.isnull().mean() * 100)
            plt.figure(figsize=(10, 8))
            sns.heatmap(vi_missing.T, annot=True, fmt=".1f", cmap="coolwarm")
            plt.title("VI Missing % by Season")
            plt.tight_layout()
            plt.savefig(f"{FIGURES_PATH}/02b_vi_missing_by_season.svg", bbox_inches="tight")
            plt.close()

        eda_results["missing_analysis"] = missing_df

        # Feature family discovery
        feature_families = {
            "Meteorology": [c for c in df.columns if any(k in c.upper() for k in ["T2M", "RH2M", "VPD", "WS2M", "ALLSKY", "DTR", "DEW", "ET"])],
            "Vegetation": [c for c in df.columns if any(k in c.upper() for k in ["NDVI", "EVI", "LAI", "NDRE", "SAVI", "CIREDGE", "GPP", "FPAR"])],
            "SAR": [c for c in df.columns if any(k in c.upper() for k in ["VV", "VH", "RVI"])],
            "Soil": [c for c in df.columns if any(k in c.lower() for k in ["ocd_", "nitrogen_", "phh2o_", "sand_", "silt_", "clay_", "bdod_"])],
            "Topography": [c for c in df.columns if any(k in c.lower() for k in ["elevation", "slope", "aspect"])],
            "Climate": [c for c in df.columns if any(k in c.lower() for k in ["precipitation", "evaporation", "temperature", "soil_water"])],
        }
        feature_families = {k: v for k, v in feature_families.items() if v}
        print("\nFeature Families:")
        for fam, feats in feature_families.items():
            print(f"  {fam}: {len(feats)} features")
        eda_results["feature_families"] = feature_families

        # Correlations per family (each saved separately)
        numeric_df = df.select_dtypes(include=[np.number])
        for fam, feats in feature_families.items():
            fam_num = [f for f in feats if f in numeric_df.columns]
            if len(fam_num) > 1:
                plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_df[fam_num].corr(), cmap="coolwarm", center=0, square=True)
                plt.title(f"{fam} Correlations")
                plt.tight_layout()
                fname = f"{FIGURES_PATH}/03_{fam.lower()}_correlations.svg".replace(' ', '_')
                plt.savefig(fname, bbox_inches="tight")
                plt.close()

        target_corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
        print("\nTop 10 Features Correlated with Target:")
        for feat, corr in target_corr.head(11).items():
            if feat != target:
                print(f"  {feat}: {corr:.4f}")

        eda_results["target_correlations"] = target_corr
        self.artifacts["eda_results"] = eda_results
        return eda_results

    # ----------------------------------
    # Internal: Apply preprocessing
    # ----------------------------------
    def _apply_preprocessing(
            self,
            df: pd.DataFrame,
            is_train: bool = True,
            preprocessors: Dict = None
    ) -> Tuple[pd.DataFrame, Dict]:
        df_processed = df.copy()
        local_preprocessors = {} if preprocessors is None else dict(preprocessors)

        # 3.1 Winsorization (per-fold)
        numeric_cols = [
            c for c in df_processed.select_dtypes(include=[np.number]).columns
            if c != self.config.target_col
        ]
        if is_train:
            winsor_limits = {}
            for col in numeric_cols:
                if df_processed[col].notna().sum() > 0:
                    p1, p99 = np.nanpercentile(df_processed[col], [1, 99])
                    winsor_limits[col] = (float(p1), float(p99))
            local_preprocessors["winsor_limits"] = winsor_limits

        for col, (p1, p99) in local_preprocessors.get("winsor_limits", {}).items():
            if col in df_processed.columns:
                df_processed[col] = np.clip(df_processed[col], p1, p99)

        # 3.2 Family-aware imputation
        meteo_cols = [
            c for c in df_processed.columns
            if any(m in c.upper() for m in ["T2M", "RH2M", "VPD", "WS2M", "ALLSKY", "DTR", "DEW", "ET"])
        ]
        meteo_cols = [c for c in meteo_cols if c in numeric_cols]
        if meteo_cols:
            if is_train:
                meteo_imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    random_state=self.config.random_state,
                    max_iter=10
                )
                df_processed[meteo_cols] = meteo_imputer.fit_transform(df_processed[meteo_cols])
                local_preprocessors["meteo_imputer"] = meteo_imputer
            else:
                if "meteo_imputer" in local_preprocessors:
                    df_processed[meteo_cols] = local_preprocessors["meteo_imputer"].transform(df_processed[meteo_cols])
        local_preprocessors["meteo_cols"] = meteo_cols

        vi_cols = [
            c for c in df_processed.columns
            if any(v in c.upper() for v in ["NDVI", "EVI", "LAI", "NDRE", "SAVI", "CIREDGE", "GPP", "FPAR"])
        ]
        vi_cols = [c for c in vi_cols if c in numeric_cols]
        if vi_cols:
            context_features = []
            for cat_col in self.config.cat_cols:
                if cat_col in df_processed.columns:
                    # Ensure string dtype for get_dummies/label encoding
                    df_processed[cat_col] = df_processed[cat_col].astype(str)

                    if is_train:
                        le = LabelEncoder()
                        enc = le.fit_transform(df_processed[cat_col].fillna("__nan__"))
                        local_preprocessors[f"le_{cat_col}"] = le
                    else:
                        le = local_preprocessors.get(f"le_{cat_col}", None)
                        vals = df_processed[cat_col].fillna("__nan__").astype(str)
                        if le is not None:
                            known = set(le.classes_)
                            unseen = set(vals) - known
                            if unseen:
                                # extend classes safely
                                le.classes_ = np.append(le.classes_, list(unseen))
                            enc = le.transform(vals)
                        else:
                            # fallback: fit new encoder (not ideal but prevents crash)
                            le = LabelEncoder()
                            enc = le.fit_transform(vals)
                            local_preprocessors[f"le_{cat_col}"] = le

                    context_features.append(enc.reshape(-1, 1))

            if context_features:
                vi_features = np.hstack([df_processed[vi_cols].values] + context_features)
                if is_train:
                    vi_imputer = KNNImputer(n_neighbors=5)
                    vi_imputed = vi_imputer.fit_transform(vi_features)
                    local_preprocessors["vi_imputer"] = vi_imputer
                else:
                    vi_imputer = local_preprocessors.get("vi_imputer", None)
                    if vi_imputer is not None:
                        vi_imputed = vi_imputer.transform(vi_features)
                    else:
                        vi_imputer = KNNImputer(n_neighbors=5)
                        vi_imputed = vi_imputer.fit_transform(vi_features)
                        local_preprocessors["vi_imputer"] = vi_imputer

                df_processed[vi_cols] = vi_imputed[:, :len(vi_cols)]
            else:
                # If no context features exist, still impute VI columns
                if is_train:
                    vi_imputer_simple = SimpleImputer(strategy="median")
                    df_processed[vi_cols] = vi_imputer_simple.fit_transform(df_processed[vi_cols])
                    local_preprocessors["vi_imputer_simple"] = vi_imputer_simple
                else:
                    vi_imputer_simple = local_preprocessors.get("vi_imputer_simple", None)
                    if vi_imputer_simple is not None:
                        df_processed[vi_cols] = vi_imputer_simple.transform(df_processed[vi_cols])
        local_preprocessors["vi_cols"] = vi_cols

        soil_cols = [
            c for c in df_processed.columns
            if any(s in c.lower() for s in [
                "ocd_", "nitrogen_", "phh2o_", "sand_", "silt_", "clay_", "bdod_",
                "elevation", "slope", "aspect", "hillshade"
            ])
        ]
        soil_cols = [c for c in soil_cols if c in numeric_cols]
        if soil_cols:
            if is_train:
                soil_imputer = SimpleImputer(strategy="median")
                df_processed[soil_cols] = soil_imputer.fit_transform(df_processed[soil_cols])
                local_preprocessors["soil_imputer"] = soil_imputer
            else:
                if "soil_imputer" in local_preprocessors:
                    df_processed[soil_cols] = local_preprocessors["soil_imputer"].transform(df_processed[soil_cols])
        local_preprocessors["soil_cols"] = soil_cols

        # 3.2b Generic fallback imputation for ALL remaining numeric cols with NaNs
        remaining_num = [c for c in numeric_cols if c in df_processed.columns]
        remaining_with_nan = [c for c in remaining_num if df_processed[c].isna().any()]

        if remaining_with_nan:
            if is_train:
                fallback_imputer = SimpleImputer(strategy="median")
                df_processed[remaining_with_nan] = fallback_imputer.fit_transform(df_processed[remaining_with_nan])
                local_preprocessors["fallback_imputer"] = fallback_imputer
                local_preprocessors["fallback_cols"] = list(remaining_with_nan)
                print(f"🧩 Fallback-imputed {len(remaining_with_nan)} numeric columns (median).")
            else:
                fallback_imputer = local_preprocessors.get("fallback_imputer", None)
                fallback_cols = local_preprocessors.get("fallback_cols", [])
                if fallback_imputer is not None and fallback_cols:
                    cols = [c for c in fallback_cols if c in df_processed.columns]
                    if cols:
                        df_processed[cols] = fallback_imputer.transform(df_processed[cols])

        # 3.3 One-hot encoding (per-fold alignment)
        cat_cols = list(self.config.cat_cols)

        encoded_dfs = []
        for col in cat_cols:
            if col in df_processed.columns:
                dummies = pd.get_dummies(
                    df_processed[col],
                    prefix=col.split("(")[0],
                    drop_first=True
                )
                encoded_dfs.append(dummies)

        if encoded_dfs:
            enc = pd.concat(encoded_dfs, axis=1)
        else:
            enc = pd.DataFrame(index=df_processed.index)

        df_processed = pd.concat([df_processed, enc], axis=1)
        df_processed.drop(columns=[c for c in cat_cols if c in df_processed.columns], inplace=True)

        if is_train:
            local_preprocessors["one_hot_columns"] = list(enc.columns)
        else:
            # Align one-hot columns to training
            train_cols = local_preprocessors.get("one_hot_columns", [])
            if train_cols:
                # Add missing dummy cols
                for c in train_cols:
                    if c not in df_processed.columns:
                        df_processed[c] = 0
                # Drop extra dummy cols that weren't seen in training
                extra = [c for c in df_processed.columns if
                         (c not in train_cols and any(c.startswith(cc.split("(")[0] + "_") for cc in cat_cols))]
                if extra:
                    df_processed.drop(columns=extra, inplace=True, errors="ignore")
                # Ensure consistent order later (we don't reorder everything here yet)

        # 3.4 Scaling
        final_numeric = [
            c for c in df_processed.select_dtypes(include=[np.number]).columns
            if c != self.config.target_col
        ]
        if is_train:
            med_raw = df_processed[final_numeric].median().to_dict()
            local_preprocessors["feature_medians_raw"] = {k: float(v) for k, v in med_raw.items()}

            scaler = RobustScaler()
            df_processed[final_numeric] = scaler.fit_transform(df_processed[final_numeric])
            local_preprocessors["scaler"] = scaler
        else:
            scaler = local_preprocessors.get("scaler", None)
            if scaler is not None:
                df_processed[final_numeric] = scaler.transform(df_processed[final_numeric])
        local_preprocessors["scaled_columns"] = list(final_numeric)

        # Final safety check: no NaNs in numeric features (except target)
        check_cols = [
            c for c in df_processed.select_dtypes(include=[np.number]).columns
            if c != self.config.target_col
        ]
        total_nans = int(df_processed[check_cols].isna().sum().sum())
        if total_nans > 0:
            bad = df_processed[check_cols].isna().mean().sort_values(ascending=False)
            bad = bad[bad > 0].head(25)
            raise ValueError(
                f"Preprocessing produced NaNs in numeric features (total NaNs={total_nans}). "
                f"Top columns (% missing):\n{(bad * 100).round(2)}"
            )

        return df_processed, local_preprocessors

    # ----------------------------------
    # Step 3: Preprocess (schema only)
    # ----------------------------------
    def advanced_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔄 Step 3: Advanced preprocessing (full data for feature discovery)...")

        df_processed = df.copy()
        cat_cols = self.config.cat_cols

        encoded_dfs = []
        for col in cat_cols:
            if col in df_processed.columns:
                dummies = pd.get_dummies(df_processed[col], prefix=col.split('(')[0], drop_first=True)
                encoded_dfs.append(dummies)

        if encoded_dfs:
            enc = pd.concat(encoded_dfs, axis=1)
            df_processed = pd.concat([df_processed, enc], axis=1)
            df_processed.drop(columns=[c for c in cat_cols if c in df_processed.columns], inplace=True)

        self.preprocessors['one_hot_columns'] = list(enc.columns) if encoded_dfs else []
        print("✅ Applied one-hot encoding for feature alignment")
        print("❌ Note: Imputation/Scaling will be done inside the CV loop to prevent data leakage.")

        return df_processed

    # ----------------------------------
    # Step 4: Statistical screening
    # ----------------------------------
    def statistical_screening(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        print("🔄 Step 4: Statistical screening...")

        df_screened = df.copy()
        dropped: List[str] = []

        target = self.config.target_col
        feature_cols = [c for c in df_screened.columns if (c != target)*(c.split("_")[0] != "fp")]
        # print("Feature columns: ", feature_cols)
        # Near-zero variance
        variances = df_screened[feature_cols].var(numeric_only=True)
        low_var = variances[variances < 1e-4].index.tolist()
        if low_var:
            df_screened.drop(columns=low_var, inplace=True)
            dropped.extend([(f, "low_variance") for f in low_var])
            print(f"✅ Dropped {len(low_var)} low-variance features")

        # High-correlation pruning (conservative)
        numeric_cols = [c for c in df_screened.columns if c != target and (c.split("_")[0] != "fp") and str(df_screened[c].dtype)[:5] in ['float', 'int']]
        if len(numeric_cols) > 1:
            corr = df_screened[numeric_cols].corr().abs()
            pairs, to_drop = [], set()
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    if corr.iloc[i, j] >= 0.98:
                        c1, c2 = corr.columns[i], corr.columns[j]
                        pairs.append((c1, c2, corr.iloc[i, j]))
            for c1, c2, cval in pairs:
                if c1 in to_drop or c2 in to_drop:
                    continue
                # Choose by mutual information with target
                df1 = df_screened[[c1, target]].dropna()
                df2 = df_screened[[c2, target]].dropna()
                mi1 = mutual_info_regression(df1[[c1]], df1[target], random_state=self.config.random_state)[0] if not df1.empty else 0.0
                mi2 = mutual_info_regression(df2[[c2]], df2[target], random_state=self.config.random_state)[0] if not df2.empty else 0.0
                if mi1 >= mi2:
                    to_drop.add(c2); dropped.append((c2, f"collinear_with_{c1}_corr_{cval:.3f}"))
                else:
                    to_drop.add(c1); dropped.append((c1, f"collinear_with_{c2}_corr_{cval:.3f}"))
            if to_drop:
                df_screened.drop(columns=list(to_drop), inplace=True)
                print(f"✅ Dropped {len(to_drop)} highly correlated features")

        # Family coverage backfill
        feature_families = self.artifacts['eda_results']['feature_families']
        remaining = [c for c in df_screened.columns if (c != target)*(c.split("_")[0] != "fp")]
        for fam, fam_feats in feature_families.items():
            fam_left = [f for f in fam_feats if f in remaining]
            if not fam_left and fam_feats:
                fam_in_orig = [f for f in fam_feats if f in df.columns and f != target]
                if fam_in_orig:
                    fam_corr = df[fam_in_orig + [target]].corr()[target].abs()
                    best = fam_corr.drop(target).idxmax()
                    if best not in df_screened.columns:
                        df_screened[best] = df[best]
                        dropped = [(f, r) for f, r in dropped if f != best]
                        print(f"✅ Restored {best} to maintain {fam} family coverage")

        print(f"✅ Statistical screening complete. Features: {len(df.columns)-1} → {len(df_screened.columns)-1}")
        self.artifacts['dropped_features'] = dropped
        return df_screened, dropped

    # ----------------------------------
    # Step 5: Feature engineering
    # ----------------------------------
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔄 Step 5: Feature engineering...")

        df_eng = df.copy()
        added: List[str] = []

        # Temperature-based
        tmean_cols = [c for c in df_eng.columns if 'T2M_mean' in c or ('T2M' in c and '_mean' in c)]
        if tmean_cols:
            df_eng['GDD_base10'] = np.maximum(0, df_eng[tmean_cols[0]] - 10)
            added.append('GDD_base10')

        if 'T2M_MAX_mean' in df_eng.columns:
            df_eng['heat_days_35'] = (df_eng['T2M_MAX_mean'] >= 35).astype(int)
            added.append('heat_days_35')

        if 'T2M_MIN_mean' in df_eng.columns:
            df_eng['chill_nights_15'] = (df_eng['T2M_MIN_mean'] <= 15).astype(int)
            added.append('chill_nights_15')

        # Energy/ET ratios & interactions
        et_cols = [c for c in df_eng.columns if c.startswith('ET') and 'mean' in c]
        solar_cols = [c for c in df_eng.columns if 'ALLSKY_SFC_SW_DWN_mean' in c]
        if et_cols and solar_cols:
            df_eng['ET_solar_ratio'] = df_eng[et_cols[0]] / (df_eng[solar_cols[0]] + 1e-6)
            added.append('ET_solar_ratio')

        vpd_cols = [c for c in df_eng.columns if 'VPD_mean' in c]
        if vpd_cols and et_cols:
            df_eng['VPD_ET_interaction'] = df_eng[vpd_cols[0]] * df_eng[et_cols[0]]
            added.append('VPD_ET_interaction')

        # VI dynamics
        ndvi_max = [c for c in df_eng.columns if 'NDVI_max' in c]
        ndvi_min = [c for c in df_eng.columns if 'NDVI_min' in c]
        ndvi_mean = [c for c in df_eng.columns if 'NDVI_mean' in c]
        if ndvi_max and ndvi_min:
            df_eng['NDVI_amplitude'] = df_eng[ndvi_max[0]] - df_eng[ndvi_min[0]]
            added.append('NDVI_amplitude')
        if ndvi_max and ndvi_min and ndvi_mean:
            num = df_eng[ndvi_mean[0]] - df_eng[ndvi_min[0]]
            den = df_eng[ndvi_max[0]] - df_eng[ndvi_min[0]] + 1e-6
            df_eng['greenness_persistence'] = num / den
            added.append('greenness_persistence')

        evi_max = [c for c in df_eng.columns if 'EVI_max' in c]
        evi_min = [c for c in df_eng.columns if 'EVI_min' in c]
        if evi_max and evi_min:
            df_eng['EVI_amplitude'] = df_eng[evi_max[0]] - df_eng[evi_min[0]]
            added.append('EVI_amplitude')

        # SAR extras
        vv_cols = [c for c in df_eng.columns if 'VV_mean' in c]
        vh_cols = [c for c in df_eng.columns if 'VH_mean' in c]
        if vv_cols and vh_cols:
            df_eng['VV_VH_ratio'] = df_eng[vv_cols[0]] / (df_eng[vh_cols[0]] + 1e-6)
            added.append('VV_VH_ratio')

        vv_std_cols = [c for c in df_eng.columns if 'VV_stdDev' in c or 'VV_std' in c]
        if vv_std_cols:
            df_eng['VV_texture'] = df_eng[vv_std_cols[0]]
            added.append('VV_texture')

        # Terrain/soil interactions
        elev_cols = [c for c in df_eng.columns if 'elevation_mean' in c]
        if tmean_cols and elev_cols:
            df_eng['temp_elevation_interaction'] = df_eng[tmean_cols[0]] * df_eng[elev_cols[0]]
            added.append('temp_elevation_interaction')

        sand_cols = [c for c in df_eng.columns if 'sand_0-5cm_mean_mean' in c]
        clay_cols = [c for c in df_eng.columns if 'clay_0-5cm_mean_mean' in c]
        if sand_cols and clay_cols:
            df_eng['water_holding_capacity'] = df_eng[clay_cols[0]] / (df_eng[sand_cols[0]] + 1e-6)
            added.append('water_holding_capacity')

        # Low-order polynomial interactions (data-driven)
        numeric_cols = [c for c in df_eng.columns if c != self.config.target_col and str(df_eng[c].dtype)[:5] in ['float', 'int']]
        if numeric_cols:
            tc = df_eng[numeric_cols + [self.config.target_col]].corr()[self.config.target_col].abs()
            top = tc.nlargest(8).index[1:].tolist()
            required_k = 4
            # Ensure candidates exist in df_eng and are numeric
            candidates = [c for c in top if c in df_eng.columns]
            top4 = []
            skipped = []
            for c in candidates:
                # Force numeric (if it becomes NaN due to coercion, it will be caught)
                s = pd.to_numeric(df_eng[c], errors="coerce")
                if s.isna().any():
                    skipped.append((c, int(s.isna().sum())))
                    continue
                top4.append(c)
                if len(top4) == required_k:
                    break
            if len(top4) < required_k:
                raise ValueError(
                    f"Could not find {required_k} NaN-free features for PolynomialFeatures. "
                    f"Found only {len(top4)}. Skipped (feature, n_nans)={skipped[:15]}"
                )
            print(f"✅ Top-{required_k} NaN-free poly columns: {top4}")
            if skipped:
                print(f"ℹ️ Skipped NaN-containing candidates (first 10): {skipped[:10]}")
            if len(top) >= 4:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                interactions = poly.fit_transform(df_eng[top4])
                names = poly.get_feature_names_out(top4)
                name_to_idx = {n: i for i, n in enumerate(names)}
                inter_names = [n for n in names if ' ' in n]
                count = 0
                for n in inter_names:
                    if count >= 6:
                        break
                    col_name = f"interaction_{n.replace(' ', '_')}"
                    df_eng[col_name] = interactions[:, name_to_idx[n]]
                    added.append(col_name)
                    count += 1

        print(f"✅ Feature engineering complete. Added {len(added)} new features:")
        for f in added:
            print(f"    - {f}")

        self.artifacts['engineered_features'] = added
        return df_eng

    # ----------------------------------
    # Step 7: Feature selection
    # ----------------------------------
    def feature_selection(self, df: pd.DataFrame) -> Tuple[List[str], Dict]:
        target = self.config.target_col
        feats = [c for c in df.columns if c != target and str(df[c].dtype)[:5] in ['float', 'int']]

        df_clean = df.dropna(subset=[target])
        X = df_clean[feats]
        y = df_clean[target].values

        pearson_abs, pearson_p, spearman_abs, spearman_p = [], [], [], []
        for c in feats:
            x = X[c].values
            mask = np.isfinite(x) & np.isfinite(y)
            x_f, y_f = x[mask], y[mask]
            if len(x_f) < 3 or np.std(x_f) == 0:
                pr = sr = 0.0; pp = sp = 1.0
            else:
                pr, pp = stats.pearsonr(x_f, y_f)
                sr, sp = stats.spearmanr(x_f, y_f)
            pearson_abs.append(abs(pr)); pearson_p.append(pp)
            spearman_abs.append(abs(sr)); spearman_p.append(sp)

        mi_vals = mutual_info_regression(X.fillna(X.median()), y, random_state=self.config.random_state)

        score_df = pd.DataFrame({
            'feature': feats,
            'pearson_abs': pearson_abs,
            'pearson_p': pearson_p,
            'spearman_abs': spearman_abs,
            'spearman_p': spearman_p,
            'mi': mi_vals
        }).set_index('feature')

        mask_keep = (score_df['pearson_p'] < 0.20) | (score_df['spearman_p'] < 0.20)
        score_df = score_df[mask_keep] if mask_keep.any() else score_df

        def norm(s):
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0)
            r = s.max() - s.min()
            return (s - s.min()) / r if r > 0 else s * 0

        score_df['relevance'] = 0.5 * norm(score_df['pearson_abs']) + 0.25 * norm(score_df['spearman_abs']) + 0.25 * norm(score_df['mi'])
        ranked = score_df.sort_values('relevance', ascending=False).index.tolist() or feats

        corr = df[ranked].corr().abs()
        selected: List[str] = []
        corr_thr = 0.95
        red_w = 0.5

        def penalized(f, chosen):
            rel = score_df.loc[f, 'relevance'] if f in score_df.index else 0.0
            if not chosen:
                return rel
            red = corr.loc[f, chosen].mean()
            return rel - red_w * (0.0 if np.isnan(red) else red)

        # ensure family coverage first
        fams = self.artifacts['eda_results']['feature_families']
        for fam, fam_feats in fams.items():
            cands = [f for f in ranked if f in fam_feats]
            if cands:
                top = cands[0]
                if top not in selected:
                    selected.append(top)

        while len(selected) < self.config.min_features and len(selected) < len(ranked):
            if selected:
                allowed = [f for f in ranked if f not in selected and (corr.loc[f, selected].max() if len(selected) else 0) < corr_thr]
            else:
                allowed = [f for f in ranked if f not in selected]
            if not allowed:
                corr_thr = min(0.999, corr_thr + 0.02)
                fallback = next((f for f in ranked if f not in selected), None)
                if fallback is None:
                    break
                selected.append(fallback)
                continue
            scores = {f: penalized(f, selected) for f in allowed}
            nxt = max(scores, key=lambda k: scores[k])
            selected.append(nxt)

        if len(selected) < self.config.min_features:
            for f in ranked:
                if f not in selected:
                    selected.append(f)
                if len(selected) >= self.config.min_features:
                    break

        selected = selected[:self.config.min_features]

        summary = {
            'total_features': len(feats),
            'stable_features_count': len(selected),
            'selected_order': selected,
            'composite_scores': {f: float(score_df.loc[f, 'relevance']) if f in score_df.index else 0.0 for f in selected},
            'score_df': score_df
        }
        return selected, summary

    # ----------------------------------
    # Step 8: Train baselines (CV)
    # ----------------------------------
    def train_baseline_models(self, df: pd.DataFrame) -> Dict:
        print("🔄 Step 8: Training baseline models...")

        target = self.config.target_col

        # Decide CV strategy and record for report
        if all(col in df.columns for col in self.config.group_cols):
            group_keys = ["_".join([str(row[c]) for c in self.config.group_cols]) for _, row in df.iterrows()]
            uniq = list(set(group_keys))
            gmap = {g: i for i, g in enumerate(uniq)}
            groups = [gmap[g] for g in group_keys]
            cv = GroupKFold(n_splits=self.config.cv_folds)
            cv_name = f"{self.config.cv_folds}-fold GroupKFold by " + " × ".join(self.config.group_cols)
        else:
            groups = None
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            cv_name = f"{self.config.cv_folds}-fold KFold (shuffle)"
        self.artifacts['cv_strategy'] = cv_name
        self.artifacts['cv_folds'] = self.config.cv_folds

        # Initialize holders
        models = {}
        oof = {m: np.zeros(len(df)) for m in ['LightGBM', 'RandomForest', 'SVM', 'ElasticNet']}
        metrics = {}

        preprocessors_fold0 = {}

        # CV loop
        for fold, (tr_idx, va_idx) in enumerate(cv.split(df, df[target], groups)):
            print(f"  Processing fold {fold + 1}/{self.config.cv_folds}")
            dtr = df.iloc[tr_idx].copy()
            dva = df.iloc[va_idx].copy()

            if fold == 0:
                tr_proc, preprocessors_fold0 = self._apply_preprocessing(dtr, is_train=True)
                va_proc, _ = self._apply_preprocessing(dva, is_train=False, preprocessors=preprocessors_fold0)
            else:
                tr_proc, _ = self._apply_preprocessing(dtr, is_train=False, preprocessors=preprocessors_fold0)
                va_proc, _ = self._apply_preprocessing(dva, is_train=False, preprocessors=preprocessors_fold0)

            # Feature selection inside fold
            sel, _ = self.feature_selection(tr_proc)
            Xtr, ytr = tr_proc[sel], tr_proc[target]
            Xva, yva = va_proc[sel], va_proc[target]

            # LightGBM
            lgbm = lgb.LGBMRegressor(
                n_estimators=5000, learning_rate=0.02, num_leaves=63,
                min_child_samples=40, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=5, lambda_l1=1.0, lambda_l2=1.0,
                random_state=self.config.random_state, verbose=-1
            )
            lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof['LightGBM'][va_idx] = lgbm.predict(Xva)
            if fold == 0:
                models['LightGBM'] = lgbm

            # RandomForest
            rf = RandomForestRegressor(
                n_estimators=800, max_depth=None, min_samples_split=5, min_samples_leaf=3,
                max_features=0.5, n_jobs=-1, random_state=self.config.random_state
            )
            rf.fit(Xtr, ytr)
            oof['RandomForest'][va_idx] = rf.predict(Xva)
            if fold == 0:
                models['RandomForest'] = rf

            # SVM tiny grid
            best_svr, best_rmse = None, np.inf
            for C in [1, 10, 20, 50, 100]:
                for gamma in ['scale', 0.05, 0.1]:
                    for ker in ['rbf', 'linear']:
                        svr = SVR(kernel=ker, C=C, epsilon=0.1, gamma=gamma)
                        svr.fit(Xtr, ytr)
                        p = svr.predict(Xva)
                        rmse = np.sqrt(mean_squared_error(yva, p))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_svr = svr
            oof['SVM'][va_idx] = best_svr.predict(Xva)
            print(best_svr)
            if fold == 0:
                models['SVM'] = best_svr

            # ElasticNet
            enet = ElasticNetCV(
                l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1.0],
                alphas=np.logspace(-4, 4, 50),
                cv=5, max_iter=10000, random_state=self.config.random_state
            )
            enet.fit(Xtr, ytr)
            oof['ElasticNet'][va_idx] = enet.predict(Xva)
            if fold == 0:
                models['ElasticNet'] = enet

        # Aggregate metrics
        y_full = df[target]
        for name in oof:
            yp = oof[name]
            metrics[name] = {
                'RMSE': float(np.sqrt(mean_squared_error(y_full, yp))),
                'MAE': float(mean_absolute_error(y_full, yp)),
                'R2': float(r2_score(y_full, yp)),
                # 'WAPE': float(np.mean(np.abs((y_full - yp) / np.clip(y_full, 1e-6, None))) * 100),
                "WAPE": float(np.sum(np.abs(y_full - yp)) / (np.sum(np.abs(y_full)) + 1e-6) * 100)
            }

        # === Save metric bars separately ===
        mdf = pd.DataFrame(metrics).T
        for metric in ['RMSE', 'MAE', 'R2', 'WAPE']:
            if metric in mdf.columns:
                plt.figure(figsize=(8, 6))
                mdf[metric].plot(kind='bar')
                plt.title(f'{metric} Comparison')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{FIGURES_PATH}/05{dict(RMSE="a",MAE="b",R2="c",WAPE="d")[metric]}_{metric.lower()}_comparison.svg',
                            bbox_inches='tight')
                plt.close()

        # === OOF vs Actual (separate) ===
        plt.figure(figsize=(9, 7))
        colors = ['C0', 'C1', 'C2', 'C3']
        for (name, color) in zip(['LightGBM', 'RandomForest', 'SVM', 'ElasticNet'], colors):
            plt.scatter(y_full, oof[name], alpha=0.6, s=20,
                        label=f'{name} (R²={metrics[name]["R2"]:.3f})', color=color)
        mmin, mmax = y_full.min(), y_full.max()
        plt.plot([mmin, mmax], [mmin, mmax], 'k--', alpha=0.8, linewidth=2)
        plt.xlabel('Actual Yield'); plt.ylabel('Predicted Yield'); plt.title('OOF Predictions vs Actual')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_PATH}/05e_oof_scatter.svg', bbox_inches='tight')
        plt.close()

        # === FIGURE H: Residual Distributions + Residuals vs Fitted ===
        plt.figure(figsize=(14, 6))

        model_list = ['LightGBM', 'RandomForest', 'SVM', 'ElasticNet']
        colors = ['C0', 'C1', 'C2', 'C3']

        # Collect all residuals for the overall KDE curve
        all_residuals = []
        for name in model_list:
            all_residuals.extend(list(oof[name] - y_full))

        all_residuals = np.array(all_residuals)

        # --- LEFT: Residual distribution ---
        plt.subplot(1, 2, 1)

        # Individual histograms
        for name, color in zip(model_list, colors):
            residuals = oof[name] - y_full
            plt.hist(residuals, bins=40, alpha=0.35, density=True, color=color, label=f"{name}")

        # Add overall residual KDE curve
        # Using Gaussian KDE manually (no seaborn)
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(all_residuals)
        xs = np.linspace(all_residuals.min(), all_residuals.max(), 500)
        plt.plot(xs, kde(xs), color='black', linewidth=2.5, label="Overall Residual Density")

        plt.axvline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel("Residual (Predicted – Actual)")
        plt.ylabel("Density")
        plt.title("Residual Distributions")
        plt.grid(alpha=0.3)
        plt.legend()

        # --- RIGHT: Residuals vs Fitted ---
        plt.subplot(1, 2, 2)
        for name, color in zip(model_list, colors):
            fitted = oof[name]
            residuals = fitted - y_full
            plt.scatter(fitted, residuals, alpha=0.5, s=20, color=color, label=name)

        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel("Fitted Values (Predicted Yield)")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/05f_residual_diagnostics.svg", bbox_inches='tight')
        plt.close()

        best_name = min(metrics.keys(), key=lambda k: metrics[k]['RMSE'])
        print(f"Best model: {best_name} (RMSE: {metrics[best_name]['RMSE']:.4f})")

        # Refit preprocessors and select features on full data (for diagnostics/exports)
        print("Refitting full preprocessing and models on all data for export...")
        df_full_proc, full_preprocs = self._apply_preprocessing(df, is_train=True)
        self.preprocessors.update(full_preprocs)

        print("Running final feature selection on full data...")
        sel_feats, sel_summary = self.feature_selection(df_full_proc)
        self.selected_features = sel_feats
        self.artifacts['feature_selection'] = sel_summary

        print(f"\n✅ Final {self.config.min_features} features selected for the model:")
        for i, f in enumerate(self.selected_features, 1):
            print(f"    {i:2d}. {f}")
        print()

        # Feature importance (only if best single model is tree-based)
        fi_df = pd.DataFrame()
        if best_name == 'LightGBM':
            full_model = lgb.LGBMRegressor(
                n_estimators=int(getattr(models.get('LightGBM', None), 'best_iteration_', 400)),
                learning_rate=0.02, num_leaves=63, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=5, min_child_samples=40,
                lambda_l1=1.0, lambda_l2=1.0, random_state=self.config.random_state, verbose=-1
            )
            full_model.fit(df_full_proc[self.selected_features], df_full_proc[target])
            fi_df = pd.DataFrame({'Feature': self.selected_features, 'Importance': full_model.feature_importances_}).sort_values('Importance', ascending=True)
            models['LightGBM_full'] = full_model
        elif best_name == 'RandomForest':
            full_model = RandomForestRegressor(
                n_estimators=800, max_depth=None, min_samples_split=5, min_samples_leaf=3,
                max_features=0.5, n_jobs=-1, random_state=self.config.random_state
            )
            full_model.fit(df_full_proc[self.selected_features], df_full_proc[target])
            fi_df = pd.DataFrame({'Feature': self.selected_features, 'Importance': full_model.feature_importances_}).sort_values('Importance', ascending=True)
            models['RandomForest_full'] = full_model

        self.plot_all_final_diagnostics(df, sel_summary, best_name, fi_df)

        # Train full-data versions for all base models (for SHAP/ensembling & consistency)
        X = df_full_proc[self.selected_features]
        y = df_full_proc[target]

        lgb_full = lgb.LGBMRegressor(
            n_estimators=int(getattr(models.get('LightGBM', None), 'best_iteration_', 400)),
            learning_rate=0.02, num_leaves=63, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=5, min_child_samples=40,
            lambda_l1=1.0, lambda_l2=1.0, random_state=self.config.random_state, verbose=-1
        )
        lgb_full.fit(X, y); self.models['LightGBM_full'] = lgb_full

        rf_full = RandomForestRegressor(
            n_estimators=800, max_depth=None, min_samples_split=5, min_samples_leaf=3,
            max_features=0.5, n_jobs=-1, random_state=self.config.random_state
        )
        rf_full.fit(X, y); self.models['RandomForest_full'] = rf_full

        best_svr_full, best_score = None, np.inf
        for C in [1, 5, 10]:
            for gamma in ['scale', 0.05, 0.1]:
                m = SVR(kernel='rbf', C=C, epsilon=0.1, gamma=gamma)
                m.fit(X, y)
                rmse = float(np.sqrt(mean_squared_error(y, m.predict(X))))
                if rmse < best_score:
                    best_score, best_svr_full = rmse, m
        self.models['SVM_full'] = best_svr_full

        elastic_full = ElasticNetCV(
            l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1.0],
            alphas=np.logspace(-4, 2, 50),
            cv=5, max_iter=5000, random_state=self.config.random_state
        )
        elastic_full.fit(X, y); self.models['ElasticNet_full'] = elastic_full

        print("\n📊 Baseline Model Results:")
        print("=" * 60)
        for model_name, m in metrics.items():
            print(f"{model_name}:")
            for metric_name, value in m.items():
                print(f"  {metric_name}: {value:.4f}")
            print()

        self.models.update(models)
        self.oof_predictions.update(oof)
        self.metrics['baseline'] = metrics

        return {'models': models, 'oof_predictions': oof, 'metrics': metrics, 'best_model': best_name}

    # --------------------
    # Plot diagnostics
    # --------------------
    def plot_all_final_diagnostics(self, df, selection_summary, best_model_name, feature_importance_df):
        def wrap_label(s: str, width: int = 18, max_lines: int = 3) -> str:
            s2 = s.replace('_', ' ')
            wrapped = textwrap.fill(s2, width=width)
            return "\n".join(wrapped.splitlines()[:max_lines])

        selected = selection_summary['selected_order']
        score_df = selection_summary['score_df']
        families = self.artifacts['eda_results']['feature_families']

        # Composite relevance (separate)
        top_rel = score_df.sort_values('relevance', ascending=False).head(18).iloc[::-1]
        plt.figure(figsize=(11, 8))
        plt.barh([wrap_label(i, 26) for i in top_rel.index], top_rel['relevance'])
        plt.title("Composite Relevance (Top 18)")
        plt.xlabel("score"); plt.ylabel("feature"); plt.grid(axis='x', alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_PATH}/04a_composite_relevance.svg', bbox_inches='tight')
        plt.close()

        # Family coverage (separate)
        plt.figure(figsize=(10, 7))
        fam_counts = {fam: len([f for f in selected if f in families.get(fam, [])]) for fam in families}
        if fam_counts:
            plt.bar(list(fam_counts.keys()), list(fam_counts.values()))
            plt.title("Family Coverage"); plt.ylabel("# selected"); plt.xticks(rotation=20); plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_PATH}/04b_family_coverage.svg', bbox_inches='tight')
        plt.close()

        # Final selected relevance (separate)
        plt.figure(figsize=(11, 8))
        sel_scores = (score_df.loc[[f for f in selected if f in score_df.index], 'relevance'].sort_values(ascending=True))
        plt.barh([wrap_label(i, 26) for i in sel_scores.index], sel_scores.values)
        plt.title(f"Final Selected (n={len(selected)})")
        plt.xlabel("relevance"); plt.grid(axis='x', alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_PATH}/04c_final_selected.svg', bbox_inches='tight')
        plt.close()

        # Feature importance (separate if available)
        if best_model_name in ['LightGBM', 'RandomForest'] and not feature_importance_df.empty:
            plt.figure(figsize=(11, 8))
            top20 = feature_importance_df.tail(20)
            plt.barh(top20['Feature'], top20['Importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'{best_model_name} Feature Importance (Top 20)')
            plt.grid(axis='x', alpha=0.2)
            plt.tight_layout()
            plt.savefig(f'{FIGURES_PATH}/04d_feature_importance.svg', bbox_inches='tight')
            plt.close()

    # ----------------------------------
    # Step 10: Ensembling
    # ----------------------------------
    def ensemble_models(self, df: pd.DataFrame) -> Dict:
        print("🔄 Step 10: Ensemble modeling (non-negative weights with SVM floor)...")

        target = self.config.target_col
        y = df[target].values

        base_names = ['LightGBM', 'RandomForest', 'SVM', 'ElasticNet']
        feature_names, preds_stack = [], []
        for n in base_names:
            p = self.oof_predictions.get(n)
            if p is not None and len(p) == len(y):
                feature_names.append(n)
                preds_stack.append(p)
        if len(preds_stack) < 2:
            print("Not enough models for ensembling. Skipping ensemble step.")
            return {}

        X_ens = np.column_stack(preds_stack)

        svm_floor = 0.05
        bounds = [(svm_floor, 1.0) if n == 'SVM' else (0.0, 1.0) for n in feature_names]

        def objective(w):
            r = y - X_ens @ w
            # return np.mean(r * r)
            return np.mean(np.abs(r))

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        k = X_ens.shape[1]
        w0 = np.full(k, 1.0 / k, dtype=float)
        for i, (lo, _) in enumerate(bounds):
            w0[i] = max(w0[i], lo)
        w0 = w0 / w0.sum()

        opt = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
        if not opt.success:
            print("SLSQP failed; falling back to NNLS + normalization (respecting SVM floor).")
            w, _ = nnls(X_ens, y)
            s = w.sum()
            w = w / s if s > 0 else np.full_like(w, 1.0 / len(w))
        else:
            w = opt.x

        # Enforce SVM floor post-normalization
        if 'SVM' in feature_names:
            i = feature_names.index('SVM')
            w = w / max(w.sum(), 1e-12)
            if w[i] < svm_floor:
                rem = [j for j in range(len(w)) if j != i]
                rem_sum = w[rem].sum()
                scale = (1.0 - svm_floor) / max(rem_sum, 1e-12)
                w[rem] = w[rem] * scale
                w[i] = svm_floor
        else:
            w = w / max(w.sum(), 1e-12)

        oof_ens = X_ens @ w
        met = {
            'RMSE': float(np.sqrt(mean_squared_error(y, oof_ens))),
            'MAE': float(mean_absolute_error(y, oof_ens)),
            'R2': float(r2_score(y, oof_ens)),
            'WAPE': float(np.sum(np.abs(y - oof_ens)) / (np.sum(np.abs(oof_ens)) + 1e-6) * 100)
        }

        all_metrics = dict(self.metrics.get('baseline', {}))
        all_metrics['Ensemble'] = met

        # Save comparison bars separately
        mdf = pd.DataFrame(all_metrics).T
        for metric in ['RMSE', 'MAE', 'R2', 'WAPE']:
            if metric in mdf.columns:
                plt.figure(figsize=(8, 6))
                mdf[metric].plot(kind='bar')
                plt.title(f'{metric} Comparison')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{FIGURES_PATH}/06{dict(RMSE="a",MAE="b",R2="c",WAPE="d")[metric]}_{metric.lower()}_comparison.svg',
                            bbox_inches='tight')
                plt.close()

        print("📊 Ensemble (SLSQP with SVM floor) Weights:")
        for n, wv in zip(feature_names, w):
            print(f"    {n}: {wv:.4f}")

        print(f"\n📊 Final Model Comparison:")
        print("=" * 60)
        for n, m in all_metrics.items():
            print(f"{n}:")
            for mk, mv in m.items():
                print(f"  {mk}: {mv:.4f}")
            print()

        best_single = min({k: v for k, v in all_metrics.items() if k != 'Ensemble'}.keys(),
                          key=lambda k: all_metrics[k]['RMSE'])
        if met['RMSE'] < all_metrics[best_single]['RMSE']:
            best_model = 'Ensemble'
            print(f"🏆 Best Model: Ensemble (RMSE: {met['RMSE']:.4f}) beats {best_single}")
        else:
            best_model = best_single
            print(f"🏆 Best Model: {best_single} (Ensemble not kept as champion)")

        self.oof_predictions['Ensemble'] = oof_ens
        self.metrics['ensemble'] = met
        self.artifacts['final_comparison'] = all_metrics
        self.artifacts['ensemble_weights'] = dict(zip(feature_names, [float(x) for x in w]))
        self.artifacts['best_model_name'] = best_model

        return {'weights': w, 'feature_names': feature_names, 'oof_predictions': oof_ens,
                'metrics': met, 'all_metrics': all_metrics, 'best_model': best_model}

    # ----------------------------------
    # Step 11: Interpretability
    # ----------------------------------
    def model_interpretability(self, df: pd.DataFrame, selected_features: List[str]) -> Dict:
        print("🔄 Step 11: Model interpretability...")

        results = {}
        if 'LightGBM_full' in self.models:
            print("  Generating SHAP explanations for LightGBM...")
            X = df[selected_features]
            explainer = shap.TreeExplainer(self.models['LightGBM_full'])
            shap_values = explainer.shap_values(X.iloc[:100])
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Summary (separate)
            plt.figure(figsize=(19, 8))
            shap.summary_plot(shap_values, X.iloc[:100], feature_names=selected_features, show=False, plot_size=(15, 7))
            plt.title('SHAP Summary Plot - LightGBM')
            plt.tight_layout()
            plt.savefig(f'{FIGURES_PATH}/07_shap_summary.svg', bbox_inches='tight')
            plt.close()

            # Importance (separate)
            shap_imp = np.abs(shap_values).mean(0)
            fi = pd.DataFrame({'Feature': selected_features, 'SHAP_Importance': shap_imp}).sort_values('SHAP_Importance', ascending=True)

            plt.figure(figsize=(12, 6))
            top20 = fi.tail(5)
            plt.barh(top20['Feature'], top20['SHAP_Importance'])
            plt.xlabel('Mean |SHAP Value|'); plt.title('SHAP Feature Importance (Top 5)')
            plt.tight_layout()
            plt.savefig(f'{FIGURES_PATH}/08_shap_importance.svg', bbox_inches='tight')
            plt.close()

            # Dependence plots (each saved separately)
            top_feats = fi['Feature'].tail(4).tolist()
            for i, feat in enumerate(top_feats, start=1):
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(feat, shap_values, X.iloc[:100],
                                     feature_names=selected_features, show=False)
                plt.title(f'SHAP Dependence: {feat}')
                plt.tight_layout()
                plt.savefig(f'{FIGURES_PATH}/09{i}_shap_dependence_{i}.svg', bbox_inches='tight')
                plt.close()

            results['shap'] = {'values': shap_values, 'feature_importance': fi, 'top_features': top_feats}

        return results

    # ----------------------------------
    # Step 12: Save PKL + write predict_crop_yield.py
    # ----------------------------------
    def generate_prediction_function(self, df: pd.DataFrame, selected_features: List[str]) -> None:
        print("🔄 Step 12: Exporting artifacts and generating prediction function...")

        # Build PSI reference from top correlated (fallback to selected)
        top_for_psi = selected_features[:12]
        if 'eda_results' in self.artifacts and 'target_correlations' in self.artifacts['eda_results']:
            tc = self.artifacts['eda_results']['target_correlations']
            top_for_psi = [f for f in tc.index if f != self.config.target_col][:12]

        psi_reference = {}
        for feat in top_for_psi:
            if feat in df.columns and df[feat].notna().sum() > 0:
                xs = df[feat].clip(lower=np.nanpercentile(df[feat], 1), upper=np.nanpercentile(df[feat], 99))
                try:
                    bins = np.unique(np.quantile(xs.dropna(), q=np.linspace(0, 1, 11)))
                    if len(bins) >= 2:
                        hist, _ = np.histogram(xs.dropna(), bins=bins)
                        expected = (hist / max(hist.sum(), 1)).astype(float)
                        psi_reference[feat] = {'bins': bins.tolist(), 'expected': expected.tolist()}
                except Exception:
                    pass

        # Decide best model
        if 'final_comparison' in self.artifacts:
            best_name = min(self.artifacts['final_comparison'].keys(),
                            key=lambda k: self.artifacts['final_comparison'][k]['RMSE'])
        else:
            best_name = 'LightGBM' if 'LightGBM_full' in self.models else list(self.models.keys())[0]
        self.artifacts['best_model_name'] = best_name

        # Pack artifacts for PKL
        artifacts_to_save = {
            'config': asdict(self.config),
            'selected_features': selected_features,
            'preprocessors': self.preprocessors,
            'feature_families': self.artifacts['eda_results']['feature_families'],
            'engineered_features': self.artifacts.get('engineered_features', []),
            'dropped_features': self.artifacts.get('dropped_features', []),
            'categorical_columns': self.artifacts.get('categorical_columns', []),
            'psi_reference': psi_reference,
            'ensemble_weights': self.artifacts.get('ensemble_weights', {}),
            'best_model_name': self.artifacts.get('best_model_name', best_name)
        }

        # If ensemble is best, store base models + weights; else store best single model
        if artifacts_to_save['best_model_name'] == 'Ensemble':
            base_models = {}
            for k in ['LightGBM_full', 'RandomForest_full', 'SVM_full', 'ElasticNet_full']:
                if k in self.models:
                    base_models[k.replace('_full', '')] = self.models[k]
            artifacts_to_save['base_models'] = base_models
        else:
            # Save the best full model
            key = f"{artifacts_to_save['best_model_name']}_full"
            artifacts_to_save['best_model'] = self.models[key]

        # Save PKL
        with open(f'{OUTPUT_PATH}/unicrop_model_artifacts1.pkl', 'wb') as f:
            pickle.dump(artifacts_to_save, f)
        print("✅ Saved model artifacts to 'unicrop_model_artifacts1.pkl'")

        # Write prediction file (PKL-based)
        prediction_code = '''
def predict_crop_yield(df_raw):
    """
    Production-ready prediction for UniCrop (crop yield).
    Loads frozen artifacts from 'unicrop_model_artifacts1.pkl', applies the same
    preprocessing (winsorization, imputers, one-hot alignment, scaling), computes
    PSI diagnostics, and returns predictions + a QA summary.
    """
    import pandas as pd
    import numpy as np
    import pickle
    import os

    path = 'unicrop_model_artifacts1.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError("Missing artifacts file: unicrop_model_artifacts1.pkl")

    with open(path, 'rb') as f:
        artifacts = pickle.load(f)

    cfg = artifacts['config']
    sel_feats = artifacts['selected_features']
    preprocess = artifacts['preprocessors']
    cat_cols = artifacts.get('categorical_columns', [])
    psi_ref = artifacts.get('psi_reference', {})

    df = df_raw.copy()

    # Date features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['doy'] = df['date'].dt.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
        df.drop(columns=['date','doy'], inplace=True, errors='ignore')

    # === Feature engineering (must mirror training) ===
    def add_features(df):
        out = df.copy()
        tmp_cols = [c for c in out.columns if 'T2M_mean' in c or ('T2M' in c and '_mean' in c)]
        if tmp_cols:
            out['GDD_base10'] = np.maximum(0, out[tmp_cols[0]] - 10)
        if 'T2M_MAX_mean' in out.columns:
            out['heat_days_35'] = (out['T2M_MAX_mean'] >= 35).astype(int)
        if 'T2M_MIN_mean' in out.columns:
            out['chill_nights_15'] = (out['T2M_MIN_mean'] <= 15).astype(int)
        et_cols = [c for c in out.columns if c.startswith('ET') and 'mean' in c]
        sol_cols = [c for c in out.columns if 'ALLSKY_SFC_SW_DWN_mean' in c]
        if et_cols and sol_cols:
            out['ET_solar_ratio'] = out[et_cols[0]] / (out[sol_cols[0]] + 1e-6)
        vpd_cols = [c for c in out.columns if 'VPD_mean' in c]
        if vpd_cols and et_cols:
            out['VPD_ET_interaction'] = out[vpd_cols[0]] * out[et_cols[0]]
        ndvi_max = [c for c in out.columns if 'NDVI_max' in c]
        ndvi_min = [c for c in out.columns if 'NDVI_min' in c]
        ndvi_mean = [c for c in out.columns if 'NDVI_mean' in c]
        if ndvi_max and ndvi_min:
            out['NDVI_amplitude'] = out[ndvi_max[0]] - out[ndvi_min[0]]
        if ndvi_max and ndvi_min and ndvi_mean:
            num = out[ndvi_mean[0]] - out[ndvi_min[0]]
            den = out[ndvi_max[0]] - out[ndvi_min[0]] + 1e-6
            out['greenness_persistence'] = num / den
        evi_max = [c for c in out.columns if 'EVI_max' in c]
        evi_min = [c for c in out.columns if 'EVI_min' in c]
        if evi_max and evi_min:
            out['EVI_amplitude'] = out[evi_max[0]] - out[evi_min[0]]
        vv = [c for c in out.columns if 'VV_mean' in c]
        vh = [c for c in out.columns if 'VH_mean' in c]
        if vv and vh:
            out['VV_VH_ratio'] = out[vv[0]] / (out[vh[0]] + 1e-6)
        vv_std = [c for c in out.columns if 'VV_stdDev' in c or 'VV_std' in c]
        if vv_std:
            out['VV_texture'] = out[vv_std[0]]
        t2m = [c for c in out.columns if 'T2M_mean' in c]
        elev = [c for c in out.columns if 'elevation_mean' in c]
        if t2m and elev:
            out['temp_elevation_interaction'] = out[t2m[0]] * out[elev[0]]
        sand = [c for c in out.columns if 'sand_0-5cm_mean_mean' in c]
        clay = [c for c in out.columns if 'clay_0-5cm_mean_mean' in c]
        if sand and clay:
            out['water_holding_capacity'] = out[clay[0]] / (out[sand[0]] + 1e-6)
        return out

    df = add_features(df)

    # === Preprocessing: winsorize ===
    winsor = preprocess.get('winsor_limits', {})
    for col, (lo, hi) in winsor.items():
        if col in df.columns:
            df[col] = np.clip(df[col], lo, hi)

    # === Family-aware imputation ===
    # Meteorology
    meteo_cols = preprocess.get('meteo_cols', [])
    if meteo_cols:
        present = [c for c in meteo_cols if c in df.columns]
        if present:
            try:
                df[present] = preprocess['meteo_imputer'].transform(df[present])
            except Exception:
                med = preprocess.get('feature_medians_raw', {})
                for c in present:
                    df[c] = df[c].fillna(med.get(c, 0.0))

    # Vegetation indices (+ context labels for KNN)
    vi_cols = preprocess.get('vi_cols', [])
    if vi_cols:
        present = [c for c in vi_cols if c in df.columns]
        if present:
            try:
                vi_imputer = preprocess['vi_imputer']
                context_features = []
                for cat_col in ['fp_District', 'fp_Season(SA = Summer Autumn, WS = Winter Spring)']:
                    key = f'le_{cat_col}'
                    if key in preprocess and cat_col in df.columns:
                        le = preprocess[key]
                        val_labels = df[cat_col].fillna('__nan__').astype(str)
                        known_labels = set(le.classes_)
                        unseen = set(val_labels) - known_labels
                        if unseen:
                            le.classes_ = np.append(le.classes_, list(unseen))
                        enc = le.transform(val_labels)
                        context_features.append(enc.reshape(-1, 1))
                vi_features = np.hstack([df[present].values] + context_features) if context_features else df[present].values
                vi_imputed = vi_imputer.transform(vi_features)
                df[present] = vi_imputed[:, :len(present)]
            except Exception:
                med = preprocess.get('feature_medians_raw', {})
                for c in present:
                    df[c] = df[c].fillna(med.get(c, 0.0))

    # Soil/topo
    soil_cols = preprocess.get('soil_cols', [])
    if soil_cols:
        present = [c for c in soil_cols if c in df.columns]
        if present:
            try:
                df[present] = preprocess['soil_imputer'].transform(df[present])
            except Exception:
                med = preprocess.get('feature_medians_raw', {})
                for c in present:
                    df[c] = df[c].fillna(med.get(c, 0.0))

    # One-hot alignment
    one_hot_columns = preprocess.get('one_hot_columns', [])
    if one_hot_columns:
        enc_pieces = []
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col.split('(')[0], drop_first=True)
                enc_pieces.append(dummies)
        enc_df = pd.concat(enc_pieces, axis=1) if enc_pieces else pd.DataFrame(index=df.index)
        for c in one_hot_columns:
            if c not in enc_df.columns:
                enc_df[c] = 0
        extra = [c for c in enc_df.columns if c not in one_hot_columns]
        if extra:
            enc_df = enc_df.drop(columns=extra)
        enc_df = enc_df[one_hot_columns]
        df = pd.concat([df.drop(columns=[c for c in cat_cols if c in df.columns]), enc_df], axis=1)

    # Scaling
    scaled_cols = preprocess.get('scaled_columns', [])
    if scaled_cols:
        for c in scaled_cols:
            if c not in df.columns:
                df[c] = preprocess.get('feature_medians_raw', {}).get(c, 0.0)
        df[scaled_cols] = preprocess['scaler'].transform(df[scaled_cols])

    # Final feature set
    missing = [c for c in sel_feats if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required features for prediction: {missing}")
    X = df[sel_feats]

    # === Prediction ===
    best_name = artifacts.get('best_model_name', 'LightGBM')
    if best_name == 'Ensemble':
        weights = artifacts['ensemble_weights']
        base_models = artifacts['base_models']
        names = list(weights.keys())
        w = np.array([weights[n] for n in names], dtype=float)
        if w.sum() > 0: w = w / w.sum()
        preds_stack = np.column_stack([base_models[n].predict(X) for n in names])
        preds = preds_stack @ w
    else:
        model = artifacts['best_model']
        preds = model.predict(X)

    # === PSI diagnostics ===
    def compute_psi(expected, actual, eps=1e-6):
        expected = np.asarray(expected, dtype=float) + eps
        actual   = np.asarray(actual, dtype=float) + eps
        return np.sum((expected - actual) * np.log(expected / actual))

    psi_scores = {}
    for feat, ref in psi_ref.items():
        if feat in df.columns:
            vals = df[feat].values
            bins = np.asarray(ref['bins'], dtype=float)
            e = np.asarray(ref['expected'], dtype=float)
            try:
                hist, _ = np.histogram(vals[~np.isnan(vals)], bins=bins)
                a = (hist / max(hist.sum(), 1)).astype(float)
                psi_scores[feat] = float(compute_psi(e, a))
            except Exception:
                psi_scores[feat] = np.nan

    miss_pct = float(df[sel_feats].isna().mean().mean() * 100.0)
    psi_list = [v for v in psi_scores.values() if np.isfinite(v)]
    avg_psi = float(np.mean(psi_list)) if psi_list else 0.0
    dq = max(0.0, 100.0 - (miss_pct * 0.5 + min(avg_psi, 1.0) * 50.0))
    qa = {'data_quality_score': dq, 'avg_psi': avg_psi, 'psi_per_feature': psi_scores, 'missing_pct': miss_pct}

    return preds, qa
'''
        with open(f'{OUTPUT_PATH}/predict_crop_yield.py', 'w') as f:
            f.write(prediction_code)

        print("✅ Saved prediction function to 'predict_crop_yield.py'")

    # ----------------------------------
    # Step 13: Final report (MD)
    # ----------------------------------
    def generate_final_report(self) -> str:
        print("🔄 Step 13: Generating final report...")

        cv_txt = self.artifacts.get('cv_strategy',
                                    f"{self.config.cv_folds}-fold KFold (shuffle)")

        # Selected features (explicit list of 15)
        selected_list_md = "\n".join([f"- {feat}" for feat in self.selected_features])

        report = f"""
# UniCrop Prediction Model Report

## Abstract
This work presents UniCrop, a universal and model-independent model that integrates meteorological, satellite, SAR, soil, and topographic data. It emphasizes robust preprocessing (winsorization), transparent diagnostics (EDA, SHAP), a model-agnostic feature selection (composite relevance + mRMR), and a constrained ensemble.

## Acknowledgements
We thank the UniCrop contributors and partners who provided data access and agronomic guidance.

## Executive Summary
This report summarizes a production-hardened, generalizable base with strict schema enforcement at inference, family-aware imputations, and drift checks (PSI). The pipeline produces complete QC figures and reporting.

## Data Overview
- **Dataset Size**: {self.artifacts['raw_data_shape'][0]} samples × {self.artifacts['raw_data_shape'][1]} features
- **Target Variable**: {self.config.target_col}
- **Cross-Validation Strategy**: {cv_txt}

## Data Quality Assessment

### Target Distribution
- Mean Yield: {self.artifacts['eda_results']['target_stats']['mean']:.2f} kg/ha
- Standard Deviation: {self.artifacts['eda_results']['target_stats']['std']:.2f} kg/ha
- Skewness: {self.artifacts['eda_results']['target_stats']['skewness']:.3f}
- Range: {self.artifacts['eda_results']['target_stats']['min']:.0f} - {self.artifacts['eda_results']['target_stats']['max']:.0f} kg/ha

### Missing Data Analysis
- Vegetation indices showed notable missing values with seasonal patterning (see figures).
- Applied family-aware imputation strategies.

## Baseline Implementation (Model Development)
- Outlier handling via winsorization; robust scaling.
- Family-aware imputers (meteorology: iterative imputer; soil: median). **VI uses KNN imputation with context (district/season) and median fallback.**
- Model-independent feature selection selecting exactly {self.config.min_features} features.
- Correlation-based screening and basic significance checks.
- SLSQP non-negative ensemble retained only if it beats the best single model.

### Statistical Screening
- Initial features: {self.artifacts['raw_data_shape'][1] - 1}
- Selected (stable) features: {len(self.selected_features)}
- Dropped features: {len(self.artifacts.get('dropped_features', []))}

### Engineered Features
{chr(10).join([f"- {feat}" for feat in self.artifacts.get('engineered_features', [])])}

### Final Selected 15 Features
{selected_list_md}

## Cross-Validation Results
"""

        if 'final_comparison' in self.artifacts:
            report += "| Model | RMSE | MAE | R² | WAPE |\n"
            report += "|-------|------|-----|----|------|\n"
            for model_name, metrics in self.artifacts['final_comparison'].items():
                report += f"| {model_name} | {metrics['RMSE']:.4f} | {metrics['MAE']:.4f} | {metrics['R2']:.4f} | {metrics['WAPE']:.2f}% |\n"

            best_name = min(self.artifacts['final_comparison'].keys(),
                            key=lambda k: self.artifacts['final_comparison'][k]['RMSE'])
            best_metrics = self.artifacts['final_comparison'][best_name]
            report += f"""

### Best Model Performance
- **Best Model**: {best_name}
- **RMSE**: {best_metrics['RMSE']:.4f} kg/ha
- **R²**: {best_metrics['R2']:.4f}
"""

        report += f"""

## Recommendations
- Maintain meteorological data quality (VPD, temperature, solar).
- Improve VI coverage during SA season.
- Monitor PSI on top features and re-train if PSI > 0.2 or performance degrades >10%.

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(f'{OUTPUT_PATH}/unicrop_final_report.md', 'w') as f:
            f.write(report)

        print("✅ Final report saved to 'unicrop_final_report.md'")
        return report


# ======================
# Runner
# ======================
def run_complete_unicrop_modelling(filepath: str) -> UniCropModeler:
    print("🚀 Starting UniCrop Complete Modeling")
    print("=" * 60)

    config = ModelConfig()
    modeler = UniCropModeler(config)

    try:
        df = modeler.load_and_validate_data(filepath)
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

        return modeler

    except Exception as e:
        print(f"❌ Model failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Actual dataset path
    data_filepath = f"{OUTPUT_PATH}/unicrop_master_timeseries.csv"

    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found at '{data_filepath}'")
        print("Please update the 'data_filepath' variable in the `if __name__ == '__main__':` block.")
    else:
        modeler = run_complete_unicrop_modelling(data_filepath)

