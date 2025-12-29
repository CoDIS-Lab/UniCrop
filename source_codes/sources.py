# =========================
# UniCropDataPipeline (NASA)
# =========================

import re
import time
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple, Callable
from datetime import timedelta, datetime
import ee
import os
from source_codes.config import *
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class UniCropNASAPipeline:
    """NASA POWER data pipeline for UniCrop project."""

    DERIVED_VARIABLE_MAPPING: Dict[str, List[str]] = {
        "DTR": ["T2M_MAX", "T2M_MIN"],
        "VPD": ["T2M", "RH2M"],
        "DEW POINT TEMPERATURE": ["T2M", "RH2M"],
    }

    def __init__(self, config: Optional[NASAConfig] = None):
        """Initialize pipeline with configuration."""
        self.config = config or NASAConfig()

        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.config.user_agent})

        # Create derived formulas using config constants
        self.DERIVED_FORMULAS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
            "DTR": lambda df: df["T2M_MAX"] - df["T2M_MIN"],
            "DEW POINT TEMPERATURE": lambda df: (
                    (self.config.magnus_b * (
                            (self.config.magnus_a * df["T2M"] / (self.config.magnus_b + df["T2M"])) +
                            np.log(df["RH2M"] / 100)
                    )) / (
                            self.config.magnus_a - (
                            (self.config.magnus_a * df["T2M"] / (self.config.magnus_b + df["T2M"])) +
                            np.log(df["RH2M"] / 100)
                    )
                    )
            ),
            "VPD": lambda df: (
                    (0.6108 * np.exp((17.27 * df["T2M"]) / (df["T2M"] + 237.3))) * (1 - (df["RH2M"] / 100))
            ),
        }

    # ---------- helpers ----------
    @staticmethod
    def _col_ci(df: pd.DataFrame, name_lc: str) -> Optional[str]:
        m = {c.strip().lower(): c for c in df.columns}
        return m.get(name_lc)

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """Validate coordinates using config bounds."""
        return (self.config.lat_min <= float(lat) <= self.config.lat_max and
                self.config.lon_min <= float(lon) <= self.config.lon_max)

    def _exp_backoff(self, fetch_func: Callable):
        """Exponential backoff with configurable parameters."""
        for i in range(self.config.max_retries):
            try:
                return fetch_func()
            except requests.exceptions.RequestException as e:
                delay = self.config.initial_delay * (2 ** i)
                print(f"HTTP error (attempt {i + 1}/{self.config.max_retries}): {e} — retry in {delay}s")
                time.sleep(delay)
            except Exception as e:
                delay = self.config.initial_delay * (2 ** i)
                print(f"Unexpected error (attempt {i + 1}/{self.config.max_retries}): {e} — retry in {delay}s")
                time.sleep(delay)
        print("❌ API call failed after retries.")
        return None

    @staticmethod
    def _normalize_var_key(name: str) -> str:
        n = re.sub(r"[^A-Z ]", "", str(name).upper())
        n = re.sub(r"\s+", " ", n).strip()
        if "DEW POINT" in n: return "DEW POINT TEMPERATURE"
        if "DIURNAL" in n or "DTR" in n: return "DTR"
        if "VPD" in n: return "VPD"
        return n

    @staticmethod
    def _find_harvest_col(df: pd.DataFrame) -> str:
        for c in df.columns:
            cl = c.lower()
            if "date" in cl and "harvest" in cl:
                return c
        for name in ["Date of harvest", "Harvest date", "date_of_harvest", "harvest_date"]:
            for c in df.columns:
                if c.strip().lower() == name.lower():
                    return c
        low_map = {c.lower(): c for c in df.columns}
        if "date" in low_map:
            return low_map["date"]
        raise ValueError("Could not find a harvest date column in fetch_plan.")

    @staticmethod
    def _clean_params_string(s: str) -> List[str]:
        toks = re.split(r"[;,]", str(s)) if pd.notna(s) else []
        out = []
        for t in toks:
            t = t.strip().upper()
            if not t or t == "DERIVED":
                continue
            out.append(t)
        return out

    # ---------- fetch ----------
    def fetch_nasa_power_data(self, latitude: float, longitude: float, start_date: str, end_date: str,
                              api_identifiers: List[str]) -> Optional[Dict[str, Dict[str, float]]]:
        """Fetch NASA POWER data using config settings."""
        start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
        end_str = pd.to_datetime(end_date).strftime("%Y%m%d")

        params = {
            "parameters": ",".join(sorted(set(api_identifiers))),
            "community": self.config.community,
            "longitude": float(longitude),
            "latitude": float(latitude),
            "start": start_str,
            "end": end_str,
            "format": "JSON",
        }
        print(f"   Fetching NASA POWER: {params['parameters']} @ ({latitude},{longitude}) {start_str}..{end_str}")
        resp = self.session.get(self.config.base_url, params=params, timeout=self.config.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()

        fetched: Dict[str, Dict[str, float]] = {}
        daily = data.get("properties", {}).get("parameter", {})
        for param_name, date_values in daily.items():
            for yyyymmdd, val in date_values.items():
                date_iso = pd.to_datetime(yyyymmdd, format="%Y%m%d").strftime("%Y-%m-%d")
                fetched.setdefault(date_iso, {})[param_name] = (np.nan if val == -999 else val)
        return fetched

    def calculate_derived(self, df: pd.DataFrame, derived_variables: List[str]) -> pd.DataFrame:
        """Calculate derived variables using configured formulas."""
        for var in derived_variables:
            bases = self.DERIVED_VARIABLE_MAPPING.get(var, [])
            if bases and all(b in df.columns for b in bases):
                try:
                    df[var] = self.DERIVED_FORMULAS[var](df)
                    print(f"     ✅ Derived: {var}")
                except Exception as e:
                    print(f"     ⚠️ Could not compute '{var}': {e}")
            else:
                if var in self.DERIVED_VARIABLE_MAPPING:
                    print(f"     ⚠️ Missing bases for '{var}': {bases}")
        return df

    # ---------- main ----------
    def run_pipeline(self,
                     fetch_plan_path: str = "fetch_plan.csv",
                     daily_output_file: Optional[str] = None,
                     stats_output_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the NASA POWER data pipeline with configurable parameters."""

        # Use default filenames from config if not provided
        daily_output_file = daily_output_file or self.config.default_daily_output
        stats_output_file = stats_output_file or self.config.default_stats_output

        print(f"\n=== UniCrop Data Pipeline - NASA POWER (window ±{self.config.stats_window_days}d) ===")
        fp = pd.read_csv(fetch_plan_path, encoding="latin1")
        fp.columns = fp.columns.str.strip()
        print(f"✅ Fetch plan loaded: {fetch_plan_path} ({len(fp)} rows)")

        lat_col = self._col_ci(fp, "latitude")
        lon_col = self._col_ci(fp, "longitude")
        if not lat_col or not lon_col:
            raise ValueError("fetch_plan must have 'latitude' and 'longitude' columns.")
        src_col = self._col_ci(fp, "source dataset") or self._col_ci(fp, "dataset")
        if not src_col:
            raise ValueError("fetch_plan must have 'Source Dataset' or 'Dataset' column.")

        harv_col = self._find_harvest_col(fp)
        fp[harv_col] = pd.to_datetime(fp[harv_col], errors="coerce")
        if fp[harv_col].isna().all():
            raise ValueError(f"Harvest date column '{harv_col}' could not be parsed to dates.")

        fp[src_col] = fp[src_col].astype(str).str.strip().str.lower()
        fp[lat_col] = pd.to_numeric(fp[lat_col], errors="coerce")
        fp[lon_col] = pd.to_numeric(fp[lon_col], errors="coerce")

        plan = fp[fp[src_col].str.contains("nasa", na=False)].copy()
        if plan.empty:
            print("ℹ️ No rows with source dataset containing 'nasa'.")
            return pd.DataFrame(), pd.DataFrame()

        groups = plan[[lat_col, lon_col, harv_col]].drop_duplicates()

        daily_rows = []
        stats_rows = []

        for _, g in groups.iterrows():
            lat = float(g[lat_col]);
            lon = float(g[lon_col])
            anchor_dt = pd.to_datetime(g[harv_col])
            anchor = anchor_dt.strftime("%Y-%m-%d")
            print(
                f"\n--- Location @ ({lat:.4f}, {lon:.4f}) | harvest {anchor} | window ±{self.config.stats_window_days}d")

            if not self.validate_coordinates(lat, lon):
                print("   Skipping invalid coordinates.")
                continue

            sub = plan[
                (plan[lat_col] == g[lat_col]) &
                (plan[lon_col] == g[lon_col]) &
                (plan[harv_col] == g[harv_col])
                ].copy()

            api_set: set[str] = set()
            derived_list: set[str] = set()
            api_param_col = self._col_ci(sub, "api parameter")
            variable_col = self._col_ci(sub, "variable")

            for _, r in sub.iterrows():
                if variable_col and pd.notna(r.get(variable_col, None)):
                    var = self._normalize_var_key(r[variable_col])
                    if var in self.DERIVED_VARIABLE_MAPPING:
                        derived_list.add(var)
                        api_set.update(self.DERIVED_VARIABLE_MAPPING[var])
                if api_param_col and pd.notna(r.get(api_param_col, None)):
                    for tok in self._clean_params_string(r[api_param_col]):
                        api_set.add(tok)

            if not api_set:
                print("   ⚠️ No valid NASA POWER parameters to fetch here.")
                continue

            if isinstance(self.config.stats_window_days, int) and self.config.stats_window_days > 0:
                start_dt = anchor_dt - pd.Timedelta(days=self.config.stats_window_days)
                end_dt = anchor_dt + pd.Timedelta(days=self.config.stats_window_days)
            else:
                start_dt = end_dt = anchor_dt

            fetched = self._exp_backoff(lambda: self.fetch_nasa_power_data(
                lat, lon,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
                sorted(api_set)
            ))
            if not fetched:
                print("   ❌ Fetch failed for this location.")
                continue

            rows = [{"date": d, **vals} for d, vals in fetched.items()]
            if not rows:
                print("   ⚠️ No daily records parsed.")
                continue

            df_all = pd.DataFrame(rows)
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
            df_all[lat_col] = lat
            df_all[lon_col] = lon
            df_all.rename(columns={lat_col: "latitude", lon_col: "longitude"}, inplace=True)

            df_all = self.calculate_derived(df_all, sorted(derived_list))

            # DAILY (exact or nearest within window), stamped as harvest date in output
            exact = df_all[df_all["date"] == anchor_dt]
            if exact.empty:
                nearest_idx = (df_all["date"] - anchor_dt).abs().sort_values().index[:1]
                daily_pick = df_all.loc[nearest_idx].copy()
                daily_pick["date"] = anchor_dt
            else:
                daily_pick = exact.copy()
            daily_rows.append(daily_pick)

            # STATS over window
            value_cols = [c for c in df_all.columns if c not in ["latitude", "longitude", "date"]]
            stat_row = {"latitude": lat, "longitude": lon, "date": anchor}
            for c in value_cols:
                s = pd.to_numeric(df_all[c], errors="coerce")
                if s.notna().any():
                    stat_row[f"{c}_mean"] = s.mean()
                    stat_row[f"{c}_min"] = s.min()
                    stat_row[f"{c}_max"] = s.max()
                    stat_row[f"{c}_std"] = s.std(ddof=0)
            stats_rows.append(stat_row)
            time.sleep(self.config.request_delay)

        if not daily_rows:
            print("\n❌ No daily data collected.")
            return pd.DataFrame(), pd.DataFrame()

        daily_df = (pd.concat(daily_rows, ignore_index=True)
                    .sort_values(["latitude", "longitude", "date"])
                    .reset_index(drop=True))
        daily_df.to_csv(daily_output_file, index=False)
        print(f"\n✅ Daily (harvest-only) saved → {daily_output_file} ({len(daily_df)} rows)")

        stats = pd.DataFrame(stats_rows)
        if not stats.empty:
            std_cols = [c for c in stats.columns if c.endswith("_std")]
            drop_std = [c for c in std_cols if pd.to_numeric(stats[c], errors="coerce").isna().all()]
            if drop_std:
                stats.drop(columns=drop_std, inplace=True)
                print(f"   ℹ️ Dropped empty std columns: {drop_std}")
            stats.to_csv(stats_output_file, index=False)
            print(f"✅ Window statistics saved → {stats_output_file} ({len(stats)} rows)")
        else:
            print("❌ No window stats produced.")

        return daily_df, stats

class UniCropS2ModisPipeline:
    """
    Complete S2/MODIS pipeline class with configurable parameters.
    Handles Sentinel-2 and MODIS data processing with statistics generation.
    """

    def __init__(
            self,
            config: S2ModisConfig = None,
            drop_rows_all_zero_stats: bool = True,
            project_id: str = 'unicrop-466421',
            **kwargs
    ):
        self.config = config or S2ModisConfig()
        self.drop_rows_all_zero_stats = drop_rows_all_zero_stats
        self.project_id = project_id

        # Allow override of config values via kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    # -------- Helper methods (general) --------
    @staticmethod
    def _find_first(df: pd.DataFrame, must_contain: list[str]) -> str | None:
        for c in df.columns:
            name = c.strip().lower()
            if all(k in name for k in must_contain):
                return c
        return None

    @staticmethod
    def _find_harvest_date_col(df: pd.DataFrame) -> str | None:
        col = UniCropS2ModisPipeline._find_first(df, ["date", "harvest"])
        if col:
            return col
        variants = ["date of harvest", "harvest date", "date_of_harvest", "harvest_date", "date"]
        low = {c.lower(): c for c in df.columns}
        for v in variants:
            if v in low:
                return low[v]
        return None

    @staticmethod
    def _lower(s):
        return None if pd.isna(s) else str(s).strip().lower()

    @staticmethod
    def _split_csv_like(s: str | None) -> list[str]:
        """Split on commas/semicolons/pipes/slashes; also strip parens and extra spaces."""
        if not s or pd.isna(s):
            return []
        t = re.sub(r'[\/|;]', ',', str(s))
        t = re.sub(r'[\(\)]', ' ', t)
        toks = [x.strip() for x in t.split(',') if x and x.strip()]
        return toks

    # -------- GEE initialization --------
    def init_gee(self):
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            print("✅ GEE initialized successfully.")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            raise

    # -------- Sentinel-2 preprocessing & derived indices --------
    @staticmethod
    def mask_scl_clouds(image):
        scl = image.select('SCL')
        cloud_shadow, water = 3, 6
        cloud_low, cloud_med, cloud_high, cirrus, snow = 8, 9, 10, 11, 11
        mask = (scl.neq(cloud_shadow)
                .And(scl.neq(cloud_low))
                .And(scl.neq(cloud_med))
                .And(scl.neq(cloud_high))
                .And(scl.neq(cirrus))
                .And(scl.neq(snow))
                .And(scl.neq(water)))
        return image.updateMask(mask)

    @staticmethod
    def add_s2_indices(image: ee.Image) -> ee.Image:
        """Add NDVI, EVI, LAI, SAVI, NDRE, CIredge (reflectance in [0,1])."""
        b2 = image.select('B2').divide(10000)  # Blue
        b4 = image.select('B4').divide(10000)  # Red
        b5 = image.select('B5').divide(10000)  # Red-edge
        b8 = image.select('B8').divide(10000)  # NIR

        ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI')

        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
            {'NIR': b8, 'RED': b4, 'BLUE': b2}
        ).rename('EVI')

        lai = ndvi.multiply(3.618).subtract(0.118).rename('LAI')

        savi = image.expression(
            '(1 + L) * (NIR - RED) / (NIR + RED + L)',
            {'NIR': b8, 'RED': b4, 'L': 0.5}
        ).rename('SAVI')

        ndre = b8.subtract(b5).divide(b8.add(b5)).rename('NDRE')
        ciredge = b8.divide(b5).subtract(1).rename('CIredge')

        return image.addBands([ndvi, evi, lai, savi, ndre, ciredge])

    @classmethod
    def preprocess_s2(cls, image):
        return cls.add_s2_indices(cls.mask_scl_clouds(image))

    # -------- Normalize S2 "derived" keywords --------
    @staticmethod
    def _normalize_s2_derived_token(tok: str) -> str | None:
        t = tok.strip().lower()
        m = {
            "ndvi": "NDVI", "evi": "EVI", "lai": "LAI", "savi": "SAVI",
            "ndre": "NDRE", "ci": "CIredge", "ci-rededge": "CIredge",
            "ci_rededge": "CIredge", "ciredge": "CIredge"
        }
        if t in m: return m[t]
        if "chlorophyll" in t and "edge" in t: return "CIredge"
        if "red" in t and "edge" in t and "nd" in t: return "NDRE"
        return None

    def _extract_s2_requests(self, grp: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Return (raw_bands, derived_bands) requested for S2."""
        raw_bands, derived = set(), set()
        wants_all = False
        for _, r in grp.iterrows():
            api = str(r.get("api parameter") or "")
            var = str(r.get("variable") or "")
            if "derived" in api.lower():
                wants_all = True
            for tok in self._split_csv_like(api) + self._split_csv_like(var):
                if re.fullmatch(r'B\d{1,2}A?', tok.strip(), flags=re.I):
                    raw_bands.add(tok.strip())
                else:
                    nm = self._normalize_s2_derived_token(tok)
                    if nm: derived.add(nm)
        if wants_all and not derived:
            derived.update(self.config.all_s2_derived)
        raw = sorted(raw_bands, key=lambda x: (len(x), x))
        der = [b for b in self.config.all_s2_derived if b in derived]
        return raw, der

    # -------- MODIS helpers --------
    @staticmethod
    def _which_product(src_value: str) -> str | None:
        """Version-agnostic product router (accepts 006/061 and suffix noise)."""
        s = (src_value or "").lower()
        if "copernicus/s2" in s or "sentinel-2" in s: return "S2"
        if "mod13q1" in s: return "MOD13Q1"
        if "mod15a2h" in s: return "MOD15A2H"
        if "mod16a2" in s: return "MOD16A2"
        if "mod17a2h" in s: return "MOD17A2H"
        return None

    @staticmethod
    def _wants_vci(grp: pd.DataFrame) -> bool:
        for _, r in grp.iterrows():
            api = str(r.get("api parameter") or "")
            var = str(r.get("variable") or "")
            if api.strip().lower() == "derived" and ("vci" in var.lower() or "vegetation condition" in var.lower()):
                return True
            for tok in UniCropS2ModisPipeline._split_csv_like(api):
                if tok.strip().lower() == "vci":
                    return True
        return False

    @staticmethod
    def _assign(row: dict, key: str, val, omit_zero: bool = False):
        """Write val into row[key], optionally skipping literal 0.0 for plain means."""
        if val is None:
            return
        try:
            if omit_zero and float(val) == 0.0:
                return
        except Exception:
            pass
        row[key] = val

    @staticmethod
    def _select_stats_case_insensitive(image: ee.Image, region, requested_bands: List[str], scale: int):
        """
        Case-insensitive selection + mean/min/max; keys are returned using the
        REQUESTED band tokens (not the dataset's exact case), e.g. 'ndvi_mean'.
        """
        if not requested_bands or image is None:
            return {}
        try:
            avail = image.bandNames().getInfo() or []
            m = {b.lower(): b for b in avail}
            pairs = [(tok, m[tok.lower()]) for tok in requested_bands if tok and tok.lower() in m]
            if not pairs:
                return {}
            exacts = [ex for _, ex in pairs]
            img = image.select(exacts)
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), '', True),
                geometry=region,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4
            ).getInfo() or {}

            # Remap from exact keys back to requested token keys
            rev = {ex: tok for tok, ex in pairs}
            out = {}
            for ex in exacts:
                tok = rev[ex]
                for suf in ["mean", "min", "max"]:
                    k_src = f"{ex}_{suf}"
                    if k_src in stats:
                        out[f"{tok}_{suf}"] = stats[k_src]
            return out
        except Exception:
            return {}

    # -------- Utilities --------
    @staticmethod
    def get_closest_image_from_collection(collection, date_iso, point, tolerance_days=3):
        target = ee.Date(date_iso)
        subset = (collection
                  .filterDate(target.advance(-tolerance_days, 'day'),
                              target.advance(tolerance_days, 'day'))
                  .filterBounds(point)
                  .sort('system:time_start', True))
        if subset.size().getInfo() == 0:
            return None
        return ee.Image(subset.first())

    # -------- Main processing method --------
    def run_gee_s2_modis(
            self,
            fetch_plan_path='fetch_plan.csv',
            output_csv_path='S2_MODIS_timeseries.csv',
            drop_rows_all_zero_stats: bool = None
    ):
        if drop_rows_all_zero_stats is None:
            drop_rows_all_zero_stats = self.drop_rows_all_zero_stats

        # Load & normalize plan
        fp = pd.read_csv(fetch_plan_path)
        fp.columns = fp.columns.str.strip()

        # detect harvest date col
        date_col = self._find_harvest_date_col(fp)
        if date_col is None:
            raise ValueError("Could not find a harvest/Date column in fetch_plan.csv")

        # lat/lon
        lat_col = self._find_first(fp, ["lat"])
        lon_col = self._find_first(fp, ["lon"])
        if lat_col is None or lon_col is None:
            raise ValueError("fetch_plan.csv must contain latitude/longitude columns.")

        df = fp.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

        # choose the source column that exists
        lower_map = {c.strip().lower(): c for c in df.columns}
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            raise ValueError("fetch_plan.csv needs a 'Source Dataset' or 'Dataset' column.")

        df[src_col] = df[src_col].apply(self._lower)
        if "api parameter" in df.columns:
            df["api parameter"] = df["api parameter"].apply(lambda x: None if pd.isna(x) else str(x).strip())
        if "variable" in df.columns:
            df["variable"] = df["variable"].apply(lambda x: None if pd.isna(x) else str(x).strip())

        # Keep only S2/MODIS rows (accepts COPERNICUS/S2)
        df = df[df[src_col].str.contains(r"modis|copernicus/s2|sentinel-2", case=False, na=False)]

        groups = df[[date_col, "latitude", "longitude"]].drop_duplicates() \
            .sort_values([date_col, "latitude", "longitude"])

        results = []
        print("\n--- GEE (Sentinel-2 + MODIS) — means + statistics; fallback if empty ---")

        for _, g in groups.iterrows():
            d_iso = pd.to_datetime(g[date_col]).strftime("%Y-%m-%d")
            lat = float(g["latitude"])
            lon = float(g["longitude"])
            print(f"\n→ ({lat:.4f},{lon:.4f}) • {d_iso}")

            sub = df[(df[date_col] == g[date_col]) &
                     (df["latitude"] == lat) &
                     (df["longitude"] == lon)].copy()

            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(self.config.buffer_m)

            # collections lazily constructed on demand
            s2_col = None
            mod13 = mod15 = mod16 = mod17 = None

            row_vals = {'date': d_iso, 'latitude': lat, 'longitude': lon}

            for src, grp in sub.groupby(src_col):
                product = self._which_product(src)
                if product is None:
                    continue

                # --- Sentinel-2 ---
                if product == "S2":
                    if s2_col is None:
                        s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                  .filterBounds(point)
                                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.config.max_cloud_percentage))
                                  .map(self.preprocess_s2))
                    s2_raw, s2_der = self._extract_s2_requests(grp)
                    want = s2_raw + s2_der
                    if want:
                        # nearest scene
                        im = self.get_closest_image_from_collection(s2_col, d_iso, point,
                                                                    tolerance_days=self.config.s2_date_tolerance_days)
                        stats = self._select_stats_case_insensitive(im, region, want,
                                                                    self.config.s2_scale) if im else {}

                        # fallback: median over window (if nearest is empty)
                        if not stats:
                            target = ee.Date(d_iso)
                            subset = s2_col.filterDate(target.advance(-self.config.s2_date_tolerance_days, 'day'),
                                                       target.advance(self.config.s2_date_tolerance_days, 'day'))
                            if subset.size().getInfo() > 0:
                                im_med = subset.median()
                                stats = self._select_stats_case_insensitive(im_med, region, want, self.config.s2_scale)

                        # write plain mean (from stats) + stats
                        for b in {k for k in want}:
                            self._assign(row_vals, b, stats.get(f"{b}_mean"),
                                         omit_zero=self.config.omit_zero_in_plain_mean)
                            self._assign(row_vals, f"{b}_mean", stats.get(f"{b}_mean"))
                            self._assign(row_vals, f"{b}_min", stats.get(f"{b}_min"))
                            self._assign(row_vals, f"{b}_max", stats.get(f"{b}_max"))

                # --- MOD13Q1 (NDVI/EVI & baseline for VCI) ---
                elif product == "MOD13Q1":
                    api_bands = []
                    for _, r in grp.iterrows():
                        api_bands += self._split_csv_like(r.get("api parameter"))
                    want_vci = self._wants_vci(grp)
                    if api_bands or want_vci:
                        if mod13 is None:
                            mod13 = ee.ImageCollection("MODIS/061/MOD13Q1").filterBounds(point)
                        im = self.get_closest_image_from_collection(mod13, d_iso, point,
                                                                    tolerance_days=self.config.modis_date_tolerance_days)
                        stats = self._select_stats_case_insensitive(im, region, api_bands, self.config.modis_scale) if (
                                    im and api_bands) else {}

                        if api_bands and not stats:
                            target = ee.Date(d_iso)
                            subset = mod13.filterDate(target.advance(-self.config.modis_date_tolerance_days, 'day'),
                                                      target.advance(self.config.modis_date_tolerance_days, 'day'))
                            if subset.size().getInfo() > 0:
                                im_med = subset.median()
                                stats = self._select_stats_case_insensitive(im_med, region, api_bands,
                                                                            self.config.modis_scale)

                        # write MOD13 bands
                        for b in {k for k in api_bands}:
                            self._assign(row_vals, b, stats.get(f"{b}_mean"),
                                         omit_zero=self.config.omit_zero_in_plain_mean)
                            self._assign(row_vals, f"{b}_mean", stats.get(f"{b}_mean"))
                            self._assign(row_vals, f"{b}_min", stats.get(f"{b}_min"))
                            self._assign(row_vals, f"{b}_max", stats.get(f"{b}_max"))

                        # VCI if requested
                        if want_vci:
                            vcur = row_vals.get('NDVI')
                            if vcur is None and im:
                                s_ndvi = self._select_stats_case_insensitive(im, region, ['NDVI'],
                                                                             self.config.modis_scale)
                                vcur = s_ndvi.get('NDVI_mean')
                                if vcur is not None:
                                    self._assign(row_vals, 'NDVI', vcur, omit_zero=self.config.omit_zero_in_plain_mean)
                                    self._assign(row_vals, 'NDVI_mean', vcur)

                            if vcur is not None:
                                year = int(d_iso[:4])
                                y0 = ee.Date.fromYMD(year, 1, 1)
                                y1 = y0.advance(1, 'year')
                                year_nd = mod13.filterDate(y0, y1).select('NDVI')
                                ndvi_min_img = year_nd.min()
                                ndvi_max_img = year_nd.max()
                                s_min = self._select_stats_case_insensitive(ndvi_min_img, region, ['NDVI'],
                                                                            self.config.modis_scale)
                                s_max = self._select_stats_case_insensitive(ndvi_max_img, region, ['NDVI'],
                                                                            self.config.modis_scale)
                                ndmin = s_min.get('NDVI_mean')
                                ndmax = s_max.get('NDVI_mean')
                                try:
                                    if ndmin is not None and ndmax is not None:
                                        denom = float(ndmax) - float(ndmin)
                                        if denom != 0.0:
                                            vci = (float(vcur) - float(ndmin)) / denom
                                            self._assign(row_vals, 'VCI', vci)
                                except Exception as e:
                                    print(f"   (VCI calc issue: {e})")

                # --- MOD15A2H ---
                elif product == "MOD15A2H":
                    bands = []
                    for _, r in grp.iterrows():
                        bands += self._split_csv_like(r.get("api parameter"))
                    if bands:
                        if mod15 is None:
                            mod15 = ee.ImageCollection("MODIS/061/MOD15A2H").filterBounds(point)
                        im = self.get_closest_image_from_collection(mod15, d_iso, point,
                                                                    tolerance_days=self.config.modis_date_tolerance_days)
                        stats = self._select_stats_case_insensitive(im, region, bands,
                                                                    self.config.modis_scale) if im else {}
                        if not stats:
                            target = ee.Date(d_iso)
                            subset = mod15.filterDate(target.advance(-self.config.modis_date_tolerance_days, 'day'),
                                                      target.advance(self.config.modis_date_tolerance_days, 'day'))
                            if subset.size().getInfo() > 0:
                                im_med = subset.median()
                                stats = self._select_stats_case_insensitive(im_med, region, bands,
                                                                            self.config.modis_scale)
                        for b in {k for k in bands}:
                            self._assign(row_vals, b, stats.get(f"{b}_mean"),
                                         omit_zero=self.config.omit_zero_in_plain_mean)
                            self._assign(row_vals, f"{b}_mean", stats.get(f"{b}_mean"))
                            self._assign(row_vals, f"{b}_min", stats.get(f"{b}_min"))
                            self._assign(row_vals, f"{b}_max", stats.get(f"{b}_max"))

                # --- MOD16A2 ---
                elif product == "MOD16A2":
                    bands = []
                    for _, r in grp.iterrows():
                        bands += self._split_csv_like(r.get("api parameter"))
                    if bands:
                        if mod16 is None:
                            mod16 = ee.ImageCollection("MODIS/061/MOD16A2").filterBounds(point)
                        im = self.get_closest_image_from_collection(mod16, d_iso, point,
                                                                    tolerance_days=self.config.modis_et_date_tolerance_days)
                        stats = self._select_stats_case_insensitive(im, region, bands,
                                                                    self.config.modis_scale) if im else {}
                        if not stats:
                            target = ee.Date(d_iso)
                            subset = mod16.filterDate(target.advance(-self.config.modis_et_date_tolerance_days, 'day'),
                                                      target.advance(self.config.modis_et_date_tolerance_days, 'day'))
                            if subset.size().getInfo() > 0:
                                im_med = subset.median()
                                stats = self._select_stats_case_insensitive(im_med, region, bands,
                                                                            self.config.modis_scale)
                        for b in {k for k in bands}:
                            self._assign(row_vals, b, stats.get(f"{b}_mean"),
                                         omit_zero=self.config.omit_zero_in_plain_mean)
                            self._assign(row_vals, f"{b}_mean", stats.get(f"{b}_mean"))
                            self._assign(row_vals, f"{b}_min", stats.get(f"{b}_min"))
                            self._assign(row_vals, f"{b}_max", stats.get(f"{b}_max"))

                # --- MOD17A2H (Gpp etc.) ---
                elif product == "MOD17A2H":
                    bands = []
                    for _, r in grp.iterrows():
                        bands += self._split_csv_like(r.get("api parameter"))
                    if bands:
                        if mod17 is None:
                            mod17 = ee.ImageCollection("MODIS/061/MOD17A2H").filterBounds(point)
                        im = self.get_closest_image_from_collection(mod17, d_iso, point,
                                                                    tolerance_days=self.config.modis_date_tolerance_days)
                        stats = self._select_stats_case_insensitive(im, region, bands,
                                                                    self.config.modis_scale) if im else {}
                        if not stats:
                            target = ee.Date(d_iso)
                            subset = mod17.filterDate(target.advance(-self.config.modis_date_tolerance_days, 'day'),
                                                      target.advance(self.config.modis_date_tolerance_days, 'day'))
                            if subset.size().getInfo() > 0:
                                im_med = subset.median()
                                stats = self._select_stats_case_insensitive(im_med, region, bands,
                                                                            self.config.modis_scale)
                        for b in {k for k in bands}:
                            self._assign(row_vals, b, stats.get(f"{b}_mean"),
                                         omit_zero=self.config.omit_zero_in_plain_mean)
                            self._assign(row_vals, f"{b}_mean", stats.get(f"{b}_mean"))
                            self._assign(row_vals, f"{b}_min", stats.get(f"{b}_min"))
                            self._assign(row_vals, f"{b}_max", stats.get(f"{b}_max"))

            results.append(row_vals)

        if not results:
            print("\nNo S2/MODIS data collected.")
            return pd.DataFrame()

        out = pd.DataFrame(results).sort_values('date').reset_index(drop=True)

        # Optional interpolation for simple numeric columns (plain means only)
        to_interp = [c for c in out.columns if (re.fullmatch(r'B\d{1,2}A?', c) or c in self.config.all_s2_derived
                                                or c in ['NDVI', 'EVI', 'VCI', 'ET', 'Fpar_500m', 'Gpp',
                                                         'PsnNet'] or c.startswith('ET'))]
        for col in to_interp:
            out[col] = pd.to_numeric(out[col], errors='coerce')
            out[col] = out[col].interpolate(method='linear', limit_direction='both')

        # Drop rows where ALL statistics columns are numerically zero
        if drop_rows_all_zero_stats:
            stat_cols = [c for c in out.columns if re.search(r'_(mean|min|max)$', c)]
            if stat_cols:
                num = out[stat_cols].apply(pd.to_numeric, errors='coerce')
                mask_has_any = num.notna().any(axis=1)
                mask_all_zero = num.fillna(0).eq(0).all(axis=1)
                to_keep = ~(mask_has_any & mask_all_zero)
                dropped = (~to_keep).sum()
                if dropped:
                    print(f"🧹 Dropping {int(dropped)} rows with all-zero statistics.")
                out = out.loc[to_keep].reset_index(drop=True)

        out.to_csv(output_csv_path, index=False)
        print(f"\n✅ Saved → {output_csv_path}  (rows={len(out)})")
        return out

    # -------- Pipeline-style wrapper --------
    def run_pipeline(
            self,
            fetch_plan_path: str = "fetch_plan.csv",
            timeseries_output_file: str = "S2_MODIS_timeseries.csv",
            stats_output_file: str | None = None,
    ):
        """
        Wrapper around run_gee_s2_modis so you can call it like the NASA pipeline.
        - timeseries_output_file: the single CSV (means + *_mean/_min/_max)
        - stats_output_file: optional, a second CSV with only stats columns extracted
        """
        # Run the stage (writes the full file with means + stats)
        full_df = self.run_gee_s2_modis(
            fetch_plan_path=fetch_plan_path,
            output_csv_path=timeseries_output_file,
            drop_rows_all_zero_stats=self.drop_rows_all_zero_stats,
        )

        # Build a stats-only view (keys + *_mean/_min/_max columns)
        if full_df is None or full_df.empty:
            stats_df = pd.DataFrame()
        else:
            stat_cols = ["date", "latitude", "longitude"] + [
                c for c in full_df.columns if re.search(r"_(mean|min|max)$", c)
            ]
            stats_df = full_df[stat_cols].copy() if len(stat_cols) > 3 else pd.DataFrame()

        # Optionally write stats-only CSV
        if stats_output_file and not stats_df.empty:
            stats_df.to_csv(stats_output_file, index=False)
            print(f"🧾 Stats-only file saved → {stats_output_file} ({len(stats_df)} rows, {len(stats_df.columns)} cols)")

        print(
            f"✅ S2/MODIS pipeline done | Timeseries: {full_df.shape if full_df is not None else (0, 0)} "
            f"| Stats-only: {stats_df.shape if stats_df is not None else (0, 0)}"
        )
        return full_df, stats_df

class UniCropSentinel1Pipeline:
    """Sentinel-1 GEE data pipeline for UniCrop project."""

    def __init__(self, config: S1Config = None,
                 project_id: str = 'unicrop-466421',
                 **kwargs
                 ):
        """Initialize pipeline with configuration."""
        self.config = config or S1Config()
        self.project_id = project_id

        # Allow override of config values via kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize GEE
        self._ensure_gee()

    def _ensure_gee(self):
        """Initialize EE only if needed (safe to call every time)."""
        try:
            ee.Image(0).getInfo()
            return
        except Exception:
            pass
        try:
            ee.Initialize(project=self.project_id)
            ee.Image(0).getInfo()
            print("GEE initialized.")
            return
        except Exception:
            self._init_gee()

    def _init_gee(self):
        """Initialize GEE with authentication."""
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            print("GEE initialized successfully.")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            raise

    # ---------- Plan parsing helpers ----------
    @staticmethod
    def _find_first(df: pd.DataFrame, must_contain: List[str]) -> Optional[str]:
        """Find first column containing all specified strings."""
        for c in df.columns:
            name = c.strip().lower()
            if all(k in name for k in must_contain):
                return c
        return None

    @staticmethod
    def _find_harvest_date_col(df: pd.DataFrame) -> Optional[str]:
        """Find harvest date column in various formats."""
        col = UniCropSentinel1Pipeline._find_first(df, ["date", "harvest"])
        if col:
            return col
        variants = ["date of harvest", "harvest date", "date_of_harvest", "harvest_date", "date"]
        low = {c.lower(): c for c in df.columns}
        for v in variants:
            if v in low:
                return low[v]
        return None

    @staticmethod
    def _split_csv_like(s: str | None) -> List[str]:
        """Split CSV-like strings, but preserve VV/VH ratio expressions."""
        if not s or pd.isna(s):
            return []

        s_str = str(s).strip()

        # Protect VV/VH patterns by temporarily replacing them
        protected = s_str
        vv_vh_patterns = [
            r'VV/VH\s+ratio', r'vv/vh\s+ratio', r'VV/VH', r'vv/vh',
            r'S1\s+VV/VH', r's1\s+vv/vh'
        ]

        temp_replacements = {}
        for i, pattern in enumerate(vv_vh_patterns):
            matches = re.findall(pattern, protected, re.IGNORECASE)
            for match in matches:
                placeholder = f"__VVVH_PLACEHOLDER_{i}__"
                temp_replacements[placeholder] = match
                protected = protected.replace(match, placeholder)

        # Now safe to split on remaining delimiters (no '/')
        t = re.sub(r'[|;]', ',', protected)
        t = re.sub(r'[\(\)]', ' ', t)

        # Split and restore placeholders
        toks = []
        for token in t.split(','):
            token = token.strip()
            if not token:
                continue
            # Restore protected patterns
            for placeholder, original in temp_replacements.items():
                token = token.replace(placeholder, original)
            toks.append(token)

        return toks

    # ---------- Sentinel-1 token mapping ----------
    @staticmethod
    def _token_to_group(token: str) -> Optional[str]:
        """Map a token to a metric group: 'VV', 'VH', 'VVVH_ratio', 'RVI'."""
        if not token:
            return None

        # Clean the token but preserve key characters
        original = str(token).strip()
        t = original.lower().replace(' ', '').replace('-', '').replace('_', '')

        # Enhanced ratio detection - be more permissive
        ratio_indicators = ['ratio', '/', 'over', 'div']
        vv_indicators = ['vv']
        vh_indicators = ['vh']

        has_ratio = any(ind in t for ind in ratio_indicators)
        has_vv = any(ind in t for ind in vv_indicators)
        has_vh = any(ind in t for ind in vh_indicators)

        # If it mentions ratio and both VV and VH, it's VVVH_ratio
        if has_ratio and has_vv and has_vh:
            return 'VVVH_ratio'

        # Specific ratio patterns
        ratio_patterns = [
            'vvvhratio', 'vv/vh', 'vvovervh', 'vvvhlinear',
            'vvvhratio', 'ratiovvvh', 'vvratio', 'vhvvratio',
            's1vv/vhratio', 's1vvvhratio', 'vvvhratio linear',
            'vvdivvh', 'vvvhdiv'
        ]

        if any(pattern in t for pattern in ratio_patterns):
            return 'VVVH_ratio'

        # RVI patterns
        if t.startswith('rvi') or 'radarvegetationindex' in t:
            return 'RVI'

        # Individual polarizations (but not if it's clearly a ratio)
        if t.startswith('vv') and not has_vh and not has_ratio:
            return 'VV'
        if t.startswith('vh') and not has_vv and not has_ratio:
            return 'VH'

        return None

    @staticmethod
    def _is_derived_flag(val: str | None) -> bool:
        """Check if value is a derived flag."""
        return isinstance(val, str) and val.strip().lower() == "derived"

    def _parse_requested_groups(self, location_plan: pd.DataFrame) -> List[str]:
        """Enhanced semantics for S1 with better automatic detection."""
        apicol = next((c for c in location_plan.columns if c.strip().lower() == 'api parameter'), None)
        varcol = next((c for c in location_plan.columns if c.strip().lower() == 'variable'), None)

        any_derived = False
        if apicol is not None:
            any_derived = any(self._is_derived_flag(location_plan.loc[i, apicol]) for i in location_plan.index)

        groups: set[str] = set()

        print(f"   DEBUG: API column = {apicol}, Variable column = {varcol}")
        print(f"   DEBUG: Any derived flag = {any_derived}")

        if any_derived:
            print("   DERIVED mode: Processing Variable column")
            if varcol is not None:
                for _, r in location_plan.iterrows():
                    var_content = r.get(varcol)
                    print(f"      Variable content: '{var_content}'")
                    for tok in self._split_csv_like(var_content):
                        print(f"         Token: '{tok}'")
                        g = self._token_to_group(tok)
                        print(f"         Mapped to: {g}")
                        if g:  # Accept ALL detected groups in derived mode
                            groups.add(g)

            # In derived mode, ALWAYS ensure we have base polarizations when we need derived metrics
            derived_metrics_present = any(derived_var in groups for derived_var in ['VVVH_ratio', 'RVI'])
            if derived_metrics_present or groups:  # If any groups detected in derived mode
                groups.update(['VV', 'VH'])  # Always add base polarizations
                print("   Added VV, VH base polarizations (required for derived calculations)")

            # If no groups detected but we're in derived mode, use defaults
            if not groups:
                print("   No variables detected in derived mode, using default S1 variables")
                groups.update(self.config.default_variables)
        else:
            print("   STANDARD mode: Processing API Parameter column")
            if apicol is not None:
                for _, r in location_plan.iterrows():
                    api_content = r.get(apicol)
                    print(f"      API Parameter content: '{api_content}'")
                    for tok in self._split_csv_like(api_content):
                        print(f"         Token: '{tok}'")
                        g = self._token_to_group(tok)
                        print(f"         Mapped to: {g}")
                        if g:
                            groups.add(g)

            if not groups and varcol is not None:
                print("   Fallback to Variable column")
                for _, r in location_plan.iterrows():
                    var_content = r.get(varcol)
                    print(f"      Variable content: '{var_content}'")
                    for tok in self._split_csv_like(var_content):
                        print(f"         Token: '{tok}'")
                        g = self._token_to_group(tok)
                        print(f"         Mapped to: {g}")
                        if g:
                            groups.add(g)

            # Use all available S1 variables if nothing specified
            if not groups:
                print("   No variables specified, using default S1 variables")
                groups.update(self.config.default_variables)

        print(f"   Final groups selected: {sorted(groups)}")
        return sorted(groups)

    def _expand_groups_to_stats(self, groups: List[str]) -> List[str]:
        """Generate all statistics for each variable group automatically."""
        keys = []
        for g in groups:
            for s in self.config.statistics:
                keys.append(f'{g}_{s}')
        return keys

    @staticmethod
    def _bands_needed(groups: List[str]) -> Dict[str, bool]:
        """Determine which bands are needed for the requested groups."""
        need_vv = 'VV' in groups
        need_vh = 'VH' in groups
        need_ratio = 'VVVH_ratio' in groups
        need_rvi = 'RVI' in groups
        if need_ratio or need_rvi:
            need_vv = True
            need_vh = True
        return dict(need_vv=need_vv, need_vh=need_vh, need_ratio=need_ratio, need_rvi=need_rvi)

    # ---------- GEE processing functions ----------
    @staticmethod
    def _despeckle_overwrite(image: ee.Image) -> ee.Image:
        """Apply speckle filtering to S1 image."""
        vv_f = image.select('VV').focal_median()
        vh_f = image.select('VH').focal_median()
        filtered = ee.Image.cat([vv_f.rename('VV'), vh_f.rename('VH')])
        return image.addBands(filtered, overwrite=True)

    def _reduce_stats_s1(self, image_all: ee.Image, region: ee.Geometry, band_names: List[str] = None) -> Dict[
        str, float]:
        """Reduce image statistics over region."""
        img = image_all.select(band_names) if band_names else image_all
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), '', True)
            .combine(ee.Reducer.minMax(), '', True),
            geometry=region,
            scale=self.config.scale,
            maxPixels=self.config.max_pixels,
            bestEffort=self.config.best_effort
        ).getInfo() or {}
        return stats

    @staticmethod
    def _closest_image(collection: ee.ImageCollection, target_date_iso: str) -> ee.Image:
        """Get closest image to target date."""
        target = ee.Date(target_date_iso)

        def add_diff(im):
            diff = im.date().difference(target, 'day').abs()
            return im.set('diff', diff)

        col = collection.map(add_diff).sort('diff')
        return ee.Image(col.first())

    def _build_final_image_from_collection(self, col: ee.ImageCollection,
                                           mode: str,
                                           target_date_iso: str,
                                           needs: Dict[str, bool]) -> Optional[ee.Image]:
        """Build an image containing only the requested bands using specified aggregation mode."""
        im = self._closest_image(col, target_date_iso) if mode == "nearest" else col.median()
        if im is None:
            return None

        bands = []
        # VV/VH in dB for direct stats:
        if needs['need_vv']:
            bands.append(im.select('VV').rename('VV'))
        if needs['need_vh']:
            bands.append(im.select('VH').rename('VH'))

        # Derived from linear backscatter (convert dB → linear σ⁰):
        if needs['need_ratio'] or needs['need_rvi']:
            # Convert from dB to linear backscatter coefficient
            vv_linear = ee.Image(10).pow(im.select('VV').divide(10.0))
            vh_linear = ee.Image(10).pow(im.select('VH').divide(10.0))

            if needs['need_ratio']:
                # VV/VH ratio from linear σ⁰
                vvvh_ratio = vv_linear.divide(vh_linear).rename('VVVH_ratio')
                bands.append(vvvh_ratio)
                print("   VVVH_ratio band created and added")

            if needs['need_rvi']:
                # Radar Vegetation Index: RVI = 4*VH/(VV+VH) from linear σ⁰
                rvi = vh_linear.multiply(4.0).divide(vv_linear.add(vh_linear)).rename('RVI')
                bands.append(rvi)

        if not bands:
            return None
        return ee.Image.cat(bands)

    # ---------- Automatic data creation ----------
    def create_default_fetch_plan(self, coordinates_list: List[tuple], dates_list: List[str] = None) -> pd.DataFrame:
        """Create a default fetch plan for Sentinel-1 when no CSV is available."""
        if dates_list is None:
            dates_list = [datetime.now().strftime('%Y-%m-%d')] * len(coordinates_list)

        if len(dates_list) != len(coordinates_list):
            # Repeat dates if needed
            dates_list = dates_list * (len(coordinates_list) // len(dates_list) + 1)
            dates_list = dates_list[:len(coordinates_list)]

        data = []
        for i, ((lat, lon), date) in enumerate(zip(coordinates_list, dates_list)):
            data.append({
                'Source Dataset': 'Sentinel-1',
                'API Parameter': 'Derived',
                'Variable': 'VV, VH, VV/VH ratio, RVI',
                'Harvest Date': date,
                'Latitude': lat,
                'Longitude': lon,
                'Location_ID': f'S1_Point_{i + 1}'
            })

        return pd.DataFrame(data)

    # ---------- Debug functions ----------
    def debug_fetch_plan_parsing(self, fetch_plan_path='fetch_plan.csv'):
        """Debug function to see exactly how your fetch_plan.csv is being parsed."""
        print("DEBUGGING FETCH PLAN PARSING")
        print("=" * 50)

        if not os.path.exists(fetch_plan_path):
            print(f"File not found: {fetch_plan_path}")
            return

        # Load the file
        fp = pd.read_csv(fetch_plan_path)
        print(f"Loaded file with {len(fp)} rows and {len(fp.columns)} columns")
        print(f"Columns: {list(fp.columns)}")
        print()

        # Clean column names
        fp.columns = fp.columns.str.strip()
        lower_map = {c.strip().lower(): c for c in fp.columns}
        print(f"Column mapping (lowercase -> original):")
        for k, v in lower_map.items():
            print(f"   '{k}' -> '{v}'")
        print()

        # Find dataset column
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            print("Could not find 'Source Dataset' or 'Dataset' column")
            return

        print(f"Dataset column: '{src_col}'")

        # Filter for S1 rows
        df = fp.copy()
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()
        print(f"Dataset values found: {df[src_col].unique()}")

        pattern = '|'.join(self.config.dataset_patterns)
        s1_df = df[df[src_col].str.contains(pattern, case=False, na=False)].copy()
        print(f"Sentinel-1 rows: {len(s1_df)}")

        if s1_df.empty:
            print("No Sentinel-1 rows found!")
            print("Try using one of these dataset names:")
            for pattern in self.config.dataset_patterns:
                print(f"   - '{pattern}'")
            return

        # Show first few S1 rows
        print(f"\nFirst {min(3, len(s1_df))} Sentinel-1 rows:")
        for i, (_, row) in enumerate(s1_df.head(3).iterrows()):
            print(f"   Row {i + 1}:")
            for col in s1_df.columns:
                print(f"      {col}: '{row[col]}'")
            print()

        # Test variable parsing
        print("TESTING VARIABLE PARSING")
        print("-" * 30)

        # Take first S1 location
        first_row = s1_df.iloc[0:1].copy()
        groups = self._parse_requested_groups(first_row)

        print(f"Final result: {groups}")

        if 'VVVH_ratio' not in groups:
            print("\nVVVH_ratio NOT detected!")
            print("Possible solutions:")
            print("   1. Set 'API Parameter' to 'Derived'")
            print("   2. Include 'VV/VH ratio' or 'VVVH_ratio' in Variable column")
            print("   3. Add 'ratio' keyword to your variable description")
        else:
            print("\nVVVH_ratio successfully detected!")

    def test_vv_vh_ratio_parsing(self) -> bool:
        """Test function to verify VV/VH ratio parsing works correctly."""
        # Create a test DataFrame simulating your fetch plan
        test_data = {
            'Source Dataset': ['Sentinel-1'],
            'API Parameter': ['Derived'],
            'Variable': ['S1 VV/VH ratio (linear σ⁰)'],
            'Date of Harvest': ['2023-06-15'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060]
        }
        test_df = pd.DataFrame(test_data)

        print("Testing VV/VH ratio parsing with your exact setup:")
        print("API Parameter: 'Derived'")
        print("Variable: 'S1 VV/VH ratio (linear σ⁰)'")
        print()

        # Test the parsing
        groups = self._parse_requested_groups(test_df)

        print(f"Result: {groups}")

        if 'VVVH_ratio' in groups and 'VV' in groups and 'VH' in groups:
            print("SUCCESS: VV/VH ratio detection working correctly!")
            print("- VVVH_ratio detected for the ratio calculation")
            print("- VV and VH detected as base polarizations")
            return True
        else:
            print("PROBLEM: VV/VH ratio not detected correctly")
            return False

    # ---------- Main pipeline ----------
    def run_pipeline(self,
                     fetch_plan_path: str = 'fetch_plan.csv',
                     output_csv_path: Optional[str] = None,
                     coordinates_list: List[tuple] = None,
                     dates_list: List[str] = None) -> pd.DataFrame:
        """Run Sentinel-1 analysis pipeline with full automation."""

        output_csv_path = output_csv_path or self.config.default_output_file

        # Automatic fetch plan creation
        if not os.path.exists(fetch_plan_path) and self.config.auto_create_plan and coordinates_list:
            print(f"No fetch plan found at {fetch_plan_path}")
            print("Creating automatic fetch plan for Sentinel-1...")
            fp = self.create_default_fetch_plan(coordinates_list, dates_list)
            print(f"Created automatic fetch plan with {len(fp)} locations")
            # Optionally save the created plan
            auto_plan_path = fetch_plan_path.replace('.csv', self.config.auto_plan_suffix)
            fp.to_csv(auto_plan_path, index=False)
        else:
            # Load existing fetch plan
            if not os.path.exists(fetch_plan_path):
                raise FileNotFoundError(
                    f"fetch_plan.csv not found at {fetch_plan_path} and auto_create_plan is disabled")
            fp = pd.read_csv(fetch_plan_path)

        fp.columns = fp.columns.str.strip()
        lower_map = {c.strip().lower(): c for c in fp.columns}
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            raise ValueError("fetch_plan.csv must contain 'Source Dataset' or 'Dataset'.")

        date_col = self._find_harvest_date_col(fp)
        if date_col is None:
            raise ValueError("Could not find a harvest/Date column in fetch_plan.csv.")
        lat_col = self._find_first(fp, ["lat"])
        lon_col = self._find_first(fp, ["lon"])
        if lat_col is None or lon_col is None:
            raise ValueError("fetch_plan.csv must contain latitude/longitude columns.")

        df = fp.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()

        # Find the variable column early for enhanced detection
        varcol = None
        for c in df.columns:
            if 'variable' in c.lower():
                varcol = c
                break
        # Accept common dataset labels for S1 - Enhanced: More permissive detection
        dataset_pattern = '|'.join(self.config.dataset_patterns)
        s1_condition = df[src_col].str.contains(dataset_pattern, case=False, na=False)

        # Also check if Variable column mentions S1/Sentinel-1/radar terms
        if varcol and varcol in df.columns:
            var_pattern = '|'.join(self.config.variable_patterns)
            var_condition = df[varcol].str.contains(var_pattern, case=False, na=False)
            s1_condition = s1_condition | var_condition

        s1_df = df[s1_condition].copy()

        if s1_df.empty:
            print("No Sentinel-1 rows found in fetch plan.")
            print("Checked for these patterns in Source Dataset:")
            print(f"  - {', '.join(self.config.dataset_patterns)}")
            print("Also checked Variable column for S1/radar/VV/VH mentions")
            return pd.DataFrame()

        groups = s1_df[[date_col, "latitude", "longitude"]].drop_duplicates() \
            .sort_values([date_col, "latitude", "longitude"]).reset_index(drop=True)

        print(f"\n--- Sentinel-1 (GEE) • {self.config.aggregation_mode} within ±{self.config.window_days} days ---")
        print(f"Processing {len(groups)} unique location-date combinations")
        results = []

        for idx, g in groups.iterrows():
            center = pd.to_datetime(g[date_col])
            start = (center - timedelta(days=self.config.window_days)).strftime("%Y-%m-%d")
            end = (center + timedelta(days=self.config.window_days)).strftime("%Y-%m-%d")
            target = center.strftime("%Y-%m-%d")
            lat = float(g["latitude"])
            lon = float(g["longitude"])

            print(f"\n[{idx + 1}/{len(groups)}] ({lat:.4f},{lon:.4f}) | {start} .. {end} (target {target})")

            loc_plan = s1_df[(s1_df[date_col] == g[date_col]) &
                             (s1_df["latitude"] == lat) &
                             (s1_df["longitude"] == lon)].copy()

            groups_req = self._parse_requested_groups(loc_plan)
            requested_keys = self._expand_groups_to_stats(groups_req)
            needs = self._bands_needed(groups_req)
            print(f"   Variables: {groups_req}")
            print(f"   Stats: {len(requested_keys)} total ({', '.join(self.config.statistics)} for each)")

            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(self.config.buffer_m)

            s1_col = (
                ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterDate(start, end)
                .filterBounds(point)
                .filter(ee.Filter.eq('instrumentMode', self.config.instrument_mode))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .filter(ee.Filter.eq('resolution_meters', self.config.resolution_meters))
                .map(lambda im: im.select(self.config.polarisations))
                .map(self._despeckle_overwrite)
                .sort('system:time_start')
            )

            image_count = s1_col.size().getInfo()
            print(f"   Found {image_count} Sentinel-1 images")

            if image_count == 0:
                print("   No S1 images found in the window; skipping.")
                continue

            im_final = self._build_final_image_from_collection(s1_col, self.config.aggregation_mode, target, needs)
            if im_final is None:
                print("   Nothing to compute for requested groups; skipping.")
                continue

            print("   Computing statistics...")
            print(f"   Bands in final image: {im_final.bandNames().getInfo()}")
            stats = self._reduce_stats_s1(im_final, region)
            print(f"   Raw stats keys: {list(stats.keys())}")

            row = {'date': target, 'latitude': lat, 'longitude': lon}
            stats_found = 0
            for k in requested_keys:
                v = stats.get(k, None)
                if v is None:
                    print(f"   Missing stat: {k}")
                    continue
                try:
                    if float(v) == 0.0:
                        continue
                except Exception:
                    pass
                row[k] = v
                stats_found += 1

            if stats_found > 0:
                results.append(row)
                print(f"   Extracted {stats_found} statistics")
            else:
                print("   No valid statistics extracted")

        if not results:
            print("\nNo Sentinel-1 data produced.")
            return pd.DataFrame()

        df_out = pd.DataFrame(results).sort_values(['date', 'latitude', 'longitude']).reset_index(drop=True)
        df_out['date'] = pd.to_datetime(df_out['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df_out.to_csv(output_csv_path, index=False)

        print(f"\nS1 time series saved to {output_csv_path}")
        print(f"Final dataset: {len(df_out):,} rows x {len(df_out.columns):,} columns")

        # Print summary of variables extracted
        variable_cols = [col for col in df_out.columns if col not in ['date', 'latitude', 'longitude']]
        unique_vars = set([col.split('_')[0] for col in variable_cols])
        print(f"Variables extracted: {sorted(unique_vars)}")
        print(f"Column names: {list(df_out.columns)}")

        return df_out


class UniCropSRTMPipeline:
    """SRTM GEE data pipeline for UniCrop project."""

    def __init__(self, config: Optional[SRTMConfig] = None,
                 project_id: str = 'unicrop-466421',
                 **kwargs
                 ):
        """Initialize pipeline with configuration."""
        self.config = config or SRTMConfig()
        self.project_id = project_id

        # Allow override of config values via kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize GEE
        self._ensure_gee()

    def _ensure_gee(self):
        """Initialize EE only if needed (safe to call any time)."""
        try:
            ee.Image(0).getInfo()
            return
        except Exception:
            pass
        try:
            ee.Initialize(project=self.project_id)
            ee.Image(0).getInfo()
            print("✅ GEE initialized.")
            return
        except Exception:
            self._init_gee()

    def _init_gee(self):
        """Initialize GEE with authentication."""
        try:
            ee.Authenticate()
            ee.Initialize(project=self.config.project_id)
            print("✅ GEE initialized successfully.")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            raise

    # ---------- Plan parsing helpers ----------
    @staticmethod
    def _find_first(df: pd.DataFrame, must_contain: List[str]) -> Optional[str]:
        """Return first column whose name contains ALL tokens (case-insensitive)."""
        for c in df.columns:
            name = c.strip().lower()
            if all(k in name for k in must_contain):
                return c
        return None

    @staticmethod
    def _find_harvest_date_col(df: pd.DataFrame) -> Optional[str]:
        """Find harvest date column in various formats."""
        col = UniCropSRTMPipeline._find_first(df, ["date", "harvest"])
        if col:
            return col
        variants = ["date of harvest", "harvest date", "date_of_harvest", "harvest_date", "date"]
        low = {c.lower(): c for c in df.columns}
        for v in variants:
            if v in low:
                return low[v]
        return None

    @staticmethod
    def _split_csv_like(s: str | None) -> List[str]:
        """Split on commas / slashes / pipes / semicolons; strip parens/spaces."""
        if not s or pd.isna(s):
            return []
        t = re.sub(r'[\/|;]', ',', str(s))
        t = re.sub(r'[\(\)]', ' ', t)
        toks = [x.strip() for x in t.split(',') if x and x.strip()]
        return toks

    def _norm_srtm_token(self, token: str) -> Optional[str]:
        """Map tokens to valid SRTM bands with configurable aliases and suffixes."""
        if not token:
            return None
        t = str(token).strip().lower().replace(' ', '')

        # Remove stat suffixes
        for suf in self.config.stat_suffixes:
            if t.endswith(suf):
                t = t[:-len(suf)]
                break

        return self.config.band_aliases.get(t, None)

    @staticmethod
    def _is_derived_flag(val: str | None) -> bool:
        """Check if value is a derived flag."""
        return isinstance(val, str) and val.strip().lower() == "derived"

    def _requested_srtm_bands(self, location_plan: pd.DataFrame) -> List[str]:
        """
        NASA-like semantics for SRTM:
          • If ANY row has API Parameter == 'Derived': use bands named in 'Variable'
          • Else: use bands named in 'API Parameter'
          • Fallback to 'Variable' if needed; else default to all bands
        """
        apicol = next((c for c in location_plan.columns if c.strip().lower() == 'api parameter'), None)
        varcol = next((c for c in location_plan.columns if c.strip().lower() == 'variable'), None)

        any_derived = False
        if apicol is not None:
            any_derived = any(self._is_derived_flag(location_plan.loc[i, apicol]) for i in location_plan.index)

        chosen_cols: List[str] = []
        if any_derived and varcol:
            chosen_cols = [varcol]
        elif apicol:
            chosen_cols = [apicol]
        elif varcol:
            chosen_cols = [varcol]

        tokens: List[str] = []
        for col in chosen_cols:
            for _, r in location_plan.iterrows():
                for tok in self._split_csv_like(r.get(col)):
                    b = self._norm_srtm_token(tok)
                    if b:
                        tokens.append(b)

        want = list(dict.fromkeys([b for b in tokens if b in self.config.valid_bands]))  # unique, keep order
        if not want:
            want = self.config.valid_bands.copy()

        print(f"   Requested SRTM bands: {want}")
        return want

    # ---------- SRTM processing ----------
    def _extract_srtm_stats(self, region: ee.Geometry, bands_to_fetch: List[str]) -> Dict:
        """Compute mean/stdDev/min/max for requested SRTM bands at native scale."""
        if not bands_to_fetch:
            return {}

        dem = ee.Image(self.config.srtm_dataset)
        terrain = ee.Terrain.products(dem).updateMask(dem.mask())

        if self.config.use_native_scale:
            native_proj = dem.projection()
            native_scale = native_proj.nominalScale()
            img = terrain.select(bands_to_fetch).reproject(native_proj)
            scale = native_scale
        else:
            img = terrain.select(bands_to_fetch)
            scale = None

        stats = img.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), '', True)
            .combine(ee.Reducer.minMax(), '', True),
            geometry=region,
            scale=scale,  # ~30 m if using native scale
            maxPixels=self.config.max_pixels,
            bestEffort=self.config.best_effort,
            tileScale=self.config.tile_scale
        ).getInfo() or {}

        # Clean up problematic statistics
        cleaned = dict(stats)

        # Drop stdDev that are 0/None if configured
        if self.config.exclude_zero_stddev:
            cleaned = {
                k: v for k, v in cleaned.items()
                if not (k.endswith('_stdDev') and (v is None or (isinstance(v, (int, float)) and float(v) == 0.0)))
            }

        # Remove explicitly excluded stats
        for exclude_stat in self.config.exclude_stats:
            cleaned.pop(exclude_stat, None)

        return cleaned

    # ---------- Auto-creation of fetch plans ----------
    def create_default_fetch_plan(self, coordinates_list: List[tuple], dates_list: List[str] = None) -> pd.DataFrame:
        """Create a default fetch plan for SRTM when no CSV is available."""
        from datetime import datetime

        if dates_list is None:
            dates_list = [datetime.now().strftime('%Y-%m-%d')] * len(coordinates_list)

        if len(dates_list) != len(coordinates_list):
            # Repeat dates if needed
            dates_list = dates_list * (len(coordinates_list) // len(dates_list) + 1)
            dates_list = dates_list[:len(coordinates_list)]

        data = []
        for i, ((lat, lon), date) in enumerate(zip(coordinates_list, dates_list)):
            data.append({
                'Source Dataset': 'SRTM',
                'API Parameter': 'Derived',
                'Variable': 'elevation, slope, aspect, hillshade',
                'Harvest Date': date,
                'Latitude': lat,
                'Longitude': lon,
                'Location_ID': f'SRTM_Point_{i + 1}'
            })

        return pd.DataFrame(data)

    # ---------- Debug functions ----------
    def debug_fetch_plan_parsing(self, fetch_plan_path: str = 'fetch_plan.csv'):
        """Debug function to see exactly how your fetch_plan.csv is being parsed."""
        print("DEBUGGING SRTM FETCH PLAN PARSING")
        print("=" * 50)

        import os
        if not os.path.exists(fetch_plan_path):
            print(f"File not found: {fetch_plan_path}")
            return

        # Load the file
        fp = pd.read_csv(fetch_plan_path)
        print(f"Loaded file with {len(fp)} rows and {len(fp.columns)} columns")
        print(f"Columns: {list(fp.columns)}")
        print()

        # Clean column names
        fp.columns = fp.columns.str.strip()
        lower_map = {c.strip().lower(): c for c in fp.columns}
        print(f"Column mapping (lowercase -> original):")
        for k, v in lower_map.items():
            print(f"   '{k}' -> '{v}'")
        print()

        # Find dataset column
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            print("Could not find 'Source Dataset' or 'Dataset' column")
            return

        print(f"Dataset column: '{src_col}'")

        # Filter for SRTM rows
        df = fp.copy()
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()
        print(f"Dataset values found: {df[src_col].unique()}")

        pattern = '|'.join(self.config.dataset_patterns)
        srtm_df = df[df[src_col].str.contains(pattern, case=False, na=False)].copy()
        print(f"SRTM rows: {len(srtm_df)}")

        if srtm_df.empty:
            print("No SRTM rows found!")
            print("Try using one of these dataset names:")
            for pattern in self.config.dataset_patterns:
                print(f"   - '{pattern.upper()}'")
            return

        # Show first few SRTM rows
        print(f"\nFirst {min(3, len(srtm_df))} SRTM rows:")
        for i, (_, row) in enumerate(srtm_df.head(3).iterrows()):
            print(f"   Row {i + 1}:")
            for col in srtm_df.columns:
                print(f"      {col}: '{row[col]}'")
            print()

        # Test variable parsing
        print("TESTING VARIABLE PARSING")
        print("-" * 30)

        # Take first SRTM location
        first_row = srtm_df.iloc[0:1].copy()
        bands = self._requested_srtm_bands(first_row)

        print(f"Final result: {bands}")

        if not bands:
            print("\nNo bands detected!")
            print("Possible solutions:")
            print("   1. Set 'API Parameter' to 'Derived'")
            print("   2. Include band names in Variable column (elevation, slope, aspect, hillshade)")
            print("   3. Check band name spelling")
        else:
            print(f"\nSRTM bands successfully detected: {bands}")

    def test_band_parsing(self) -> bool:
        """Test function to verify SRTM band parsing works correctly."""
        # Create a test DataFrame simulating your fetch plan
        test_data = {
            'Source Dataset': ['SRTM'],
            'API Parameter': ['Derived'],
            'Variable': ['elevation, slope, aspect, hillshade'],
            'Date of Harvest': ['2023-06-15'],
            'Latitude': [40.7128],
            'Longitude': [-74.0060]
        }
        test_df = pd.DataFrame(test_data)

        print("Testing SRTM band parsing with test setup:")
        print("API Parameter: 'Derived'")
        print("Variable: 'elevation, slope, aspect, hillshade'")
        print()

        # Test the parsing
        bands = self._requested_srtm_bands(test_df)

        print(f"Result: {bands}")

        expected = ['elevation', 'slope', 'aspect', 'hillshade']
        if set(bands) == set(expected):
            print("SUCCESS: SRTM band detection working correctly!")
            print(f"- All expected bands detected: {expected}")
            return True
        else:
            print("PROBLEM: SRTM band detection not working correctly")
            print(f"Expected: {expected}")
            print(f"Got: {bands}")
            return False

    # ---------- Main pipeline ----------
    def run_pipeline(self,
                     fetch_plan_path: str = 'fetch_plan.csv',
                     output_csv_path: Optional[str] = None,
                     coordinates_list: List[tuple] = None,
                     dates_list: List[str] = None) -> pd.DataFrame:
        """Run SRTM analysis pipeline with full automation."""

        output_csv_path = output_csv_path or self.config.default_output_file

        import os
        # Automatic fetch plan creation
        if not os.path.exists(fetch_plan_path) and self.config.auto_create_plan and coordinates_list:
            print(f"No fetch plan found at {fetch_plan_path}")
            print("Creating automatic fetch plan for SRTM...")
            fp = self.create_default_fetch_plan(coordinates_list, dates_list)
            print(f"Created automatic fetch plan with {len(fp)} locations")
            # Optionally save the created plan
            auto_plan_path = fetch_plan_path.replace('.csv', self.config.auto_plan_suffix)
            fp.to_csv(auto_plan_path, index=False)
        else:
            # Load existing fetch plan
            if not os.path.exists(fetch_plan_path):
                raise FileNotFoundError(
                    f"fetch_plan.csv not found at {fetch_plan_path} and auto_create_plan is disabled")
            fp = pd.read_csv(fetch_plan_path)

        fp.columns = fp.columns.str.strip()

        # Find required columns
        date_col = self._find_harvest_date_col(fp)
        if date_col is None:
            raise ValueError("Could not find a harvest/Date column in fetch_plan.csv.")

        lat_col = self._find_first(fp, ["lat"])
        lon_col = self._find_first(fp, ["lon"])
        if lat_col is None or lon_col is None:
            raise ValueError("fetch_plan.csv must contain latitude/longitude columns.")

        # Source column: prefer 'Source Dataset', else 'Dataset'
        lower_map = {c.strip().lower(): c for c in fp.columns}
        if 'source dataset' in lower_map:
            src_col = lower_map['source dataset']
        elif 'dataset' in lower_map:
            src_col = lower_map['dataset']
        else:
            raise ValueError("fetch_plan.csv needs a 'Source Dataset' or 'Dataset' column.")

        # Normalize types
        df = fp.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
        df['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()

        # Keep only SRTM rows
        pattern = '|'.join(self.config.dataset_patterns)
        srtm_df = df[df[src_col].str.contains(pattern, na=False)].copy()
        if srtm_df.empty:
            print("ℹ️ No SRTM rows found in fetch plan.")
            print(f"Looked for patterns: {self.config.dataset_patterns}")
            return pd.DataFrame()

        # Unique groups by (harvest date, lat, lon)
        groups = (
            srtm_df[[date_col, 'latitude', 'longitude']]
            .drop_duplicates()
            .sort_values([date_col, 'latitude', 'longitude'])
            .reset_index(drop=True)
        )

        print(f"\n--- GEE SRTM — static stats per (harvest_date, lat, lon) | Buffer: {self.config.buffer_m}m ---")
        print(f"Processing {len(groups)} unique location-date combinations")
        t0 = time.time()
        results = []

        for idx, g in groups.iterrows():
            d_iso = pd.to_datetime(g[date_col]).strftime('%Y-%m-%d')
            lat = float(g['latitude'])
            lon = float(g['longitude'])
            print(f"[{idx + 1}/{len(groups)}] → ({lat:.4f},{lon:.4f}) • {d_iso}")

            # Subset for this location/date
            sub = srtm_df[
                (srtm_df[date_col] == g[date_col]) &
                (srtm_df['latitude'] == lat) &
                (srtm_df['longitude'] == lon)
                ].copy()

            # Get requested bands using NASA-like rule
            want = self._requested_srtm_bands(sub)

            # Geometry
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(self.config.buffer_m)

            # Compute stats once (static)
            print(f"   Computing stats for bands: {want}")
            stats = self._extract_srtm_stats(region, want)
            if not stats:
                print("   ⚠️ No SRTM stats returned.")
                continue

            row = {'date': d_iso, 'latitude': lat, 'longitude': lon}
            row.update(stats)
            results.append(row)
            print(f"   ✅ Extracted {len(stats)} statistics")

        if not results:
            print("\n❌ No SRTM data produced.")
            return pd.DataFrame()

        out = pd.DataFrame(results).sort_values(['date', 'latitude', 'longitude']).reset_index(drop=True)
        out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        out.to_csv(output_csv_path, index=False)

        elapsed = time.time() - t0
        print(f"\n✅ SRTM stats saved → {output_csv_path}")
        print(f"Rows: {len(out):,} | Cols: {len(out.columns):,} | Elapsed: {elapsed:.1f}s")

        # Print summary of variables extracted
        variable_cols = [col for col in out.columns if col not in ['date', 'latitude', 'longitude']]
        unique_vars = set([col.split('_')[0] for col in variable_cols])
        print(f"Variables extracted: {sorted(unique_vars)}")

        return out

    # ---------- Convenience methods ----------
    def run_analysis_simple(self,
                            coordinates_list: List[tuple],
                            dates_list: List[str] = None,
                            output_path: str = None) -> pd.DataFrame:
        """Convenience function for quick usage with coordinate lists."""
        if output_path is None:
            output_path = 'srtm_analysis_results.csv'

        return self.run_pipeline(
            fetch_plan_path='auto_srtm_fetch_plan.csv',  # Will be auto-created
            output_csv_path=output_path,
            coordinates_list=coordinates_list,
            dates_list=dates_list
        )

class UniCropERA5Pipeline:
    """ERA5-Land GEE data pipeline for UniCrop project."""

    def __init__(self, config: Optional[ERA5Config] = None,
                 project_id: str = 'unicrop-466421',
                 **kwargs):
        """Initialize pipeline with configuration."""
        self.config = config or ERA5Config()
        self.project_id = project_id

        # Allow override of config values via kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize GEE
        self._ensure_gee()

    def _ensure_gee(self):
        """Initialize EE only if needed (safe to call any time)."""
        try:
            ee.Image(0).getInfo()
            return
        except Exception:
            pass
        try:
            ee.Initialize(project=self.project_id)
            ee.Image(0).getInfo()
            print("✅ GEE initialized.")
            return
        except Exception:
            self._init_gee()

    def _init_gee(self):
        """Initialize GEE with authentication."""
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            print("✅ GEE initialized successfully.")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            raise

    # ---------- Plan parsing helpers ----------
    @staticmethod
    def _find_first(df: pd.DataFrame, must_contain: List[str]) -> Optional[str]:
        """Return first column whose name contains ALL tokens (case-insensitive)."""
        for c in df.columns:
            name = c.strip().lower()
            if all(k in name for k in must_contain):
                return c
        return None

    @staticmethod
    def _find_harvest_date_col(df: pd.DataFrame) -> Optional[str]:
        """Find harvest date column in various formats."""
        col = UniCropERA5Pipeline._find_first(df, ["date", "harvest"])
        if col:
            return col
        variants = ["date of harvest", "harvest date", "date_of_harvest", "harvest_date", "date"]
        low = {c.lower(): c for c in df.columns}
        for v in variants:
            if v in low:
                return low[v]
        return None

    @staticmethod
    def _split_csv_like(s: Optional[str]) -> List[str]:
        """Split 'api parameter' robustly on comma/semicolon/pipe/slash."""
        if not s or pd.isna(s):
            return []
        t = re.sub(r'[\/|;]', ',', str(s))
        t = re.sub(r'[\(\)]', ' ', t)
        toks = [x.strip() for x in t.split(',') if x and x.strip()]
        return toks

    def _norm_era5_band(self, tok: str) -> str:
        """Normalize a token to an ERA5 band name (lowercase; spaces/dashes -> underscores)."""
        t = str(tok).strip().lower()
        t = t.replace('-', '_').replace(' ', '_')
        return self.config.band_aliases.get(t, t)

    def _collect_requested_bands(self, sub: pd.DataFrame) -> Tuple[set, bool]:
        """
        From the location's ERA5 rows, gather requested bands from 'api parameter'.
        Also detect if 'Irrigation' is requested as Derived (optional).
        """
        lower_map = {c.strip().lower(): c for c in sub.columns}
        api_col = lower_map.get('api parameter')
        var_col = lower_map.get('variable')

        bands = set()
        want_irrigation = False

        for _, r in sub.iterrows():
            api = str(r.get(api_col, '') or '').strip()
            var = str(r.get(var_col, '') or '').strip()
            if api and api.lower() != 'derived':
                for tok in self._split_csv_like(api):
                    bands.add(self._norm_era5_band(tok))
            else:
                if var.lower() == 'irrigation':
                    want_irrigation = True
                    bands.update(['total_precipitation', 'potential_evaporation'])

        return bands, want_irrigation

    def _reduce_stats(self, image: ee.Image, region: ee.Geometry, band_list: List[str]) -> Dict[str, float]:
        """Compute statistics for ERA5 bands over region."""
        if not band_list:
            return {}
        img = image.select(band_list)
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), '', True),
            geometry=region,
            scale=self.config.era5_scale,
            maxPixels=self.config.max_pixels,
            bestEffort=self.config.best_effort,
            tileScale=self.config.tile_scale
        ).getInfo() or {}
        return stats

    def _apply_unit_conversions(self, row: Dict) -> Dict:
        """Apply unit conversions to the data row."""
        converted_row = row.copy()

        # Unit fixes: precip & evap (m → mm) [evap negative per convention]
        for k in list(converted_row.keys()):
            v = converted_row[k]
            if isinstance(v, (int, float)):
                if 'total_precipitation' in k:
                    converted_row[k] = round(float(v) * self.config.precipitation_factor, 3)
                elif 'potential_evaporation' in k:
                    converted_row[k] = round(float(v) * self.config.evaporation_factor, 3)

        # Temperatures (K → °C), including skin_temperature*
        for k in list(converted_row.keys()):
            v = converted_row[k]
            if isinstance(v, (int, float)) and ('temperature' in k) and not k.endswith('_stdDev'):
                converted_row[k] = round(float(v) - self.config.temperature_offset, 2)

        return converted_row

    def _calculate_irrigation(self, row: Dict) -> Optional[float]:
        """Calculate irrigation if requested and data available."""
        if not self.config.enable_irrigation_calculation:
            return None

        pev = row.get('potential_evaporation_mean')
        tp = row.get('total_precipitation_mean')

        if (pev is not None) and (tp is not None):
            try:
                return round(max(0.0, -(float(pev) + float(tp)) * 1000.0), 3)
            except Exception:
                pass
        return None

    # ---------- Auto-creation of fetch plans ----------
    def create_default_fetch_plan(self, coordinates_list: List[tuple], dates_list: List[str] = None) -> pd.DataFrame:
        """Create a default fetch plan for ERA5 when no CSV is available."""
        if dates_list is None:
            dates_list = [datetime.now().strftime('%Y-%m-%d')] * len(coordinates_list)

        if len(dates_list) != len(coordinates_list):
            # Repeat dates if needed
            dates_list = dates_list * (len(coordinates_list) // len(dates_list) + 1)
            dates_list = dates_list[:len(coordinates_list)]

        data = []
        for i, ((lat, lon), date) in enumerate(zip(coordinates_list, dates_list)):
            data.append({
                'Source Dataset': 'ERA5',
                'API Parameter': 'total_precipitation,potential_evaporation,temperature_2m,skin_temperature',
                'Variable': '',
                'Harvest Date': date,
                'Latitude': lat,
                'Longitude': lon,
                'Location_ID': f'ERA5_Point_{i + 1}'
            })

        return pd.DataFrame(data)

    # ---------- Debug functions ----------
    def debug_fetch_plan_parsing(self, fetch_plan_path: str = 'fetch_plan.csv'):
        """Debug function to see exactly how your fetch_plan.csv is being parsed."""
        print("DEBUGGING ERA5 FETCH PLAN PARSING")
        print("=" * 50)

        import os
        if not os.path.exists(fetch_plan_path):
            print(f"File not found: {fetch_plan_path}")
            return

        # Load the file
        fp = pd.read_csv(fetch_plan_path)
        print(f"Loaded file with {len(fp)} rows and {len(fp.columns)} columns")
        print(f"Columns: {list(fp.columns)}")
        print()

        # Clean column names
        fp.columns = fp.columns.str.strip()
        lower_map = {c.strip().lower(): c for c in fp.columns}
        print(f"Column mapping (lowercase -> original):")
        for k, v in lower_map.items():
            print(f"   '{k}' -> '{v}'")
        print()

        # Find dataset column
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            print("Could not find 'Source Dataset' or 'Dataset' column")
            return

        print(f"Dataset column: '{src_col}'")

        # Filter for ERA5 rows
        df = fp.copy()
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()
        print(f"Dataset values found: {df[src_col].unique()}")

        pattern = '|'.join(self.config.dataset_patterns)
        era5_df = df[df[src_col].str.contains(pattern, case=False, na=False)].copy()
        print(f"ERA5 rows: {len(era5_df)}")

        if era5_df.empty:
            print("No ERA5 rows found!")
            print("Try using one of these dataset names:")
            for pattern in self.config.dataset_patterns:
                print(f"   - '{pattern.upper()}'")
            return

        # Show first few ERA5 rows
        print(f"\nFirst {min(3, len(era5_df))} ERA5 rows:")
        for i, (_, row) in enumerate(era5_df.head(3).iterrows()):
            print(f"   Row {i + 1}:")
            for col in era5_df.columns:
                print(f"      {col}: '{row[col]}'")
            print()

        # Test variable parsing
        print("TESTING VARIABLE PARSING")
        print("-" * 30)

        # Take first ERA5 location
        first_row = era5_df.iloc[0:1].copy()
        bands, want_irrig = self._collect_requested_bands(first_row)

        print(f"Final result - Bands: {bands}, Irrigation: {want_irrig}")

        if not bands:
            print("\nNo bands detected!")
            print("Possible solutions:")
            print("   1. Check API Parameter column has valid ERA5 band names")
            print("   2. Use 'Derived' in API Parameter and specify bands in Variable column")
            print("   3. Check band name spelling against available ERA5 bands")
        else:
            print(f"\nERA5 bands successfully detected: {sorted(bands)}")

    # ---------- Main pipeline ----------
    def run_pipeline(self,
                     fetch_plan_path: str = "fetch_plan.csv",
                     output_csv_path: Optional[str] = None,
                     coordinates_list: List[tuple] = None,
                     dates_list: List[str] = None) -> pd.DataFrame:
        """Run ERA5-Land analysis pipeline with full automation."""

        output_csv_path = output_csv_path or self.config.default_output_file

        import os
        # Automatic fetch plan creation
        if not os.path.exists(fetch_plan_path) and self.config.auto_create_plan and coordinates_list:
            print(f"No fetch plan found at {fetch_plan_path}")
            print("Creating automatic fetch plan for ERA5...")
            fp = self.create_default_fetch_plan(coordinates_list, dates_list)
            print(f"Created automatic fetch plan with {len(fp)} locations")
            # Optionally save the created plan
            auto_plan_path = fetch_plan_path.replace('.csv', self.config.auto_plan_suffix)
            fp.to_csv(auto_plan_path, index=False)
        else:
            # Load existing fetch plan
            if not os.path.exists(fetch_plan_path):
                raise FileNotFoundError(
                    f"fetch_plan.csv not found at {fetch_plan_path} and auto_create_plan is disabled")
            fp = pd.read_csv(fetch_plan_path)

        # 1) Load plan
        fp.columns = fp.columns.str.strip()

        # choose source column
        lower_map = {c.strip().lower(): c for c in fp.columns}
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            raise ValueError("fetch_plan.csv needs a 'Source Dataset' or 'Dataset' column.")

        # detect harvest date & lat/lon
        date_col = self._find_harvest_date_col(fp)
        if date_col is None:
            raise ValueError("Could not find a harvest/Date column in fetch_plan.csv.")
        lat_col = self._find_first(fp, ["lat"])
        lon_col = self._find_first(fp, ["lon"])
        if lat_col is None or lon_col is None:
            raise ValueError("fetch_plan.csv must contain latitude/longitude columns.")

        # canonical frame
        df = fp.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()

        # ERA5-only
        pattern = '|'.join(self.config.dataset_patterns)
        era = df[df[src_col].str.contains(pattern, na=False)].copy()
        if era.empty:
            print("ℹ️ No ERA5 rows found in fetch_plan.")
            return pd.DataFrame()

        # groups: one row per (date, lat, lon)
        groups = (era[[date_col, "latitude", "longitude"]]
                  .drop_duplicates()
                  .sort_values([date_col, "latitude", "longitude"])
                  .reset_index(drop=True))

        # dataset
        era5_hourly = ee.ImageCollection(self.config.era5_dataset)

        results = []
        print("\n--- ERA5-Land (daily @ harvest date) ---")

        for _, g in groups.iterrows():
            d_iso = pd.to_datetime(g[date_col]).strftime("%Y-%m-%d")
            lat = float(g["latitude"])
            lon = float(g["longitude"])
            print(f"\n→ ({lat:.4f},{lon:.4f}) • {d_iso}")

            sub = era[(era[date_col] == g[date_col]) &
                      (era["latitude"] == lat) &
                      (era["longitude"] == lon)].copy()

            bands, want_irrig = self._collect_requested_bands(sub)
            if not bands:
                print("   ⚠️ No ERA5 bands requested here; skipping.")
                continue

            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(self.config.buffer_m)

            # daily window [d, d+1)
            start = ee.Date(d_iso)
            end = start.advance(1, 'day')
            col = era5_hourly.filterDate(start, end).filterBounds(region)

            if col.size().getInfo() == 0:
                print("   ⚠️ No ERA5 hourly images for that date; skipping.")
                continue

            # validate bands against collection
            avail = set(col.first().bandNames().getInfo() or [])
            bands = [b for b in bands if b in avail]
            if not bands:
                print("   ⚠️ Requested bands not present in ERA5; skipping.")
                continue

            # split into sum vs mean
            sum_bands = sorted({b for b in bands if b in self.config.sum_bands})
            mean_bands = sorted({b for b in bands if b not in self.config.sum_bands})

            row = {'date': d_iso, 'latitude': lat, 'longitude': lon}

            # SUM daily for precip/evap
            if sum_bands:
                summed = col.select(sum_bands).sum()
                s = self._reduce_stats(summed, region, sum_bands)
                for k, v in s.items():
                    row[k] = v

            # MEAN daily for other bands (e.g., skin_temperature)
            if mean_bands:
                meaned = col.select(mean_bands).mean()
                m = self._reduce_stats(meaned, region, mean_bands)
                for k, v in m.items():
                    row[k] = v

            # Optional derived: Irrigation (mm) = max(0, -(pev + tp)) * 1000
            if want_irrig:
                irrigation = self._calculate_irrigation(row)
                if irrigation is not None:
                    row['irrigation_mm'] = irrigation

            # Apply unit conversions
            row = self._apply_unit_conversions(row)

            results.append(row)

        if not results:
            print("\n❌ No ERA5 data produced.")
            return pd.DataFrame()

        out = (pd.DataFrame(results)
               .sort_values(['date', 'latitude', 'longitude'])
               .reset_index(drop=True))
        out.to_csv(output_csv_path, index=False)
        print(f"\n✅ ERA5 daily saved → {output_csv_path}")
        print(f"Rows: {len(out):,} | Cols: {len(out.columns):,}")

        return out

    # ---------- Convenience methods ----------
    def run_analysis_simple(self,
                            coordinates_list: List[tuple],
                            dates_list: List[str] = None,
                            output_path: str = None) -> pd.DataFrame:
        """Convenience function for quick usage with coordinate lists."""
        if output_path is None:
            output_path = 'era5_analysis_results.csv'

        return self.run_pipeline(
            fetch_plan_path='auto_era5_fetch_plan.csv',  # Will be auto-created
            output_csv_path=output_path,
            coordinates_list=coordinates_list,
            dates_list=dates_list
        )

class UniCropSoilGridsPipeline:
    """SoilGrids (ISRIC) data pipeline for UniCrop project."""

    def __init__(self, config: Optional[SoilGridsConfig] = None,
                 project_id: str = 'unicrop-466421',
                 **kwargs):
        """Initialize pipeline with configuration."""
        self.config = config or SoilGridsConfig()
        self.project_id = project_id

        # Allow override of config values via kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize GEE
        self._ensure_gee()

    def _ensure_gee(self):
        """Initialize EE only if needed (safe to call any time)."""
        try:
            ee.Image(0).getInfo()
            return
        except Exception:
            pass
        try:
            ee.Initialize(project=self.project_id)
            ee.Image(0).getInfo()
            print("✅ GEE initialized.")
            return
        except Exception:
            self._init_gee()

    def _init_gee(self):
        """Initialize GEE with authentication."""
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            print("✅ GEE initialized successfully.")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            raise

    # ---------- Plan parsing helpers ----------
    @staticmethod
    def _find_first(df: pd.DataFrame, must_contain: List[str]) -> Optional[str]:
        """Return first column whose name contains ALL tokens (case-insensitive)."""
        for c in df.columns:
            name = c.strip().lower()
            if all(k in name for k in must_contain):
                return c
        return None

    @staticmethod
    def _find_harvest_date_col(df: pd.DataFrame) -> Optional[str]:
        """Find harvest date column in various formats."""
        col = UniCropSoilGridsPipeline._find_first(df, ["date", "harvest"])
        if col:
            return col
        variants = ["date of harvest", "harvest date", "date_of_harvest", "harvest_date", "date"]
        low = {c.lower(): c for c in df.columns}
        for v in variants:
            if v in low:
                return low[v]
        return None

    @staticmethod
    def _split_csv_like(s: Optional[str]) -> List[str]:
        """Split 'api parameter' robustly on comma/semicolon/pipe/slash."""
        if not s or pd.isna(s):
            return []
        t = re.sub(r'[\/|;]', ',', str(s))
        t = re.sub(r'[\(\)]', ' ', t)
        toks = [x.strip() for x in t.split(',') if x and x.strip()]
        return toks

    def _norm_soilgrids_band(self, tok: str) -> str:
        """Normalize a token to a SoilGrids band name."""
        t = re.sub(r'\s+', '_', str(tok).strip().lower())
        # Apply aliases
        for alias, target in self.config.band_aliases.items():
            t = t.replace(f'{alias}_', f'{target}_')
        return t

    def _collect_requested_bands(self, sub: pd.DataFrame) -> Dict[str, set]:
        """
        From the location's SoilGrids rows, gather requested bands from 'api parameter'.
        Returns dict of ds_key -> set of bands.
        """
        lower_map = {c.strip().lower(): c for c in sub.columns}
        api_col = lower_map.get('api parameter')

        raw_tokens = []
        if api_col:
            for _, r in sub.iterrows():
                raw_tokens += self._split_csv_like(r.get(api_col))

        if not raw_tokens:
            return {}

        tokens = [self._norm_soilgrids_band(t) for t in raw_tokens]

        # Organize by dataset_key -> {bands}
        wanted: Dict[str, set] = {}
        for t in tokens:
            if '_' not in t:
                continue
            ds_key = t.split('_', 1)[0]
            if ds_key in self.config.datasets:
                wanted.setdefault(ds_key, set()).add(t)

        return wanted

    def _reduce_stats(self, image: ee.Image, region: ee.Geometry, band_list: List[str]) -> Dict[str, float]:
        """Compute statistics for SoilGrids bands over region."""
        if not band_list:
            return {}
        img = image.select(band_list)
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), '', True),
            geometry=region,
            scale=self.config.scale,
            maxPixels=self.config.max_pixels,
            bestEffort=self.config.best_effort,
            tileScale=self.config.tile_scale
        ).getInfo() or {}
        return stats

    # ---------- Auto-creation of fetch plans ----------
    def create_default_fetch_plan(self, coordinates_list: List[tuple], dates_list: List[str] = None) -> pd.DataFrame:
        """Create a default fetch plan for SoilGrids when no CSV is available."""
        if dates_list is None:
            dates_list = [datetime.now().strftime('%Y-%m-%d')] * len(coordinates_list)

        if len(dates_list) != len(coordinates_list):
            # Repeat dates if needed
            dates_list = dates_list * (len(coordinates_list) // len(dates_list) + 1)
            dates_list = dates_list[:len(coordinates_list)]

        data = []
        default_api = ','.join(self.config.default_bands)
        for i, ((lat, lon), date) in enumerate(zip(coordinates_list, dates_list)):
            data.append({
                'Source Dataset': 'SoilGrids',
                'API Parameter': default_api,
                'Variable': '',
                'Harvest Date': date,
                'Latitude': lat,
                'Longitude': lon,
                'Location_ID': f'SoilGrids_Point_{i + 1}'
            })

        return pd.DataFrame(data)

    # ---------- Debug functions ----------
    def debug_fetch_plan_parsing(self, fetch_plan_path: str = 'fetch_plan.csv'):
        """Debug function to see exactly how your fetch_plan.csv is being parsed."""
        print("DEBUGGING SOILGRIDS FETCH PLAN PARSING")
        print("=" * 50)

        import os
        if not os.path.exists(fetch_plan_path):
            print(f"File not found: {fetch_plan_path}")
            return

        # Load the file
        fp = pd.read_csv(fetch_plan_path)
        print(f"Loaded file with {len(fp)} rows and {len(fp.columns)} columns")
        print(f"Columns: {list(fp.columns)}")
        print()

        # Clean column names
        fp.columns = fp.columns.str.strip()
        lower_map = {c.strip().lower(): c for c in fp.columns}
        print(f"Column mapping (lowercase -> original):")
        for k, v in lower_map.items():
            print(f"   '{k}' -> '{v}'")
        print()

        # Find dataset column
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            print("Could not find 'Source Dataset' or 'Dataset' column")
            return

        print(f"Dataset column: '{src_col}'")

        # Filter for SoilGrids rows
        df = fp.copy()
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()
        print(f"Dataset values found: {df[src_col].unique()}")

        pattern = '|'.join(self.config.dataset_patterns)
        soilgrids_df = df[df[src_col].str.contains(pattern, case=False, na=False)].copy()
        print(f"SoilGrids rows: {len(soilgrids_df)}")

        if soilgrids_df.empty:
            print("No SoilGrids rows found!")
            print("Try using one of these dataset names:")
            for pattern in self.config.dataset_patterns:
                print(f"   - '{pattern.upper()}'")
            return

        # Show first few SoilGrids rows
        print(f"\nFirst {min(3, len(soilgrids_df))} SoilGrids rows:")
        for i, (_, row) in enumerate(soilgrids_df.head(3).iterrows()):
            print(f"   Row {i + 1}:")
            for col in soilgrids_df.columns:
                print(f"      {col}: '{row[col]}'")
            print()

        # Test variable parsing
        print("TESTING VARIABLE PARSING")
        print("-" * 30)

        # Take first SoilGrids location
        first_row = soilgrids_df.iloc[0:1].copy()
        wanted = self._collect_requested_bands(first_row)

        print(f"Final result: {wanted}")

        if not wanted:
            print("\nNo bands detected!")
            print("Possible solutions:")
            print("   1. Check API Parameter column has valid SoilGrids band names (e.g., 'sand_0-5cm_mean')")
            print("   2. Check band name spelling against available SoilGrids bands")
        else:
            print(f"\nSoilGrids bands successfully detected: {wanted}")

    # ---------- Main pipeline ----------
    def run_pipeline(self,
                     fetch_plan_path: str = "fetch_plan.csv",
                     output_csv_path: Optional[str] = None,
                     coordinates_list: List[tuple] = None,
                     dates_list: List[str] = None) -> pd.DataFrame:
        """Run SoilGrids analysis pipeline with full automation."""

        output_csv_path = output_csv_path or self.config.default_output_file

        import os
        # Automatic fetch plan creation
        if not os.path.exists(fetch_plan_path) and self.config.auto_create_plan and coordinates_list:
            print(f"No fetch plan found at {fetch_plan_path}")
            print("Creating automatic fetch plan for SoilGrids...")
            fp = self.create_default_fetch_plan(coordinates_list, dates_list)
            print(f"Created automatic fetch plan with {len(fp)} locations")
            # Optionally save the created plan
            auto_plan_path = fetch_plan_path.replace('.csv', self.config.auto_plan_suffix)
            fp.to_csv(auto_plan_path, index=False)
        else:
            # Load existing fetch plan
            if not os.path.exists(fetch_plan_path):
                raise FileNotFoundError(
                    f"fetch_plan.csv not found at {fetch_plan_path} and auto_create_plan is disabled")
            fp = pd.read_csv(fetch_plan_path)

        # 1) Load plan
        fp.columns = fp.columns.str.strip()

        # choose source column
        lower_map = {c.strip().lower(): c for c in fp.columns}
        if "source dataset" in lower_map:
            src_col = lower_map["source dataset"]
        elif "dataset" in lower_map:
            src_col = lower_map["dataset"]
        else:
            raise ValueError("fetch_plan.csv needs a 'Source Dataset' or 'Dataset' column.")

        # detect harvest date & lat/lon
        date_col = self._find_harvest_date_col(fp)
        if date_col is None:
            raise ValueError("Could not find a harvest/Date column in fetch_plan.csv.")
        lat_col = self._find_first(fp, ["lat"])
        lon_col = self._find_first(fp, ["lon"])
        if lat_col is None or lon_col is None:
            raise ValueError("fetch_plan.csv must contain latitude/longitude columns.")

        # canonical frame
        df = fp.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        df[src_col] = df[src_col].astype(str).str.strip().str.lower()

        # SoilGrids-only
        pattern = '|'.join(self.config.dataset_patterns)
        sg = df[df[src_col].str.contains(pattern, na=False)].copy()
        if sg.empty:
            print("ℹ️ No SoilGrids rows found in fetch_plan.")
            return pd.DataFrame()

        # groups: one row per (date, lat, lon)
        groups = (sg[[date_col, "latitude", "longitude"]]
                  .drop_duplicates()
                  .sort_values([date_col, "latitude", "longitude"])
                  .reset_index(drop=True))

        results = []
        print("\n--- SoilGrids (static @ harvest date) — mean/min/max ---")

        for _, g in groups.iterrows():
            d_iso = pd.to_datetime(g[date_col]).strftime("%Y-%m-%d")
            lat = float(g["latitude"])
            lon = float(g["longitude"])
            print(f"\n→ ({lat:.4f},{lon:.4f}) • {d_iso}")

            sub = sg[(sg[date_col] == g[date_col]) &
                     (sg["latitude"] == lat) &
                     (sg["longitude"] == lon)].copy()

            wanted = self._collect_requested_bands(sub)
            if not wanted:
                print("   ⚠️ No SoilGrids bands requested here; skipping.")
                continue

            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(self.config.buffer_m)

            row = {'date': d_iso, 'latitude': lat, 'longitude': lon}
            added = False

            # Query each dataset separately
            for ds_key, bands_set in wanted.items():
                ds_id = self.config.datasets.get(ds_key)
                if not ds_id:
                    print(f"   ↪ dropped (unknown dataset key): '{ds_key}' from bands {sorted(bands_set)}")
                    continue

                im = ee.Image(ds_id)

                # Keep only bands actually present in this dataset
                avail = set(im.bandNames().getInfo() or [])
                req = sorted(bands_set & avail)
                drop = sorted(bands_set - set(req))
                if drop:
                    print(f"   ↪ dropped bands (not in {ds_key}): {drop}")

                if not req:
                    continue

                stats = self._reduce_stats(im, region, req)
                for k, v in (stats or {}).items():
                    if v is not None:
                        row[k] = v
                        added = True

            if added:
                results.append(row)
            else:
                print("   ⚠️ No numeric values for this location; row skipped.")

        if not results:
            print("\n❌ No SoilGrids data produced.")
            return pd.DataFrame()

        out = (pd.DataFrame(results)
               .sort_values(['date', 'latitude', 'longitude'])
               .reset_index(drop=True))
        out.to_csv(output_csv_path, index=False)
        print(f"\n✅ SoilGrids saved → {output_csv_path}")
        print(f"Rows: {len(out):,} | Cols: {len(out.columns):,}")

        return out

    # ---------- Convenience methods ----------
    def run_analysis_simple(self,
                            coordinates_list: List[tuple],
                            dates_list: List[str] = None,
                            output_path: str = None) -> pd.DataFrame:
        """Convenience function for quick usage with coordinate lists."""
        if output_path is None:
            output_path = 'soilgrids_analysis_results.csv'

        return self.run_pipeline(
            fetch_plan_path='auto_soilgrids_fetch_plan.csv',  # Will be auto-created
            output_csv_path=output_path,
            coordinates_list=coordinates_list,
            dates_list=dates_list
        )


class UNetTabular(nn.Module):
    def __init__(self, in_channels: int, grid_size: int = 4):
        super(UNetTabular, self).__init__()
        self.grid_size = grid_size
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Reshape input to [batch, in_channels, grid_size, grid_size]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.grid_size, self.grid_size)

        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(pool2)

        # Decoder with skip connections
        up2 = self.upconv2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Output
        out = self.out_conv(dec1)
        return out.view(batch_size, -1)
