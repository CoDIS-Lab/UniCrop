# pipeline.py (updated with UniCropPipeline using UniCropConfig)

from source_codes.sources import *

class UniCropPipeline:
    JOIN_KEYS = ["latitude", "longitude", "date"]
    FP_PREFIX = "fp_"

    def __init__(self, config: Optional[UniCropConfig] = None, **kwargs):
        self.config = config or UniCropConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.nasa = UniCropNASAPipeline()
        self.s2modis = UniCropS2ModisPipeline(drop_rows_all_zero_stats=True, project_id=self.config.project_id)
        self.s1 = UniCropSentinel1Pipeline(project_id=self.config.project_id)
        self.srtm = UniCropSRTMPipeline(project_id=self.config.project_id)
        self.era5 = UniCropERA5Pipeline(project_id=self.config.project_id)
        self.soil = UniCropSoilGridsPipeline(project_id=self.config.project_id)
        self._init_gee()
        self.s2modis.init_gee()

    def _init_gee(self):
        try:
            ee.Authenticate()
            ee.Initialize(project=self.config.project_id)
            print("✅ GEE initialized successfully.")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            raise

    @staticmethod
    def _ensure_df(x):
        if x is None:
            return pd.DataFrame()
        if isinstance(x, tuple) and len(x) >= 1:
            return x[0] if isinstance(x[0], pd.DataFrame) else pd.DataFrame()
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "date" not in out.columns and "month" in out.columns:
            out = out.rename(columns={"month": "date"})
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "latitude" in out.columns:
            out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").round(6)
        if "longitude" in out.columns:
            out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").round(6)
        if all(k in out.columns for k in UniCropPipeline.JOIN_KEYS):
            out = out.dropna(subset=UniCropPipeline.JOIN_KEYS)
            out = out.sort_values(UniCropPipeline.JOIN_KEYS).drop_duplicates(subset=UniCropPipeline.JOIN_KEYS, keep="first").reset_index(drop=True)
        return out

    @staticmethod
    def _merge(left: pd.DataFrame, right: pd.DataFrame, suffix: str) -> pd.DataFrame:
        right = UniCropPipeline._standardize(right)
        if left.empty or right.empty:
            return left
        left = UniCropPipeline._standardize(left)
        overlaps = [c for c in right.columns if c in left.columns and c not in UniCropPipeline.JOIN_KEYS]
        if overlaps:
            right = right.rename(columns={c: f"{c}{suffix}" for c in overlaps})
        merged = pd.merge(
            left.drop_duplicates(subset=UniCropPipeline.JOIN_KEYS),
            right.drop_duplicates(subset=UniCropPipeline.JOIN_KEYS),
            on=UniCropPipeline.JOIN_KEYS, how="outer"
        )
        return merged.sort_values(UniCropPipeline.JOIN_KEYS).reset_index(drop=True)

    @staticmethod
    def _collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        tmp = df.copy()
        tmp["_non_null"] = tmp.notna().sum(axis=1)
        tmp = tmp.sort_values(UniCropPipeline.JOIN_KEYS + ["_non_null"], ascending=[True, True, True, False])
        tmp = tmp.drop_duplicates(subset=UniCropPipeline.JOIN_KEYS, keep="first").drop(columns="_non_null")
        return tmp.reset_index(drop=True)

    @staticmethod
    def _find_first(df: pd.DataFrame, must_contain: list[str]) -> str | None:
        for c in df.columns:
            name = c.strip().lower()
            if all(k in name for k in must_contain):
                return c
        return None

    @staticmethod
    def _find_harvest_date_col(df: pd.DataFrame) -> str | None:
        col = UniCropPipeline._find_first(df, ["date", "harvest"])
        if col:
            return col
        variants = ["date of harvest", "harvest date", "date_of_harvest", "harvest_date", "date"]
        low = {c.lower(): c for c in df.columns}
        for v in variants:
            if v in low:
                return low[v]
        return None

    @staticmethod
    def _canon(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z]", " ", s.lower())).strip()

    def _build_fetch_plan_meta(self, fetch_plan_path: str) -> pd.DataFrame:
        fp_raw = pd.read_csv(fetch_plan_path)
        fp_raw.columns = [c.strip() for c in fp_raw.columns]
        date_col = self._find_harvest_date_col(fp_raw)
        if date_col is None:
            return pd.DataFrame(columns=self.JOIN_KEYS)
        lat_col = self._find_first(fp_raw, ["lat"])
        lon_col = self._find_first(fp_raw, ["lon"])
        if lat_col is None or lon_col is None:
            return pd.DataFrame(columns=self.JOIN_KEYS)
        meta = fp_raw.copy()
        meta["date"] = pd.to_datetime(meta[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        meta["latitude"] = pd.to_numeric(meta[lat_col], errors="coerce").round(6)
        meta["longitude"] = pd.to_numeric(meta[lon_col], errors="coerce").round(6)
        meta = meta.dropna(subset=self.JOIN_KEYS)
        mapping_names = {
            "variable", "api parameter", "api", "band", "source dataset",
            "dataset", "dataset gee nasa other", "frequency",
            "detailed notes", "detailed_notes", "detailed notes calculation derivation"
        }
        canon_map = {c: self._canon(c) for c in meta.columns}
        keep_cols = []
        for c in meta.columns:
            if c in self.JOIN_KEYS:
                continue
            is_mapping = canon_map[c] in mapping_names
            if self.config.include_mapping_meta or not is_mapping:
                keep_cols.append(c)
        meta_prefixed = meta[[*self.JOIN_KEYS]].copy()
        for c in keep_cols:
            meta_prefixed[f"{self.FP_PREFIX}{c}"] = meta[c]
        aggs = {c: "first" for c in meta_prefixed.columns if c not in self.JOIN_KEYS}
        meta_one = (meta_prefixed
                    .sort_values(self.JOIN_KEYS)
                    .groupby(self.JOIN_KEYS, as_index=False)
                    .agg(aggs))
        return self._standardize(meta_one)

    def _reorder_with_fp_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        first = [c for c in self.config.priority_columns if c in df.columns]
        rest = [c for c in df.columns if c not in first]
        return df[first + rest]

    @staticmethod
    def _remove_duplicate_date_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "date" in df.columns and "fp_Date of Harvest" in df.columns:
            df = df.drop(columns=["fp_Date of Harvest"])
            print("Removed duplicate 'fp_Date of Harvest' column (same as 'date')")
        return df

    def run_all(self,
                fetch_plan_path="fetch_plan.csv",
                f_name_suffix="trial",
                master_timeseries_csv="unicrop_master_timeseries.csv",
                columns_manifest_csv="unicrop_columns_manifest.csv") -> pd.DataFrame:
        print("\n================ UNICROP AUTOMATED PIPELINE ================")
        fNameSuffix = self.config.f_name_suffix
        nasa_daily_file = os.path.join(self.config.output_dir, f"nasa_daily_{fNameSuffix}.csv")
        nasa_monthly_file = os.path.join(self.config.output_dir, f"nasa_monthly_{fNameSuffix}.csv")
        if os.path.exists(nasa_daily_file) and os.path.exists(nasa_monthly_file):
            print("ℹ️ Loading existing NASA files.")
            nasa_daily_df = pd.read_csv(nasa_daily_file)
            nasa_monthly_df = pd.read_csv(nasa_monthly_file)
        else:
            print("Running NASA pipeline...")
            nasa_daily_df, nasa_monthly_df = self.nasa.run_pipeline(
                fetch_plan_path=fetch_plan_path,
                daily_output_file=nasa_daily_file,
                stats_output_file=nasa_monthly_file,
            )
        s2modis_file = os.path.join(self.config.output_dir, f"S2_MODIS_timeseries_{fNameSuffix}.csv")
        if os.path.exists(s2modis_file):
            print("ℹ️ Loading existing S2/MODIS file.")
            s2modis_df = pd.read_csv(s2modis_file)
        else:
            print("Running S2/MODIS pipeline...")
            s2modis_df = self.s2modis.run_pipeline(
                fetch_plan_path=fetch_plan_path,
                timeseries_output_file=s2modis_file
            )
        s1_file = os.path.join(self.config.output_dir, f"Sentinel1_Data_Timeseries_{fNameSuffix}.csv")
        if os.path.exists(s1_file):
            print("ℹ️ Loading existing Sentinel-1 file.")
            s1_df = pd.read_csv(s1_file)
        else:
            print("Running Sentinel-1 pipeline...")
            s1_df = self.s1.run_pipeline(
                fetch_plan_path=fetch_plan_path,
                output_csv_path=s1_file
            )
        srtm_file = os.path.join(self.config.output_dir, f"SRTM_Data_Timeseries_{fNameSuffix}.csv")
        if os.path.exists(srtm_file):
            print("ℹ️ Loading existing SRTM file.")
            srtm_df = pd.read_csv(srtm_file)
        else:
            print("Running SRTM pipeline...")
            srtm_df = self.srtm.run_pipeline(
                fetch_plan_path=fetch_plan_path,
                output_csv_path=srtm_file
            )
        era5_file = os.path.join(self.config.output_dir, f"ERA5_Data_{fNameSuffix}.csv")
        if os.path.exists(era5_file):
            print("ℹ️ Loading existing ERA5 file.")
            era5_df = pd.read_csv(era5_file)
        else:
            print("Running ERA5 pipeline...")
            era5_df = self.era5.run_pipeline(
                fetch_plan_path=fetch_plan_path,
                output_csv_path=era5_file
            )
        soil_file = os.path.join(self.config.output_dir, f"SoilGrids_Data_{fNameSuffix}.csv")
        if os.path.exists(soil_file):
            print("ℹ️ Loading existing SoilGrids file.")
            soil_df = pd.read_csv(soil_file)
        else:
            print("Running SoilGrids pipeline...")
            soil_df = self.soil.run_pipeline(
                fetch_plan_path=fetch_plan_path,
                output_csv_path=soil_file
            )

        nasa_daily_df = self._standardize(self._ensure_df(nasa_daily_df))
        nasa_monthly_df = self._standardize(self._ensure_df(nasa_monthly_df))
        s2modis_df = self._standardize(self._ensure_df(s2modis_df))
        s1_df = self._standardize(self._ensure_df(s1_df))
        srtm_df = self._standardize(self._ensure_df(srtm_df))
        era5_df = self._standardize(self._ensure_df(era5_df))
        soil_df = self._standardize(self._ensure_df(soil_df))
        print(f"\nShapes → NASA daily {nasa_daily_df.shape} | NASA monthly {nasa_monthly_df.shape} | "
              f"S2/MODIS {s2modis_df.shape} | S1 {s1_df.shape} | SRTM {srtm_df.shape} | "
              f"ERA5 {era5_df.shape} | SoilGrids {soil_df.shape}")
        base = None
        for cand in [nasa_daily_df, s2modis_df, s1_df, srtm_df, era5_df, soil_df]:
            if not cand.empty:
                base = cand.copy()
                break
        if base is None:
            print("❌ No stage produced data. Nothing to merge.")
            return pd.DataFrame()
        for k in self.JOIN_KEYS:
            if k not in base.columns:
                base[k] = pd.NA
        base = self._standardize(base)
        print("\n🔗 Merging sources...")
        merged = base
        merged = self._merge(merged, nasa_monthly_df, suffix="_nasa_m")
        merged = self._merge(merged, s2modis_df, suffix="_s2")
        merged = self._merge(merged, s1_df, suffix="_s1")
        merged = self._merge(merged, srtm_df, suffix="_srtm")
        merged = self._merge(merged, era5_df, suffix="_era5")
        merged = self._merge(merged, soil_df, suffix="_soil")
        for b in ['elevation', 'slope', 'aspect', 'hillshade']:
            for s in ['mean', 'min', 'max']:
                base_col = f'{b}_{s}'
                srtm_col = f'{base_col}_srtm'
                if srtm_col in merged.columns and base_col not in merged.columns:
                    merged.rename(columns={srtm_col: base_col}, inplace=True)
        merged = self._collapse_duplicates(merged)
        fp_meta = self._build_fetch_plan_meta(fetch_plan_path)
        if not fp_meta.empty:
            merged = pd.merge(merged, fp_meta, on=self.JOIN_KEYS, how="left")
        merged = self._remove_duplicate_date_columns(merged)
        merged = self._reorder_with_fp_priority(merged)
        master_path = os.path.join(self.config.output_dir, master_timeseries_csv)
        merged.to_csv(master_path, index=False)
        print(f"✅ Master time-series saved → {master_path}")
        print(f"Rows: {len(merged):,} | Cols: {len(merged.columns):,}")
        src_sets = {
            "nasa_daily": set(nasa_daily_df.columns) if not nasa_daily_df.empty else set(),
            "nasa_monthly": set(nasa_monthly_df.columns) if not nasa_monthly_df.empty else set(),
            "sentinel2_modis": set(s2modis_df.columns) if not s2modis_df.empty else set(),
            "sentinel1": set(s1_df.columns) if not s1_df.empty else set(),
            "srtm": set(srtm_df.columns) if not srtm_df.empty else set(),
            "era5_land": set(era5_df.columns) if not era5_df.empty else set(),
            "soilgrids": set(soil_df.columns) if not soil_df.empty else set(),
            "fetch_plan_meta": set(fp_meta.columns) if not fp_meta.empty else set(),
        }
        def _origin(col: str) -> str:
            if col in UniCropPipeline.JOIN_KEYS: return "key"
            for name, cols in src_sets.items():
                if col in cols: return name
                for suf in ["_nasa_m", "_s2", "_s1", "_srtm", "_era5", "_soil"]:
                    if col.endswith(suf) and col[:-len(suf)] in cols:
                        return name
            return "unknown"
        manifest = pd.DataFrame({"column": merged.columns})
        manifest["source"] = manifest["column"].map(_origin)
        manifest_path = os.path.join(self.config.output_dir, columns_manifest_csv)
        manifest.to_csv(manifest_path, index=False)
        print(f"🧭 Columns manifest saved → {manifest_path}")
        return merged