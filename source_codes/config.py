# config.py (updated with UniCropConfig at the end)

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    target_col: str = 'fp_yield (kg / ha)'#'fp_Rice Yield (kg/ha)'
    group_cols: List[str] = None
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 10000291
    min_features: int = 15  # exact number of features to keep
    cat_cols: List[str] = field(default_factory=lambda: ['fp_Area', 'fp_Season',
                                                         'fp_Intensity'])
    season_col: str = 'fp_Season'
    district_col: str = 'fp_Harvest Area'
    def __post_init__(self):
        if self.group_cols is None:
            # self.group_cols = ['fp_District', 'fp_Season(SA = Summer Autumn, WS = Winter Spring)']
            self.group_cols = ['fp_Area', 'fp_Season']

@dataclass
class UniCropConfig:
    """Configuration class for the main UniCrop pipeline."""

    # Suffix for file names
    f_name_suffix: str = "trial"

    # GEE settings
    project_id: str = 'glass-arcade-366520'

    # Output directory
    output_dir: str = './outputs'

    # Fetch plan metadata inclusion
    include_mapping_meta: bool = False

    # Priority columns for fetch plan metadata
    priority_columns: List[str] = field(default_factory=lambda: [
        "fp_Harvest Date",#"fp_Date of Harvest",
        "fp_Area",#"fp_District",
        "fp_Latitude",
        "fp_Longitude",
        "fp_Season",#"fp_Season(SA = Summer Autumn, WS = Winter Spring)",
        "fp_Intensity",#"fp_Rice Crop Intensity(D=Double, T=Triple)",
        "fp_Harvest Area",#"fp_Field size (ha)",
        "fp_yield (kg / ha)"#"fp_Rice Yield (kg/ha)",
    ])






@dataclass
class NASAConfig:
    """Configuration class for NASA POWER data pipeline parameters."""

    # Core pipeline settings
    stats_window_days: int = 15  # set 0 if you truly want stats == daily

    # API settings
    base_url: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    community: str = "AG"
    timeout_seconds: int = 60

    # Request handling
    max_retries: int = 5
    initial_delay: int = 1
    request_delay: float = 0.2  # delay between requests

    # User agent for API requests
    user_agent: str = "UniCrop/1.0 (Research Project)"

    # Magnus formula constants for derived calculations
    magnus_a: float = 17.27
    magnus_b: float = 237.7

    # Output file names (can be overridden in run_pipeline)
    default_daily_output: str = "nasa_daily_output.csv"
    default_stats_output: str = "nasa_perdate_stats.csv"

    # Coordinate validation bounds
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0



@dataclass
class S2ModisConfig:
    buffer_m: int = 500
    max_cloud_percentage: int = 20
    s2_date_tolerance_days: int = 7
    modis_date_tolerance_days: int = 16
    modis_et_date_tolerance_days: int = 8
    s2_scale: int = 10
    modis_scale: int = 500
    omit_zero_in_plain_mean: bool = True
    all_s2_derived: List[str] = None

    def __post_init__(self):
        if self.all_s2_derived is None:
            self.all_s2_derived = ["NDVI", "EVI", "LAI", "SAVI", "NDRE", "CIredge"]


@dataclass
class S1Config:
    """Configuration class for Sentinel-1 GEE pipeline parameters."""

    # Core pipeline settings
    window_days: int = 25  # acquisitions within ± this many days around harvest date
    buffer_m: int = 450  # spatial buffer (m)
    scale: int = 10  # Sentinel-1 GRD resolution (m)
    aggregation_mode: str = "window_median"  # or "nearest"

    # GEE settings
    max_pixels: float = 1e9
    best_effort: bool = True

    # S1 collection filters
    instrument_mode: str = 'IW'
    resolution_meters: int = 10
    polarisations: List[str] = field(default_factory=lambda: ['VV', 'VH'])

    # Default variables when no fetch plan is available
    default_variables: List[str] = field(default_factory=lambda: ['VV', 'VH', 'VVVH_ratio', 'RVI'])

    # Statistics to compute for each variable
    statistics: List[str] = field(default_factory=lambda: ['mean', 'stdDev', 'min', 'max'])

    # Output file names (can be overridden)
    default_output_file: str = 'Sentinel1_Data_Timeseries.csv'

    # Dataset detection patterns
    dataset_patterns: List[str] = field(default_factory=lambda: [
        r'copernicus/s1', r'sentinel-1', r'sentinel1', r's1_grd', r's1',
        r'sar', r'radar', r'c-band'
    ])

    # Variable detection patterns
    variable_patterns: List[str] = field(default_factory=lambda: [
        r's1', r'sentinel.?1', r'radar', r'sar', r'vv', r'vh'
    ])

    # Auto-creation settings
    auto_create_plan: bool = True
    auto_plan_suffix: str = '_auto.csv'


@dataclass
class SRTMConfig:
    """Configuration class for SRTM GEE pipeline parameters."""

    # Core pipeline settings
    buffer_m: int = 600  # spatial buffer radius (m)

    # GEE settings
    project_id: str = 'unicrop-466421'
    max_pixels: float = 1e9
    best_effort: bool = True
    tile_scale: int = 4

    # SRTM dataset and processing
    srtm_dataset: str = 'USGS/SRTMGL1_003'
    use_native_scale: bool = True  # Use SRTM native ~30m scale

    # Available SRTM bands and their aliases
    valid_bands: List[str] = field(default_factory=lambda: ['elevation', 'slope', 'aspect', 'hillshade'])
    band_aliases: Dict[str, str] = field(default_factory=lambda: {
        'elev': 'elevation',
        'elevation': 'elevation',
        'slope': 'slope',
        'aspect': 'aspect',
        'hillshade': 'hillshade',
        'shade': 'hillshade',
    })

    # Statistics to exclude (these are problematic for certain bands)
    exclude_stats: List[str] = field(default_factory=lambda: ['aspect_min'])  # circular statistic
    exclude_zero_stddev: bool = True  # Remove stdDev stats that are 0/None

    # Statistics suffixes to recognize and strip when parsing
    stat_suffixes: List[str] = field(default_factory=lambda: ['_mean', '_min', '_max', '_stddev', '_std'])

    # Dataset detection patterns
    dataset_patterns: List[str] = field(default_factory=lambda: ['srtm'])

    # Output file names
    default_output_file: str = 'SRTM_Data_Timeseries.csv'

    # Auto-creation settings for fetch plans
    auto_create_plan: bool = True
    auto_plan_suffix: str = '_auto.csv'


# ==============================================================================
# ERA5-Land Pipeline Class (restructured)
# ==============================================================================


@dataclass
class ERA5Config:
    """Configuration class for ERA5-Land GEE pipeline parameters."""

    # Core pipeline settings
    buffer_m: int = 600  # small buffer; ERA5-Land ~11km, buffer mainly to form a region
    era5_scale: int = 11132  # ~11.1 km

    # GEE settings
    max_pixels: float = 1e9
    best_effort: bool = True
    tile_scale: int = 1

    # ERA5 dataset
    era5_dataset: str = 'ECMWF/ERA5_LAND/HOURLY'

    # Band processing - which bands to sum vs mean
    sum_bands: List[str] = field(default_factory=lambda: ['total_precipitation', 'potential_evaporation'])

    # Band aliases for normalization
    band_aliases: Dict[str, str] = field(default_factory=lambda: {
        'skin_temperature': 'skin_temperature',
        'temperature_2m': 'temperature_2m',
        'dewpoint_temperature_2m': 'dewpoint_temperature_2m',
        'u_component_of_wind_10m': 'u_component_of_wind_10m',
        'v_component_of_wind_10m': 'v_component_of_wind_10m',
        'total_precipitation': 'total_precipitation',
        'potential_evaporation': 'potential_evaporation',
        'surface_pressure': 'surface_pressure',
    })

    # Dataset detection patterns
    dataset_patterns: List[str] = field(default_factory=lambda: ['era5'])

    # Output file names
    default_output_file: str = 'ERA5_Data.csv'

    # Unit conversion factors
    precipitation_factor: float = 1000.0  # m to mm
    evaporation_factor: float = -1000.0  # m to mm (negative per convention)
    temperature_offset: float = 273.15  # K to °C

    # Irrigation calculation (derived variable)
    enable_irrigation_calculation: bool = True

    # Auto-creation settings for fetch plans
    auto_create_plan: bool = True
    auto_plan_suffix: str = '_auto.csv'


@dataclass
class SoilGridsConfig:
    """Configuration class for SoilGrids GEE pipeline parameters."""

    # Core pipeline settings
    buffer_m: int = 600  # spatial buffer radius (m)
    scale: int = 250  # SoilGrids resolution (m)

    # GEE settings
    max_pixels: float = 1e9
    best_effort: bool = True
    tile_scale: int = 4

    # SoilGrids datasets registry (prefix -> dataset ID)
    datasets: Dict[str, str] = field(default_factory=lambda: {
        'ocd': 'projects/soilgrids-isric/ocd_mean',
        'phh2o': 'projects/soilgrids-isric/phh2o_mean',
        'sand': 'projects/soilgrids-isric/sand_mean',
        'silt': 'projects/soilgrids-isric/silt_mean',
        'clay': 'projects/soilgrids-isric/clay_mean',
        'nitrogen': 'projects/soilgrids-isric/nitrogen_mean',
        'cec': 'projects/soilgrids-isric/cec_mean',
        'cfvo': 'projects/soilgrids-isric/cfvo_mean',
        'bdod': 'projects/soilgrids-isric/bdod_mean',
    })

    # Band aliases for normalization
    band_aliases: Dict[str, str] = field(default_factory=lambda: {
        'bulk': 'bdod',
        'balk': 'bdod',
    })

    # Dataset detection patterns
    dataset_patterns: List[str] = field(default_factory=lambda: ['soilgrids', 'isric'])

    # Output file names
    default_output_file: str = 'SoilGrids_Data.csv'

    # Auto-creation settings for fetch plans
    auto_create_plan: bool = True
    auto_plan_suffix: str = '_auto.csv'

    # Default bands for auto-created plans
    default_bands: List[str] = field(default_factory=lambda: [
        'bdod_0-5cm_mean', 'clay_0-5cm_mean', 'sand_0-5cm_mean', 'silt_0-5cm_mean',
        'ocd_0-5cm_mean', 'phh2o_0-5cm_mean', 'nitrogen_0-5cm_mean', 'cec_0-5cm_mean', 'cfvo_0-5cm_mean'
    ])

