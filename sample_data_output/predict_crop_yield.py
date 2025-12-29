
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
