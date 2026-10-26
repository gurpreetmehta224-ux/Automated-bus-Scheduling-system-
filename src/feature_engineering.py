import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 + 
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)
    
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg

def time_str_to_seconds(t_str):
    parts = t_str.strip().split(':')
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2])
    return h * 3600 + m * 60 + s

def engineer_features():
    print("=" * 70)
    print("STARTING ML PREPARATION & FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv_path = os.path.join(base_dir, "data", "final_merged_with_stops - final_merged_with_stops.csv")
    output_dir = os.path.join(base_dir, "outputs")
    data_out_dir = os.path.join(output_dir, "data")
    reports_out_dir = os.path.join(output_dir, "reports")
    
    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(reports_out_dir, exist_ok=True)
    
    df = pd.read_csv(raw_csv_path)
    print(f"Loaded raw dataset: {len(df):,} rows")
    
    # -------------------------------------------------------------
    # 1. Clean consecutive duplicate entries for ML segments
    # -------------------------------------------------------------
    # Flag consecutive duplicate records (same trip, same stop, same time)
    is_consec_dup = (df['trip_id'] == df['trip_id'].shift(1)) & \
                    (df['stop_id'] == df['stop_id'].shift(1)) & \
                    (df['arrival_time'] == df['arrival_time'].shift(1))
    
    print(f"Filtering out {is_consec_dup.sum():,} consecutive identical stop duplicates for clean segment transitions...")
    df_clean = df[~is_consec_dup].copy().reset_index(drop=True)
    
    # -------------------------------------------------------------
    # 2. Time parsing & Segment Target Construction
    # -------------------------------------------------------------
    df_clean['arr_sec'] = df_clean['arrival_time'].apply(time_str_to_seconds)
    df_clean['dep_sec'] = df_clean['departure_time'].apply(time_str_to_seconds)
    df_clean['stop_sequence'] = df_clean.groupby('trip_id').cumcount() + 1
    
    # Next stop lookahead
    df_clean['next_stop_id'] = df_clean.groupby('trip_id')['stop_id'].shift(-1)
    df_clean['next_stop_name'] = df_clean.groupby('trip_id')['stop_name'].shift(-1)
    df_clean['next_stop_lat'] = df_clean.groupby('trip_id')['stop_lat'].shift(-1)
    df_clean['next_stop_lon'] = df_clean.groupby('trip_id')['stop_lon'].shift(-1)
    df_clean['next_arr_sec'] = df_clean.groupby('trip_id')['arr_sec'].shift(-1)
    
    # Exclude last stop of each trip since it has no next stop (no onward segment)
    df_segments = df_clean[df_clean['next_stop_id'].notnull()].copy().reset_index(drop=True)
    df_segments['next_stop_id'] = df_segments['next_stop_id'].astype(int)
    
    # Construct TARGET VARIABLE: travel_time_sec
    df_segments['travel_time_sec'] = df_segments['next_arr_sec'] - df_segments['arr_sec']
    
    # Calculate segment distance
    df_segments['segment_dist_km'] = haversine_distance(
        df_segments['stop_lat'], df_segments['stop_lon'],
        df_segments['next_stop_lat'], df_segments['next_stop_lon']
    )
    
    # Handle any remaining 0 or negative travel times safely (filter out invalid segment artifacts)
    valid_mask = (df_segments['travel_time_sec'] > 0) & (df_segments['segment_dist_km'] > 0)
    print(f"Retaining {valid_mask.sum():,} valid transition segments ({len(df_segments) - valid_mask.sum():,} zero/negative time/dist filtered).")
    df_ml = df_segments[valid_mask].copy().reset_index(drop=True)
    
    # -------------------------------------------------------------
    # 3. Feature Engineering: Spatial, Temporal, Centrality, Complexity
    # -------------------------------------------------------------
    print("\nComputing comprehensive feature categories...")
    
    # A. Spatial Features
    DELHI_CENTER_LAT, DELHI_CENTER_LON = 28.645678, 77.155909
    df_ml['dist_from_delhi_center_km'] = haversine_distance(
        df_ml['stop_lat'], df_ml['stop_lon'],
        DELHI_CENTER_LAT, DELHI_CENTER_LON
    )
    df_ml['dest_dist_from_delhi_center_km'] = haversine_distance(
        df_ml['next_stop_lat'], df_ml['next_stop_lon'],
        DELHI_CENTER_LAT, DELHI_CENTER_LON
    )
    
    bearing_vals = calculate_bearing(
        df_ml['stop_lat'], df_ml['stop_lon'],
        df_ml['next_stop_lat'], df_ml['next_stop_lon']
    )
    df_ml['bearing_deg'] = bearing_vals
    df_ml['sin_bearing'] = np.sin(np.radians(bearing_vals))
    df_ml['cos_bearing'] = np.cos(np.radians(bearing_vals))
    
    # B. Cyclical Temporal Features
    # Normal 24h cycle (modulo 24 for circadian periodic cycle)
    hour_24 = (df_ml['arr_sec'] // 3600) % 24
    minute_val = (df_ml['arr_sec'] % 3600) // 60
    sec_of_day = df_ml['arr_sec'] % 86400
    
    df_ml['departure_hour'] = hour_24
    df_ml['departure_minute'] = minute_val
    df_ml['sin_hour'] = np.sin(2 * np.pi * hour_24 / 24.0)
    df_ml['cos_hour'] = np.cos(2 * np.pi * hour_24 / 24.0)
    df_ml['sin_time_of_day'] = np.sin(2 * np.pi * sec_of_day / 86400.0)
    df_ml['cos_time_of_day'] = np.cos(2 * np.pi * sec_of_day / 86400.0)
    
    # Binary peak indicators
    raw_hour = df_ml['arr_sec'] // 3600
    df_ml['is_morning_peak'] = ((raw_hour >= 7) & (raw_hour < 11)).astype(int)
    df_ml['is_midday_offpeak'] = ((raw_hour >= 11) & (raw_hour < 16)).astype(int)
    df_ml['is_evening_peak'] = ((raw_hour >= 16) & (raw_hour < 20)).astype(int)
    df_ml['is_night'] = ((raw_hour >= 20) & (raw_hour < 24)).astype(int)
    df_ml['is_extended_service'] = (raw_hour >= 24).astype(int)
    
    # C. Stop Centrality & Degree Features
    stop_route_counts = df_clean.groupby('stop_id')['route_id'].nunique().to_dict()
    df_ml['origin_stop_degree'] = df_ml['stop_id'].map(stop_route_counts).fillna(1)
    df_ml['dest_stop_degree'] = df_ml['next_stop_id'].map(stop_route_counts).fillna(1)
    df_ml['is_origin_major_hub'] = (df_ml['origin_stop_degree'] >= 50).astype(int)
    df_ml['is_dest_major_hub'] = (df_ml['dest_stop_degree'] >= 50).astype(int)
    df_ml['corridor_degree_sum'] = df_ml['origin_stop_degree'] + df_ml['dest_stop_degree']
    
    # D. Route Complexity & Topological Metrics
    trip_total_stops = df_clean.groupby('trip_id')['stop_id'].count().to_dict()
    df_ml['total_stops_on_route'] = df_ml['trip_id'].map(trip_total_stops)
    df_ml['sequence_ratio'] = df_ml['stop_sequence'] / df_ml['total_stops_on_route']
    df_ml['remaining_stops'] = df_ml['total_stops_on_route'] - df_ml['stop_sequence']
    
    # Route total distance and tortuosity
    route_dist_dict = df_ml.groupby('trip_id')['segment_dist_km'].sum().to_dict()
    df_ml['total_route_distance_km'] = df_ml['trip_id'].map(route_dist_dict)
    
    # Origin-to-Destination direct distance
    route_endpoints = df_clean.groupby('trip_id').agg(
        start_lat=('stop_lat', 'first'),
        start_lon=('stop_lon', 'first'),
        end_lat=('stop_lat', 'last'),
        end_lon=('stop_lon', 'last')
    )
    route_endpoints['euclidean_dist_km'] = haversine_distance(
        route_endpoints['start_lat'], route_endpoints['start_lon'],
        route_endpoints['end_lat'], route_endpoints['end_lon']
    )
    euclid_dict = route_endpoints['euclidean_dist_km'].to_dict()
    df_ml['route_euclid_dist_km'] = df_ml['trip_id'].map(euclid_dict)
    df_ml['route_tortuosity'] = np.where(
        df_ml['route_euclid_dist_km'] > 0.5,
        df_ml['total_route_distance_km'] / df_ml['route_euclid_dist_km'],
        1.0
    ).clip(1.0, 10.0) # Clip outliers
    
    # -------------------------------------------------------------
    # 4. Grouped Train / Validation / Test Split (Strict Leakage Prevention)
    # -------------------------------------------------------------
    print("\nPerforming Group-based Train/Validation/Test split by Route...")
    
    # 80% Train, 10% Validation, 10% Test grouped by route_id
    unique_routes = df_ml['route_id'].unique()
    np.random.seed(42)
    
    # 1st split: Train+Val (90%) vs Test (10%)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df_ml, groups=df_ml['route_id']))
    
    train_val_df = df_ml.iloc[train_val_idx].copy()
    test_df = df_ml.iloc[test_idx].copy()
    
    # 2nd split: Train (80% of total) vs Val (10% of total -> 1/9th of train_val)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
    train_sub_idx, val_sub_idx = next(gss_val.split(train_val_df, groups=train_val_df['route_id']))
    
    train_df = train_val_df.iloc[train_sub_idx].copy()
    val_df = train_val_df.iloc[val_sub_idx].copy()
    
    print(f"Train set: {len(train_df):,} segments ({train_df['route_id'].nunique()} routes)")
    print(f"Validation set: {len(val_df):,} segments ({val_df['route_id'].nunique()} routes)")
    print(f"Test set: {len(test_df):,} segments ({test_df['route_id'].nunique()} routes)")
    
    # -------------------------------------------------------------
    # 5. Target Encoding (Fitted STRICTLY on Train Set)
    # -------------------------------------------------------------
    print("\nFitting smoothed Target Encoding strictly on Training partition...")
    global_mean_travel_time = train_df['travel_time_sec'].mean()
    SMOOTHING_WEIGHT = 10.0
    
    route_stats = train_df.groupby('route_id')['travel_time_sec'].agg(['count', 'mean'])
    route_te_map = (
        (route_stats['count'] * route_stats['mean'] + SMOOTHING_WEIGHT * global_mean_travel_time) /
        (route_stats['count'] + SMOOTHING_WEIGHT)
    ).to_dict()
    
    train_df['route_target_enc_time'] = train_df['route_id'].map(route_te_map).fillna(global_mean_travel_time)
    val_df['route_target_enc_time'] = val_df['route_id'].map(route_te_map).fillna(global_mean_travel_time)
    test_df['route_target_enc_time'] = test_df['route_id'].map(route_te_map).fillna(global_mean_travel_time)
    
    # Feature columns for modeling
    feature_cols = [
        # Spatial
        'segment_dist_km', 'stop_lat', 'stop_lon', 'next_stop_lat', 'next_stop_lon',
        'dist_from_delhi_center_km', 'dest_dist_from_delhi_center_km',
        'bearing_deg', 'sin_bearing', 'cos_bearing',
        # Temporal & Cyclical
        'departure_hour', 'departure_minute', 'sin_hour', 'cos_hour',
        'sin_time_of_day', 'cos_time_of_day',
        'is_morning_peak', 'is_midday_offpeak', 'is_evening_peak', 'is_night', 'is_extended_service',
        # Centrality & Hubs
        'origin_stop_degree', 'dest_stop_degree', 'is_origin_major_hub', 'is_dest_major_hub', 'corridor_degree_sum',
        # Route Complexity
        'stop_sequence', 'total_stops_on_route', 'sequence_ratio', 'remaining_stops',
        'total_route_distance_km', 'route_tortuosity',
        # Target Encoding
        'route_target_enc_time'
    ]
    
    target_col = 'travel_time_sec'
    
    # Save partitioned datasets
    train_csv = os.path.join(data_out_dir, "ml_train_features.csv")
    val_csv = os.path.join(data_out_dir, "ml_val_features.csv")
    test_csv = os.path.join(data_out_dir, "ml_test_features.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Saved Train features ({len(train_df):,} rows) to: {train_csv}")
    print(f"Saved Validation features ({len(val_df):,} rows) to: {val_csv}")
    print(f"Saved Test features ({len(test_df):,} rows) to: {test_csv}")
    
    # Feature manifest report
    manifest = {
        "target_variable": {
            "name": target_col,
            "type": "Continuous (seconds)",
            "problem_type": "Supervised Regression",
            "train_mean_seconds": float(round(train_df[target_col].mean(), 2)),
            "train_median_seconds": float(round(train_df[target_col].median(), 2)),
            "train_std_seconds": float(round(train_df[target_col].std(), 2)),
            "train_min_seconds": float(round(train_df[target_col].min(), 2)),
            "train_max_seconds": float(round(train_df[target_col].max(), 2))
        },
        "dataset_split_summary": {
            "split_strategy": "GroupShuffleSplit by route_id (Zero Route-Overlap Leakage)",
            "train_rows": int(len(train_df)),
            "train_routes": int(train_df['route_id'].nunique()),
            "val_rows": int(len(val_df)),
            "val_routes": int(val_df['route_id'].nunique()),
            "test_rows": int(len(test_df)),
            "test_routes": int(test_df['route_id'].nunique()),
            "total_feature_count": len(feature_cols)
        },
        "feature_list": feature_cols
    }
    
    manifest_path = os.path.join(reports_out_dir, "ml_feature_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Saved ML Feature Manifest to: {manifest_path}")
    print("=" * 70)
    print("ML FEATURE PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    engineer_features()
