import os
import json
import joblib
import numpy as np
import pandas as pd

def compute_route_runtimes():
    print("=" * 75)
    print("STEP 2: CALCULATING ROUTE RUNTIME USING SEGMENT & ML DATA")
    print("=" * 75)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "outputs", "data")
    config_path = os.path.join(base_dir, "config", "scheduling_config.json")
    model_path = os.path.join(base_dir, "models", "best_travel_time_model.joblib")
    meta_path = os.path.join(base_dir, "models", "model_features.json")
    
    # 1. Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    delay_buffer = config.get("ml_runtime_uncertainty", {}).get("delay_uncertainty_buffer", 1.05)
    print(f"Loaded scheduling config. Applied delay buffer: {delay_buffer}x")
    
    # 2. Load ML model and features
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        features_meta = json.load(f)
    feature_cols = features_meta['features']
    print(f"Using {len(feature_cols)} features for segment runtime estimation.")
    
    # 3. Load all partitioned segments
    print("Loading segmented transit data across partitions...")
    train_df = pd.read_csv(os.path.join(data_dir, "ml_train_features.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "ml_val_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "ml_test_features.csv"))
    
    all_segments = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Total segment records: {len(all_segments):,} across {all_segments['route_id'].nunique():,} unique routes.")
    
    # 4. Predict segment travel time with ML model
    print("Estimating segment travel times with ML model...")
    all_segments['ml_pred_travel_time_sec'] = model.predict(all_segments[feature_cols])
    # Apply configurable uncertainty/delay buffer
    all_segments['buffered_pred_travel_time_sec'] = all_segments['ml_pred_travel_time_sec'] * delay_buffer
    
    # 5. Route-level aggregation
    print("Aggregating route-level metrics...")
    
    # Sort by trip_id and stop_sequence to ensure strict sequential order
    all_segments = all_segments.sort_values(by=['trip_id', 'stop_sequence']).reset_index(drop=True)
    
    def format_time_str(secs):
        secs = max(0, int(round(secs)))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    route_groups = all_segments.groupby('route_id')
    
    route_records = []
    for route_id, group in route_groups:
        trip_id = group['trip_id'].iloc[0]
        start_stop_id = group['stop_id'].iloc[0]
        start_stop_name = group['stop_name'].iloc[0]
        start_lat = group['stop_lat'].iloc[0]
        start_lon = group['stop_lon'].iloc[0]
        
        end_stop_id = group['next_stop_id'].iloc[-1]
        end_stop_name = group['next_stop_name'].iloc[-1]
        end_lat = group['next_stop_lat'].iloc[-1]
        end_lon = group['next_stop_lon'].iloc[-1]
        
        start_time_sec = group['arr_sec'].iloc[0]
        start_time_str = group['arrival_time'].iloc[0]
        
        total_segments = len(group)
        total_stops = total_segments + 1
        total_dist_km = group['segment_dist_km'].sum()
        
        # Scheduled durations
        scheduled_runtime_sec = group['travel_time_sec'].sum()
        scheduled_runtime_min = scheduled_runtime_sec / 60.0
        scheduled_end_sec = start_time_sec + scheduled_runtime_sec
        scheduled_end_str = format_time_str(scheduled_end_sec)
        
        # ML Predicted durations
        ml_runtime_sec = group['ml_pred_travel_time_sec'].sum()
        ml_runtime_min = ml_runtime_sec / 60.0
        ml_end_sec = start_time_sec + ml_runtime_sec
        ml_end_str = format_time_str(ml_end_sec)
        
        # Buffered ML durations
        buffered_runtime_sec = group['buffered_pred_travel_time_sec'].sum()
        buffered_runtime_min = buffered_runtime_sec / 60.0
        buffered_end_sec = start_time_sec + buffered_runtime_sec
        buffered_end_str = format_time_str(buffered_end_sec)
        
        # Speeds
        sched_speed_kmh = (total_dist_km / (scheduled_runtime_sec / 3600.0)) if scheduled_runtime_sec > 0 else 0.0
        ml_speed_kmh = (total_dist_km / (ml_runtime_sec / 3600.0)) if ml_runtime_sec > 0 else 0.0
        
        route_records.append({
            'route_id': route_id,
            'trip_id': trip_id,
            'total_stops': total_stops,
            'total_segments': total_segments,
            'total_distance_km': round(total_dist_km, 3),
            'start_stop_id': start_stop_id,
            'start_stop_name': start_stop_name,
            'start_stop_lat': round(start_lat, 6),
            'start_stop_lon': round(start_lon, 6),
            'end_stop_id': end_stop_id,
            'end_stop_name': end_stop_name,
            'end_stop_lat': round(end_lat, 6),
            'end_stop_lon': round(end_lon, 6),
            'start_time_sec': start_time_sec,
            'start_time_str': start_time_str,
            'start_hour': round(start_time_sec / 3600.0, 2),
            # Scheduled runtime
            'scheduled_runtime_sec': round(scheduled_runtime_sec, 1),
            'scheduled_runtime_min': round(scheduled_runtime_min, 2),
            'scheduled_end_sec': scheduled_end_sec,
            'scheduled_end_str': scheduled_end_str,
            # ML Predicted runtime
            'ml_predicted_runtime_sec': round(ml_runtime_sec, 1),
            'ml_predicted_runtime_min': round(ml_runtime_min, 2),
            'ml_predicted_end_sec': round(ml_end_sec, 1),
            'ml_predicted_end_str': ml_end_str,
            # Buffered runtime
            'buffered_runtime_sec': round(buffered_runtime_sec, 1),
            'buffered_runtime_min': round(buffered_runtime_min, 2),
            'buffered_end_sec': round(buffered_end_sec, 1),
            'buffered_end_str': buffered_end_str,
            # Speeds
            'scheduled_speed_kmh': round(sched_speed_kmh, 2),
            'ml_predicted_speed_kmh': round(ml_speed_kmh, 2)
        })
        
    route_df = pd.DataFrame(route_records)
    
    # Save route runtime summary
    out_csv = os.path.join(data_dir, "route_runtime_summary.csv")
    route_df.to_csv(out_csv, index=False)
    print(f"\nSaved route runtime summary ({len(route_df):,} routes) to: {out_csv}")
    
    # Print quantitative summary
    print("\n--- Route Runtime Summary Statistics ---")
    print(f"Total Routes Compiled: {len(route_df):,}")
    print(f"Route Distances: Mean = {route_df['total_distance_km'].mean():.2f} km | Median = {route_df['total_distance_km'].median():.2f} km | Range = [{route_df['total_distance_km'].min():.2f}, {route_df['total_distance_km'].max():.2f}] km")
    print(f"Stops per Route: Mean = {route_df['total_stops'].mean():.1f} | Median = {route_df['total_stops'].median():.0f} | Range = [{route_df['total_stops'].min()}, {route_df['total_stops'].max()}]")
    print(f"Scheduled Runtime: Mean = {route_df['scheduled_runtime_min'].mean():.2f} min | Median = {route_df['scheduled_runtime_min'].median():.2f} min | Range = [{route_df['scheduled_runtime_min'].min():.2f}, {route_df['scheduled_runtime_min'].max():.2f}] min")
    print(f"ML Predicted Runtime: Mean = {route_df['ml_predicted_runtime_min'].mean():.2f} min | Median = {route_df['ml_predicted_runtime_min'].median():.2f} min")
    print(f"Buffered Runtime ({delay_buffer}x): Mean = {route_df['buffered_runtime_min'].mean():.2f} min | Median = {route_df['buffered_runtime_min'].median():.2f} min")
    print("=" * 75)
    return route_df

if __name__ == '__main__':
    compute_route_runtimes()
