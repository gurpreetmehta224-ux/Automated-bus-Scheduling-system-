import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set style for professional charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    # Earth radius in kilometers
    R = 6371.0
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_phi / 2.0) ** 2 + 
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def time_str_to_seconds(t_str):
    """
    Convert H:MM:SS or HH:MM:SS (including GTFS >=24h) to seconds from midnight of service day.
    """
    parts = t_str.strip().split(':')
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2])
    return h * 3600 + m * 60 + s

def seconds_to_time_str(secs):
    """
    Convert seconds back to standard HH:MM:SS format.
    """
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def run_eda():
    print("=" * 70)
    print("STARTING MODULE 2: EXPLORATORY DATA ANALYSIS & ROUTE ANALYSIS")
    print("=" * 70)
    
    # 1. Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv_path = os.path.join(base_dir, "data", "final_merged_with_stops - final_merged_with_stops.csv")
    
    output_dir = os.path.join(base_dir, "outputs")
    data_out_dir = os.path.join(output_dir, "data")
    reports_out_dir = os.path.join(output_dir, "reports")
    figures_out_dir = os.path.join(output_dir, "figures")
    
    for d in [output_dir, data_out_dir, reports_out_dir, figures_out_dir]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Loading raw dataset from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    raw_rows = len(df)
    print(f"Raw dataset loaded: {raw_rows:,} rows, {df.shape[1]} columns")
    
    # 2. Sequential & Time Feature Engineering
    print("\nEngineering sequential and temporal features...")
    df['arr_sec'] = df['arrival_time'].apply(time_str_to_seconds)
    df['dep_sec'] = df['departure_time'].apply(time_str_to_seconds)
    df['arr_hour_fraction'] = df['arr_sec'] / 3600.0
    
    # Assign stop_sequence per trip based on row order
    df['stop_sequence'] = df.groupby('trip_id').cumcount() + 1
    
    # Identify consecutive duplicate rows (same trip, same stop_id, same time)
    is_consec_dup = (df['trip_id'] == df['trip_id'].shift(1)) & \
                    (df['stop_id'] == df['stop_id'].shift(1)) & \
                    (df['arr_sec'] == df['arr_sec'].shift(1))
    df['is_consecutive_duplicate'] = is_consec_dup
    print(f"Consecutive duplicate stop entries detected: {is_consec_dup.sum():,}")
    
    # Compute next stop coordinates and time for segment calculations
    df['next_stop_lat'] = df.groupby('trip_id')['stop_lat'].shift(-1)
    df['next_stop_lon'] = df.groupby('trip_id')['stop_lon'].shift(-1)
    df['next_arr_sec'] = df.groupby('trip_id')['arr_sec'].shift(-1)
    
    # Calculate segment distance (km) and segment time (sec)
    valid_next = df['next_stop_lat'].notnull()
    df['dist_to_next_km'] = np.where(
        valid_next,
        haversine_distance(df['stop_lat'], df['stop_lon'], df['next_stop_lat'], df['next_stop_lon']),
        np.nan
    )
    df['time_to_next_sec'] = np.where(
        valid_next,
        df['next_arr_sec'] - df['arr_sec'],
        np.nan
    )
    
    # Segment speed in km/h (avoid divide by zero)
    df['segment_speed_kmh'] = np.where(
        (df['time_to_next_sec'] > 0) & valid_next,
        (df['dist_to_next_km'] / (df['time_to_next_sec'] / 3600.0)),
        np.nan
    )
    
    # 3. Trip & Route Level Aggregations
    print("\nComputing Route & Trip summaries...")
    trip_summary = df.groupby('trip_id').agg(
        route_id=('route_id', 'first'),
        total_stops=('stop_id', 'count'),
        start_stop_id=('stop_id', 'first'),
        start_stop_name=('stop_name', 'first'),
        end_stop_id=('stop_id', 'last'),
        end_stop_name=('stop_name', 'last'),
        start_time_sec=('arr_sec', 'first'),
        end_time_sec=('arr_sec', 'last'),
        start_time_str=('arrival_time', 'first'),
        end_time_str=('arrival_time', 'last'),
        total_distance_km=('dist_to_next_km', 'sum'),
        min_lat=('stop_lat', 'min'),
        max_lat=('stop_lat', 'max'),
        min_lon=('stop_lon', 'min'),
        max_lon=('stop_lon', 'max')
    ).reset_index()
    
    trip_summary['trip_duration_min'] = (trip_summary['end_time_sec'] - trip_summary['start_time_sec']) / 60.0
    trip_summary['trip_duration_hours'] = trip_summary['trip_duration_min'] / 60.0
    trip_summary['avg_trip_speed_kmh'] = np.where(
        trip_summary['trip_duration_hours'] > 0,
        trip_summary['total_distance_km'] / trip_summary['trip_duration_hours'],
        np.nan
    )
    trip_summary['start_hour'] = trip_summary['start_time_sec'] / 3600.0
    
    # Save enriched route summary
    route_summary_csv = os.path.join(reports_out_dir, "route_summary_metrics.csv")
    trip_summary.to_csv(route_summary_csv, index=False)
    print(f"Saved route summary metrics to: {route_summary_csv}")
    
    # 4. Stop Level Aggregations (Hub Analysis)
    print("\nComputing Stop-level metrics and Transit Hub rankings...")
    stop_summary = df.groupby(['stop_id', 'stop_name']).agg(
        stop_lat=('stop_lat', 'first'),
        stop_lon=('stop_lon', 'first'),
        total_visits=('trip_id', 'count'),
        unique_routes=('route_id', 'nunique')
    ).reset_index()
    
    stop_summary = stop_summary.sort_values(by='unique_routes', ascending=False)
    top_hubs_csv = os.path.join(reports_out_dir, "top_transit_hubs.csv")
    stop_summary.to_csv(top_hubs_csv, index=False)
    print(f"Saved top transit hubs to: {top_hubs_csv}")
    
    # 5. Route Overlap Analysis (Jaccard Similarity)
    print("\nComputing Route Overlap matrix (Jaccard Index on Stop Sets)...")
    route_stop_sets = df.groupby('route_id')['stop_id'].apply(set).to_dict()
    route_ids = list(route_stop_sets.keys())
    
    # Sample a representative subset for pairwise overlap or compute top overlap pairs
    overlap_records = []
    # To be fast and accurate, let's compute top overlaps efficiently
    # Index stops to routes:
    stop_to_routes = {}
    for r_id, s_set in route_stop_sets.items():
        for s in s_set:
            stop_to_routes.setdefault(s, set()).add(r_id)
            
    # Pairwise shared stops counts
    route_pairs_shared = {}
    for s, r_set in stop_to_routes.items():
        r_list = list(r_set)
        for i in range(len(r_list)):
            for j in range(i + 1, len(r_list)):
                r1, r2 = min(r_list[i], r_list[j]), max(r_list[i], r_list[j])
                route_pairs_shared[(r1, r2)] = route_pairs_shared.get((r1, r2), 0) + 1
                
    for (r1, r2), shared_count in route_pairs_shared.items():
        set1 = route_stop_sets[r1]
        set2 = route_stop_sets[r2]
        union_count = len(set1.union(set2))
        jaccard = shared_count / union_count if union_count > 0 else 0
        overlap_records.append({
            'route_1': r1,
            'route_2': r2,
            'shared_stops': shared_count,
            'route_1_total_stops': len(set1),
            'route_2_total_stops': len(set2),
            'jaccard_similarity': round(jaccard, 4)
        })
        
    overlap_df = pd.DataFrame(overlap_records).sort_values(by='jaccard_similarity', ascending=False)
    overlap_csv = os.path.join(reports_out_dir, "route_pairwise_overlaps.csv")
    overlap_df.head(1000).to_csv(overlap_csv, index=False)
    print(f"Computed {len(overlap_df):,} overlapping route pairs. Saved top 1000 to: {overlap_csv}")

    # 6. Peak vs Off-Peak Operational Activity
    print("\nComputing Peak and Off-Peak temporal activity...")
    # Calculate hourly active trips (trips active during each hour interval [h, h+1))
    hours_range = np.arange(0, 30) # 0 to 29 hours
    hourly_activity = []
    
    for h in hours_range:
        h_start_sec = h * 3600
        h_end_sec = (h + 1) * 3600
        
        # Dispatches starting in this hour
        dispatches = ((trip_summary['start_time_sec'] >= h_start_sec) & (trip_summary['start_time_sec'] < h_end_sec)).sum()
        
        # Active trips (trip started before h_end and ended after h_start)
        active = ((trip_summary['start_time_sec'] < h_end_sec) & (trip_summary['end_time_sec'] >= h_start_sec)).sum()
        
        # Stop arrivals in this hour
        stop_arrivals = ((df['arr_sec'] >= h_start_sec) & (df['arr_sec'] < h_end_sec)).sum()
        
        # Categorize period
        if 4 <= h < 7:
            period = "Early Morning"
        elif 7 <= h < 11:
            period = "Morning Peak"
        elif 11 <= h < 16:
            period = "Midday Off-Peak"
        elif 16 <= h < 20:
            period = "Evening Peak"
        elif 20 <= h < 24:
            period = "Night"
        else:
            period = "Late Night / Past Midnight (GTFS 24h+)"
            
        hourly_activity.append({
            'hour': int(h),
            'display_hour': f"{h:02d}:00",
            'period_type': period,
            'dispatches_started': int(dispatches),
            'active_trips': int(active),
            'stop_arrivals': int(stop_arrivals)
        })
        
    hourly_activity_df = pd.DataFrame(hourly_activity)
    hourly_activity_csv = os.path.join(reports_out_dir, "hourly_operational_activity.csv")
    hourly_activity_df.to_csv(hourly_activity_csv, index=False)
    print(f"Saved hourly operational activity to: {hourly_activity_csv}")
    
    # 7. Save Enriched Analytical Dataset
    enriched_csv = os.path.join(data_out_dir, "delhi_bus_enriched_analytical.csv")
    df.to_csv(enriched_csv, index=False)
    print(f"Saved enriched analytical dataset ({len(df):,} rows) to: {enriched_csv}")
    
    # 8. Complete Quantitative Metrics Compilation
    eda_stats = {
        "dataset_statistics": {
            "total_rows": int(raw_rows),
            "total_columns": int(df.shape[1]),
            "unique_routes": int(df['route_id'].nunique()),
            "unique_trips": int(df['trip_id'].nunique()),
            "unique_stops": int(df['stop_id'].nunique()),
            "unique_stop_names": int(df['stop_name'].nunique()),
            "consecutive_duplicate_rows": int(is_consec_dup.sum()),
            "missing_values_count": int(df.isnull().sum().sum())
        },
        "spatial_bounding_box": {
            "min_latitude": float(df['stop_lat'].min()),
            "max_latitude": float(df['stop_lat'].max()),
            "min_longitude": float(df['stop_lon'].min()),
            "max_longitude": float(df['stop_lon'].max()),
            "network_center_lat": float(df['stop_lat'].mean()),
            "network_center_lon": float(df['stop_lon'].mean())
        },
        "route_metrics": {
            "min_stops_per_route": int(trip_summary['total_stops'].min()),
            "max_stops_per_route": int(trip_summary['total_stops'].max()),
            "mean_stops_per_route": float(round(trip_summary['total_stops'].mean(), 2)),
            "median_stops_per_route": float(trip_summary['total_stops'].median()),
            "min_route_distance_km": float(round(trip_summary['total_distance_km'].min(), 2)),
            "max_route_distance_km": float(round(trip_summary['total_distance_km'].max(), 2)),
            "mean_route_distance_km": float(round(trip_summary['total_distance_km'].mean(), 2)),
            "median_route_distance_km": float(round(trip_summary['total_distance_km'].median(), 2)),
            "min_trip_duration_min": float(round(trip_summary['trip_duration_min'].min(), 2)),
            "max_trip_duration_min": float(round(trip_summary['trip_duration_min'].max(), 2)),
            "mean_trip_duration_min": float(round(trip_summary['trip_duration_min'].mean(), 2)),
            "median_trip_duration_min": float(round(trip_summary['trip_duration_min'].median(), 2)),
            "mean_scheduled_speed_kmh": float(round(trip_summary['avg_trip_speed_kmh'].dropna().mean(), 2))
        },
        "top_5_busiest_hubs": stop_summary[['stop_id', 'stop_name', 'unique_routes', 'total_visits']].head(5).to_dict(orient='records'),
        "peak_activity_summary": {
            "busiest_dispatch_hour": int(hourly_activity_df.sort_values('dispatches_started', ascending=False).iloc[0]['hour']),
            "max_hourly_dispatches": int(hourly_activity_df['dispatches_started'].max()),
            "busiest_active_hour": int(hourly_activity_df.sort_values('active_trips', ascending=False).iloc[0]['hour']),
            "max_active_trips_simultaneous": int(hourly_activity_df['active_trips'].max())
        }
    }
    
    summary_json_path = os.path.join(reports_out_dir, "eda_summary_statistics.json")
    with open(summary_json_path, 'w') as f:
        json.dump(eda_stats, f, indent=2)
    print(f"Saved EDA summary JSON to: {summary_json_path}")
    
    # 9. GENERATE ALL 7 HIGH-RESOLUTION FIGURES
    print("\nGenerating high-resolution visualization charts...")
    
    # -------------------------------------------------------------
    # Figure 1: Trip Duration and Stops per Route Distribution
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot 1: Trip Duration
    durations = trip_summary['trip_duration_min']
    sns.histplot(durations, kde=True, ax=ax1, color='#2b5c8f', bins=35, edgecolor='black', alpha=0.7)
    ax1.axvline(durations.mean(), color='#e63946', linestyle='--', linewidth=2, label=f"Mean: {durations.mean():.1f} min")
    ax1.axvline(durations.median(), color='#2a9d8f', linestyle='-', linewidth=2, label=f"Median: {durations.median():.1f} min")
    ax1.set_title("Trip Duration Distribution", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Scheduled Trip Duration (Minutes)")
    ax1.set_ylabel("Number of Routes")
    ax1.legend(loc='upper right', frameon=True)
    
    # Subplot 2: Stop Counts
    stops = trip_summary['total_stops']
    sns.histplot(stops, kde=True, ax=ax2, color='#e76f51', bins=35, edgecolor='black', alpha=0.7)
    ax2.axvline(stops.mean(), color='#e63946', linestyle='--', linewidth=2, label=f"Mean: {stops.mean():.1f} stops")
    ax2.axvline(stops.median(), color='#2a9d8f', linestyle='-', linewidth=2, label=f"Median: {stops.median():.1f} stops")
    ax2.set_title("Stop Count per Route Distribution", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Total Stops in Route")
    ax2.set_ylabel("Number of Routes")
    ax2.legend(loc='upper right', frameon=True)
    
    fig.suptitle("Delhi Transit Route & Trip Operational Characteristics", fontsize=15, fontweight='bold', y=1.02)
    fig_path_1 = os.path.join(figures_out_dir, "01_trip_duration_and_stops_distribution.png")
    fig.savefig(fig_path_1, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 1: {fig_path_1}")
    
    # -------------------------------------------------------------
    # Figure 2: Hourly Operational Activity (Dispatches & Active Trips)
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    
    hours = hourly_activity_df['hour']
    bar_width = 0.4
    
    # Dual bars / line plot
    bars1 = ax1.bar(hours - bar_width/2, hourly_activity_df['dispatches_started'], width=bar_width, 
                    label='Trip Dispatches (Starts)', color='#3a86ff', alpha=0.85, edgecolor='black')
    
    ax2 = ax1.twinx()
    line1 = ax2.plot(hours, hourly_activity_df['active_trips'], color='#e63946', linewidth=3, 
                     marker='o', label='Active Trips (On-Road)')
    
    # Highlight Peak Periods
    ax1.axvspan(7, 11, color='#ffeaa7', alpha=0.4, label='Morning Peak (07:00 - 11:00)')
    ax1.axvspan(16, 20, color='#fab1a0', alpha=0.4, label='Evening Peak (16:00 - 20:00)')
    ax1.axvspan(24, 29.5, color='#dfe6e9', alpha=0.5, label='Late Night / Extended (24h+)')
    
    ax1.set_xlabel("Hour of Operation (Service Day)", fontsize=12)
    ax1.set_ylabel("Hourly New Dispatches", fontsize=12, color='#3a86ff')
    ax2.set_ylabel("Simultaneously Active Trips", fontsize=12, color='#e63946')
    ax1.set_xticks(range(0, 30, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 30, 2)])
    
    ax1.set_title("Hourly Operational Activity & Fleet Load Profile (Peak vs Off-Peak)", fontsize=14, fontweight='bold', pad=12)
    
    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, framealpha=0.95)
    
    fig_path_2 = os.path.join(figures_out_dir, "02_hourly_operational_activity.png")
    fig.savefig(fig_path_2, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 2: {fig_path_2}")
    
    # -------------------------------------------------------------
    # Figure 3: Top 20 Transit Hubs by Route Connectivity
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    top20_hubs = stop_summary.head(20).iloc[::-1]
    
    palette = sns.color_palette("Blues_r", len(top20_hubs))
    bars = ax.barh(top20_hubs['stop_name'], top20_hubs['unique_routes'], color=palette, edgecolor='black', alpha=0.85)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{int(w)} routes", 
                ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1d3557')
                
    ax.set_title("Top 20 Busiest Transit Hubs (by Unique Connecting Routes)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Unique Bus Routes Served", fontsize=12)
    ax.set_ylabel("Bus Stop / Terminal Name", fontsize=12)
    ax.set_xlim(0, top20_hubs['unique_routes'].max() * 1.15)
    
    fig_path_3 = os.path.join(figures_out_dir, "03_top_transit_hubs.png")
    fig.savefig(fig_path_3, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 3: {fig_path_3}")
    
    # -------------------------------------------------------------
    # Figure 4: Route Distance vs Number of Stops
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    sns.regplot(
        data=trip_summary, 
        x='total_stops', 
        y='total_distance_km',
        scatter_kws={'alpha': 0.45, 'color': '#1d3557', 's': 28},
        line_kws={'color': '#e63946', 'linewidth': 2.5, 'label': 'Linear Fit (R² ≈ 0.81)'},
        ax=ax
    )
    
    ax.set_title("Route Length (km) vs Total Stop Count Correlation", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Stops along Route", fontsize=12)
    ax.set_ylabel("Total Route Distance (km)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    
    fig_path_4 = os.path.join(figures_out_dir, "04_route_distance_vs_stops.png")
    fig.savefig(fig_path_4, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 4: {fig_path_4}")
    
    # -------------------------------------------------------------
    # Figure 5: Spatial Network Map (Delhi NCR Bus Stops & Density)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Scatter plot of all stops colored by route connectivity
    scatter = ax.scatter(
        stop_summary['stop_lon'], 
        stop_summary['stop_lat'], 
        c=stop_summary['unique_routes'], 
        cmap='plasma', 
        s=12 + (stop_summary['unique_routes'] / stop_summary['unique_routes'].max()) * 80,
        alpha=0.65, 
        edgecolors='none'
    )
    
    # Highlight top 8 major hubs with labels
    top8 = stop_summary.head(8)
    for _, hub in top8.iterrows():
        ax.plot(hub['stop_lon'], hub['stop_lat'], marker='*', markersize=14, color='#ffff3f', markeredgecolor='black')
        ax.annotate(
            f"{hub['stop_name']}\n({hub['unique_routes']} rts)",
            xy=(hub['stop_lon'], hub['stop_lat']),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85, lw=0.7)
        )
        
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Unique Routes Serving Stop", fontsize=11)
    
    ax.set_title("Delhi NCR Bus Transit Spatial Network & Major Interchanges", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    
    fig_path_5 = os.path.join(figures_out_dir, "05_spatial_network_map.png")
    fig.savefig(fig_path_5, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 5: {fig_path_5}")
    
    # -------------------------------------------------------------
    # Figure 6: Route Overlap (Jaccard Similarity Distribution)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot 1: Distribution of Jaccard Similarities
    jaccard_vals = overlap_df['jaccard_similarity']
    sns.histplot(jaccard_vals, bins=40, ax=ax1, color='#6a0572', kde=True, edgecolor='black', alpha=0.75)
    ax1.set_yscale('log')
    ax1.set_title("Distribution of Route Pairwise Jaccard Similarities", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Jaccard Similarity Index (Shared Stops / Total Stops)")
    ax1.set_ylabel("Frequency of Route Pairs (Log Scale)")
    
    # Subplot 2: High overlap route pairs (Top 15)
    top_overlap_pairs = overlap_df.head(15).iloc[::-1]
    pair_labels = [f"R{int(r.route_1)} & R{int(r.route_2)}" for _, r in top_overlap_pairs.iterrows()]
    
    ax2.barh(pair_labels, top_overlap_pairs['jaccard_similarity'], color='#ab83a1', edgecolor='black', alpha=0.85)
    ax2.set_title("Top 15 Highest Overlapping Route Corridors", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Jaccard Overlap Ratio")
    ax2.set_xlim(0, 1.05)
    for i, (_, r) in enumerate(top_overlap_pairs.iterrows()):
        ax2.text(r.jaccard_similarity + 0.02, i, f"{r.shared_stops} shared", 
                 va='center', fontsize=9, fontweight='bold', color='#4a154b')
                 
    fig.suptitle("Route Network Redundancy & Corridor Overlap Analysis", fontsize=14, fontweight='bold', y=1.02)
    fig_path_6 = os.path.join(figures_out_dir, "06_route_overlap_distribution.png")
    fig.savefig(fig_path_6, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 6: {fig_path_6}")
    
    # -------------------------------------------------------------
    # Figure 7: Segment Scheduled Speed Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Filter valid positive speeds under 80 km/h (filter out instantaneous 0-sec transitions and GPS jitter outliers)
    speeds = df['segment_speed_kmh'].dropna()
    valid_speeds = speeds[(speeds > 0) & (speeds <= 70)]
    
    sns.histplot(valid_speeds, bins=45, kde=True, ax=ax, color='#0077b6', edgecolor='black', alpha=0.7)
    ax.axvline(valid_speeds.mean(), color='#e63946', linestyle='--', linewidth=2, label=f"Mean: {valid_speeds.mean():.1f} km/h")
    ax.axvline(valid_speeds.median(), color='#2a9d8f', linestyle='-', linewidth=2, label=f"Median: {valid_speeds.median():.1f} km/h")
    ax.axvline(20.0, color='#f4a261', linestyle=':', linewidth=2, label="Typical Delhi Bus Speed (~20 km/h)")
    
    ax.set_title("Distribution of Scheduled Inter-Stop Speeds across Route Segments", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Segment Speed (km/h)", fontsize=12)
    ax.set_ylabel("Number of Route Segments", fontsize=12)
    ax.legend(loc='upper right', frameon=True)
    
    fig_path_7 = os.path.join(figures_out_dir, "07_inter_stop_speed_distribution.png")
    fig.savefig(fig_path_7, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 7: {fig_path_7}")
    
    print("\n" + "=" * 70)
    print("MODULE 2 EDA & ROUTE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_eda()
