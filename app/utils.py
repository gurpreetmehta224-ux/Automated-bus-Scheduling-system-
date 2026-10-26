import os
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs" / "data"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
CONFIG_PATH = BASE_DIR / "config" / "scheduling_config.json"
MODEL_PATH = BASE_DIR / "models" / "best_travel_time_model.joblib"
MODEL_META_PATH = BASE_DIR / "models" / "model_features.json"

@st.cache_data
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_scheduling_metrics():
    path = REPORTS_DIR / "scheduling_metrics_summary.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_ml_metrics():
    path = REPORTS_DIR / "model_evaluation_metrics.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_route_summary():
    path = DATA_DIR / "route_runtime_summary.csv"
    return pd.read_csv(path)

@st.cache_data
def load_bus_schedule():
    path = DATA_DIR / "bus_schedule.csv"
    return pd.read_csv(path)

@st.cache_data
def load_linked_duties():
    path = DATA_DIR / "linked_duty_schedule.csv"
    return pd.read_csv(path)

@st.cache_data
def load_unlinked_duties():
    path = DATA_DIR / "unlinked_duty_schedule.csv"
    return pd.read_csv(path)

@st.cache_data
def load_bus_utilization():
    path = DATA_DIR / "bus_utilization_statistics.csv"
    return pd.read_csv(path)

@st.cache_data
def load_idle_recovery_stats():
    path = DATA_DIR / "idle_recovery_statistics.csv"
    return pd.read_csv(path)

@st.cache_data
def load_all_stops_data():
    path = DATA_DIR / "delhi_bus_enriched_analytical.csv"
    cols = [
        'route_id', 'trip_id', 'stop_sequence', 'stop_id', 'stop_name',
        'stop_lat', 'stop_lon', 'arrival_time', 'departure_time',
        'dist_to_next_km', 'time_to_next_sec'
    ]
    return pd.read_csv(path, usecols=cols)

@st.cache_resource
def load_ml_model():
    if MODEL_PATH.exists() and MODEL_META_PATH.exists():
        model = joblib.load(MODEL_PATH)
        with open(MODEL_META_PATH, 'r') as f:
            meta = json.load(f)
        return model, meta['features']
    return None, []

@st.cache_data
def load_test_sample_segments():
    path = DATA_DIR / "ml_test_features.csv"
    df = pd.read_csv(path)
    return df

def get_figure_path(filename):
    return str(FIGURES_DIR / filename)

def build_pydeck_route_map(route_stops):
    """
    Build an interactive PyDeck map showing the route path and stops.
    """
    if route_stops.empty:
        return None
        
    stops = route_stops.copy().sort_values('stop_sequence').reset_index(drop=True)
    
    # Coordinates list for the path line
    path_coords = stops[['stop_lon', 'stop_lat']].values.tolist()
    
    # Tag stops: Start (Emerald Green), End (Crimson Red), Intermediate (Cyan Glow)
    def assign_color(idx, total):
        if idx == 0:
            return [16, 185, 129, 240]  # Emerald Green Origin
        elif idx == total - 1:
            return [239, 68, 68, 240]   # Crimson Red Destination
        else:
            return [0, 210, 255, 200]   # Cyan Intermediate
            
    stops['color'] = [assign_color(i, len(stops)) for i in range(len(stops))]
    stops['radius'] = [140 if i in (0, len(stops)-1) else 65 for i in range(len(stops))]
    
    view_state = pdk.ViewState(
        latitude=float(stops['stop_lat'].mean()),
        longitude=float(stops['stop_lon'].mean()),
        zoom=11,
        pitch=0
    )
    
    # Path layer connecting stops with cyan glow
    path_data = [{'path': path_coords, 'name': f"Route {stops['route_id'].iloc[0]}"}]
    path_layer = pdk.Layer(
        "PathLayer",
        path_data,
        get_path="path",
        get_color=[0, 210, 255, 220], # Electric Cyan
        width_min_pixels=3.5,
        width_scale=1,
        pickable=True
    )
    
    # Scatterplot layer for stops
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        stops,
        get_position=["stop_lon", "stop_lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True
    )
    
    map_style = getattr(pdk.map_styles, 'CARTO_DARK', 'carto-dark')
    deck = pdk.Deck(
        layers=[path_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "Stop #{stop_sequence}: {stop_name}\nSegment: {dist_to_next_km} km | {time_to_next_sec}s"},
        map_style=map_style
    )
    return deck
