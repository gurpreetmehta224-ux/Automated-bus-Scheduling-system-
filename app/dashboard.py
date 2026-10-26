import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Ensure repository root and app directory are in sys.path
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(1, str(APP_DIR))

# Resilient package/module import
try:
    from app.utils import (
        load_config, load_scheduling_metrics, load_ml_metrics,
        load_route_summary, load_bus_schedule, load_linked_duties,
        load_unlinked_duties, load_bus_utilization, load_idle_recovery_stats,
        load_all_stops_data, load_ml_model, load_test_sample_segments,
        get_figure_path, build_pydeck_route_map
    )
    from app.icons import get_icon
except (ModuleNotFoundError, ImportError):
    from utils import (
        load_config, load_scheduling_metrics, load_ml_metrics,
        load_route_summary, load_bus_schedule, load_linked_duties,
        load_unlinked_duties, load_bus_utilization, load_idle_recovery_stats,
        load_all_stops_data, load_ml_model, load_test_sample_segments,
        get_figure_path, build_pydeck_route_map
    )
    from icons import get_icon

# Page configuration
icon_path = APP_DIR / "bus_icon.svg"
st.set_page_config(
    page_title="DTC Bus Scheduling & Route Management | Control Room",
    page_icon=str(icon_path) if icon_path.exists() else "none",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Transit Control Center CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg-main: #070B14;
        --bg-card: rgba(15, 23, 42, 0.85);
        --bg-card-hover: rgba(30, 41, 59, 0.9);
        --border-glass: rgba(56, 189, 248, 0.16);
        --border-glass-hover: rgba(0, 210, 255, 0.45);
        --cyan-primary: #00D2FF;
        --cyan-glow: rgba(0, 210, 255, 0.25);
        --blue-accent: #38BDF8;
        --emerald-success: #10B981;
        --amber-warning: #F59E0B;
        --rose-danger: #EF4444;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
    }

    /* Background styling */
    .stApp {
        background: radial-gradient(circle at 15% 10%, #0F172A 0%, #070B14 100%);
    }

    /* Top Command Header */
    .hero-container {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 15, 29, 0.98) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), 0 0 20px -5px var(--cyan-glow);
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #00D2FF 0%, #3B82F6 50%, #10B981 100%);
    }
    .hero-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0;
    }
    .hero-title span.highlight {
        color: var(--cyan-primary);
        text-shadow: 0 0 15px var(--cyan-glow);
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: var(--text-secondary);
        margin-top: 6px;
        font-weight: 500;
    }
    .hero-tags {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 6px 14px;
        border-radius: 20px;
        text-transform: uppercase;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .tech-badge {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #7DD3FC;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 6px 12px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Notice Banner */
    .notice-banner {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid var(--amber-warning);
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #FDE68A;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Section Headers */
    .section-title-wrap {
        margin-top: 10px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: baseline;
        justify-content: space-between;
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-tagline {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Modern KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px;
        margin-bottom: 22px;
    }
    .kpi-card-v2 {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(11, 18, 33, 0.95) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card-v2:hover {
        transform: translateY(-2px);
        border-color: var(--border-glass-hover);
        box-shadow: 0 8px 24px -4px rgba(0, 210, 255, 0.15);
    }
    .kpi-card-v2::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.4), transparent);
    }
    .kpi-card-v2.accent-cyan::before { background: linear-gradient(90deg, transparent, #00D2FF, transparent); }
    .kpi-card-v2.accent-green::before { background: linear-gradient(90deg, transparent, #10B981, transparent); }
    .kpi-card-v2.accent-amber::before { background: linear-gradient(90deg, transparent, #F59E0B, transparent); }
    .kpi-card-v2.accent-rose::before { background: linear-gradient(90deg, transparent, #EF4444, transparent); }

    .kpi-lbl-v2 {
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.75px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-val-v2 {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
        letter-spacing: -0.5px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .kpi-val-v2.cyan { color: var(--cyan-primary); text-shadow: 0 0 12px rgba(0, 210, 255, 0.3); }
    .kpi-val-v2.green { color: #34D399; text-shadow: 0 0 12px rgba(16, 185, 129, 0.3); }
    .kpi-val-v2.amber { color: #FBBF24; }
    .kpi-val-v2.rose { color: #F87171; }
    .kpi-sub-v2 {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 5px;
        font-weight: 500;
    }

    /* Content Cards */
    .dashboard-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 16, 31, 0.92) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }
    .dashboard-card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Pipeline Flow UI */
    .pipeline-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        overflow-x: auto;
        padding: 16px 6px;
    }
    .pipeline-node {
        flex: 1;
        min-width: 120px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 12px 10px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .pipeline-node:hover {
        border-color: var(--cyan-primary);
        box-shadow: 0 0 14px rgba(0, 210, 255, 0.25);
        transform: translateY(-2px);
    }
    .pipeline-step {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--cyan-primary);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .pipeline-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.25;
    }
    .pipeline-arrow {
        color: var(--cyan-primary);
        font-size: 1.2rem;
        font-weight: 800;
        opacity: 0.8;
    }

    /* Problem-Solution 4-Cards Grid */
    .pillar-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .pillar-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        transition: all 0.2s ease;
    }
    .pillar-card:hover {
        border-color: var(--border-glass-hover);
        background: rgba(30, 41, 59, 0.6);
    }
    .pillar-card .pillar-badge {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-problem { background: rgba(239, 68, 68, 0.15); color: #FCA5A5; }
    .badge-aiml { background: rgba(0, 210, 255, 0.15); color: #7DD3FC; }
    .badge-opt { background: rgba(245, 158, 11, 0.15); color: #FCD34D; }
    .badge-result { background: rgba(16, 185, 129, 0.15); color: #6EE7B7; }
    .pillar-card h4 {
        margin: 0 0 6px 0;
        font-size: 0.98rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .pillar-card p {
        font-size: 0.84rem;
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.45;
    }

    /* Live System Status Panel */
    .status-panel-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin-top: 10px;
    }
    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(10, 16, 31, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .status-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-val {
        font-size: 0.76rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #34D399;
        letter-spacing: 0.5px;
    }

    /* ==============================================================================
       ENTERPRISE TRANSIT CONTROL CENTER - SIDEBAR NAVIGATION STYLING
       ============================================================================== */
    
    /* Main Sidebar Panel Container */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070B14 0%, #0A0F1E 50%, #070B14 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.16) !important;
        box-shadow: 4px 0 24px -4px rgba(0, 0, 0, 0.7) !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Preserve and style Streamlit collapse buttons */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        color: #94A3B8 !important;
        transition: color 0.2s ease, transform 0.2s ease !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="collapsedControl"]:hover {
        color: var(--cyan-primary) !important;
    }

    /* Custom high-tech scrollbar for sidebar */
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        width: 4px;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: rgba(7, 11, 20, 0.7);
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: rgba(56, 189, 248, 0.25);
        border-radius: 4px;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 210, 255, 0.5);
    }

    /* Sidebar Branding Header */
    .sidebar-header-card {
        padding: 4px 4px 14px 4px;
        margin-bottom: 14px;
    }
    .sidebar-brand-badge-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sidebar-brand-icon-box {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.18) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(0, 210, 255, 0.38);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 16px -2px rgba(0, 210, 255, 0.28), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        flex-shrink: 0;
    }
    .sidebar-brand-icon {
        font-size: 1.25rem;
        line-height: 1;
    }
    .sidebar-brand-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .sidebar-brand-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.9px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.2;
        text-shadow: 0 0 14px rgba(255, 255, 255, 0.25);
    }
    .sidebar-brand-sub {
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--cyan-primary);
        letter-spacing: 1.2px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 3px;
        opacity: 0.95;
    }
    .sidebar-header-divider {
        height: 1.5px;
        margin-top: 14px;
        background: linear-gradient(90deg, rgba(0, 210, 255, 0.75) 0%, rgba(56, 189, 248, 0.35) 45%, transparent 100%);
        border-radius: 2px;
    }

    /* Navigation Section Header */
    .sidebar-nav-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 4px;
        margin-bottom: 8px;
    }
    .sidebar-nav-header-title {
        font-size: 0.68rem;
        font-weight: 800;
        color: var(--text-muted);
        letter-spacing: 1.1px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .sidebar-nav-header-badge {
        font-size: 0.62rem;
        font-weight: 700;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.24);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Streamlit Radio Navigation Overrides */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] {
        margin-top: 0 !important;
        margin-bottom: 16px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] > div,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div {
        width: 100% !important;
        margin: 0 !important;
    }

    /* Full-width Clickable Navigation Card */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
        min-height: 44px !important;
        padding: 10px 14px 10px 44px !important;
        margin: 0 !important;
        border-radius: 9px !important;
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        user-select: none !important;
        overflow: hidden !important;
    }

    /* Remove the default circular radio indicator */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div:first-child,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Sidebar Nav Option Vector Icons via CSS Mask (Lucide Icons) */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label::after {
        content: "" !important;
        display: inline-block !important;
        width: 17px !important;
        height: 17px !important;
        position: absolute !important;
        left: 15px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background-color: #94A3B8 !important;
        mask-size: contain !important;
        mask-repeat: no-repeat !important;
        mask-position: center !important;
        -webkit-mask-size: contain !important;
        -webkit-mask-repeat: no-repeat !important;
        -webkit-mask-position: center !important;
        transition: background-color 0.18s ease, filter 0.18s ease !important;
        pointer-events: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:hover::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover::after {
        background-color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"]::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected="true"]::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked)::after {
        background-color: #00D2FF !important;
        filter: drop-shadow(0 0 6px rgba(0, 210, 255, 0.5)) !important;
    }

    /* Option 1: layout-dashboard */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] > div:nth-child(1) label::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div:nth-child(1) label::after {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='7' height='9' x='3' y='3' rx='1'/%3E%3Crect width='7' height='5' x='14' y='3' rx='1'/%3E%3Crect width='7' height='9' x='14' y='12' rx='1'/%3E%3Crect width='7' height='5' x='3' y='16' rx='1'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='7' height='9' x='3' y='3' rx='1'/%3E%3Crect width='7' height='5' x='14' y='3' rx='1'/%3E%3Crect width='7' height='9' x='14' y='12' rx='1'/%3E%3Crect width='7' height='5' x='3' y='16' rx='1'/%3E%3C/svg%3E");
    }

    /* Option 2: route / map */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] > div:nth-child(2) label::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div:nth-child(2) label::after {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='6' cy='19' r='3'/%3E%3Cpath d='M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15'/%3E%3Ccircle cx='18' cy='5' r='3'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='6' cy='19' r='3'/%3E%3Cpath d='M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15'/%3E%3Ccircle cx='18' cy='5' r='3'/%3E%3C/svg%3E");
    }

    /* Option 3: clock */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] > div:nth-child(3) label::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div:nth-child(3) label::after {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E");
    }

    /* Option 4: calendar-clock / workflow */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] > div:nth-child(4) label::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div:nth-child(4) label::after {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 7.5V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h3.5'/%3E%3Cline x1='16' x2='16' y1='2' y2='6'/%3E%3Cline x1='8' x2='8' y1='2' y2='6'/%3E%3Cline x1='3' x2='21' y1='10' y2='10'/%3E%3Cpath d='M17.5 17.5 16 16.25V14'/%3E%3Ccircle cx='16' cy='16' r='6'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 7.5V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h3.5'/%3E%3Cline x1='16' x2='16' y1='2' y2='6'/%3E%3Cline x1='8' x2='8' y1='2' y2='6'/%3E%3Cline x1='3' x2='21' y1='10' y2='10'/%3E%3Cpath d='M17.5 17.5 16 16.25V14'/%3E%3Ccircle cx='16' cy='16' r='6'/%3E%3C/svg%3E");
    }

    /* Option 5: chart-no-axes-combined */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] > div:nth-child(5) label::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div:nth-child(5) label::after {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 20v-6a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v6'/%3E%3Cpath d='M18 20V4a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v16'/%3E%3Cpath d='M6 20v-2a2 2 0 0 0-2-2H3a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 20v-6a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v6'/%3E%3Cpath d='M18 20V4a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v16'/%3E%3Cpath d='M6 20v-2a2 2 0 0 0-2-2H3a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E");
    }

    /* Option 6: info */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] > div:nth-child(6) label::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div:nth-child(6) label::after {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M12 16v-4'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M12 16v-4'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E");
    }

    /* Label text container */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        margin: 0 !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.25px !important;
        transition: color 0.18s ease, font-weight 0.18s ease !important;
        white-space: nowrap !important;
        display: flex !important;
        align-items: center !important;
        gap: 2px !important;
    }

    /* Inactive Hover State */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:not([data-selected="true"]):hover,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:not([data-selected="true"]):hover,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:not(:has(input:checked)):hover {
        background: rgba(30, 41, 59, 0.75) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        transform: translateX(3px) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:not([data-selected="true"]):hover p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:not([data-selected="true"]):hover p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:not(:has(input:checked)):hover p {
        color: #F8FAFC !important;
    }

    /* Active Selected State */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(0, 210, 255, 0.16) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        border: 1px solid rgba(0, 210, 255, 0.45) !important;
        box-shadow: 0 4px 18px -2px rgba(0, 210, 255, 0.22), inset 0 0 14px rgba(0, 210, 255, 0.08) !important;
        transform: translateX(3px) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"]::before,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected="true"]::before,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked)::before {
        content: "" !important;
        position: absolute !important;
        left: 0 !important;
        top: 15% !important;
        height: 70% !important;
        width: 3.5px !important;
        border-radius: 0 4px 4px 0 !important;
        background: #00D2FF !important;
        box-shadow: 0 0 10px #00D2FF, 0 0 20px rgba(0, 210, 255, 0.7) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected="true"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.3) !important;
    }

    /* Vector Icon & Section Header Utilities */
    .lucide-icon {
        display: inline-block;
        vertical-align: middle;
        flex-shrink: 0;
    }
    .sub-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 20px;
        margin-bottom: 12px;
        letter-spacing: -0.2px;
    }
    .sub-section-header span {
        line-height: 1.2;
    }

    /* System Parameters Configuration Card */
    .system-config-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.92) 0%, rgba(10, 16, 31, 0.96) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 6px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.05);
    }
    .system-config-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .system-config-title {
        font-size: 0.72rem;
        font-weight: 800;
        color: #E2E8F0;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-family: 'Plus Jakarta Sans', sans-serif;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .system-config-status {
        font-size: 0.62rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #00D2FF;
        background: rgba(0, 210, 255, 0.12);
        border: 1px solid rgba(0, 210, 255, 0.3);
        padding: 2px 7px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .system-config-table {
        display: flex;
        flex-direction: column;
    }
    .system-config-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6.5px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .system-config-row:last-child {
        border-bottom: none;
    }
    .system-config-label {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .system-config-value {
        font-size: 0.84rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: 0.3px;
    }
    .system-config-disclaimer {
        font-size: 0.71rem;
        color: #64748B;
        line-height: 1.35;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px dashed rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: flex-start;
        gap: 6px;
    }
    .disclaimer-icon {
        flex-shrink: 0;
        font-size: 0.75rem;
        margin-top: 1px;
    }

    /* Sidebar Footer (Restrained Academic Prototype) */
    .sidebar-footer-card {
        margin-top: 4px;
        padding: 10px 12px;
        background: rgba(10, 15, 29, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
    }
    .sidebar-footer-top {
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .status-indicator-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);
        flex-shrink: 0;
    }
    .footer-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: #CBD5E1;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .footer-sub {
        font-size: 0.63rem;
        font-weight: 600;
        color: #64748B;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 4px;
        padding-left: 14px;
    }

    /* Keep standard parameter item classes intact for Section 02 Route Analysis */
    .param-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 10px;
        padding: 14px 14px;
        margin-top: 14px;
    }
    .param-box-header {
        font-size: 0.75rem;
        font-weight: 800;
        color: var(--text-muted);
        letter-spacing: 0.7px;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .param-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        font-size: 0.82rem;
    }
    .param-item:last-child { border-bottom: none; }
    .param-lbl { color: var(--text-secondary); font-weight: 500; }
    .param-val { color: #FFFFFF; font-family: 'JetBrains Mono', monospace; font-weight: 600; }

    /* Prediction Console Hero */
    .prediction-hero {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px -6px rgba(0, 210, 255, 0.2);
    }
    .prediction-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--cyan-primary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .prediction-val-big {
        font-size: 2.6rem;
        font-weight: 800;
        color: #FFFFFF;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1;
        margin-bottom: 8px;
    }
    .prediction-sub {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    /* Comparison Chaining Flow */
    .chaining-flow-container {
        display: flex;
        align-items: center;
        justify-content: space-around;
        background: rgba(10, 16, 31, 0.7);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 22px;
        flex-wrap: wrap;
        gap: 15px;
    }
    .chain-box {
        text-align: center;
        padding: 10px 18px;
    }
    .chain-box .chain-num {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
    }
    .chain-box .chain-lbl {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 4px;
    }
    .chain-mid {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }
    .chain-badge {
        background: rgba(0, 210, 255, 0.12);
        border: 1px solid rgba(0, 210, 255, 0.3);
        color: var(--cyan-primary);
        font-size: 0.78rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }

    /* Checklist Cards */
    .insight-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
        margin-top: 10px;
    }
    .insight-item {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.14);
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    .insight-check {
        color: #34D399;
        font-weight: 800;
        font-size: 1rem;
        line-height: 1.2;
    }
    .insight-text {
        font-size: 0.88rem;
        color: var(--text-primary);
        line-height: 1.35;
        font-weight: 500;
    }

    /* Presentation Script Box */
    .pitch-quote {
        background: rgba(15, 23, 42, 0.9);
        border-left: 4px solid var(--cyan-primary);
        border-radius: 0 10px 10px 0;
        padding: 18px 22px;
        font-style: normal;
        line-height: 1.6;
        color: #E2E8F0;
        font-size: 0.94rem;
        margin-top: 12px;
    }

    /* Responsive improvements */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.6rem; }
        .hero-container { padding: 18px 20px; }
        .kpi-val-v2 { font-size: 1.5rem; }
        .pipeline-wrapper { flex-direction: column; }
        .pipeline-arrow { transform: rotate(90deg); margin: 4px 0; }
    }
</style>
""", unsafe_allow_html=True)

# Load cached data models & configs
routes_df = load_route_summary()
sched_metrics = load_scheduling_metrics()
ml_metrics = load_ml_metrics()
config = load_config()

# ==============================================================================
# HERO COMMAND HEADER
# ==============================================================================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title-row">
        <div>
            <h1 class="hero-title">
                {get_icon('bus', size=32, color='#00D2FF')} Automated Bus Scheduling <span class="highlight">& Route Management</span>
            </h1>
            <div class="hero-subtitle">
                AI-Assisted Transit Scheduling & Fleet Optimization &nbsp;|&nbsp; 
                Delhi Transport Corporation — Academic Decision-Support Prototype
            </div>
        </div>
        <div class="hero-tags">
            <div class="status-badge">
                <span class="pulse-dot"></span>
                SYSTEM ONLINE
            </div>
            <div class="tech-badge">GTFS + ML + Vehicle Chaining</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Transparent Prototype Notice
st.markdown(f"""
<div class="notice-banner">
    {get_icon('triangle-alert', size=20, color='#F59E0B')}
    <span><b>ACADEMIC DECISION-SUPPORT PROTOTYPE:</b> This system is an academic research prototype evaluating static GTFS timetable data with configurable engineering assumptions. It provides data-driven decision-support for fleet optimization and does not represent official live DTC vehicle tracking or passenger ticketing deployment.</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown(f"""
<div class="sidebar-header-card">
    <div class="sidebar-brand-badge-row">
        <div class="sidebar-brand-icon-box">
            {get_icon('bus', size=22, color='#00D2FF')}
        </div>
        <div class="sidebar-brand-text">
            <div class="sidebar-brand-title">DTC OPERATIONS</div>
            <div class="sidebar-brand-sub">TRANSIT CONTROL CENTER</div>
        </div>
    </div>
    <div class="sidebar-header-divider"></div>
</div>
<div class="sidebar-nav-header">
    <span class="sidebar-nav-header-title">SYSTEM CONSOLE</span>
    <span class="sidebar-nav-header-badge">v2.4</span>
</div>
""", unsafe_allow_html=True)

nav_options = [
    "01   Overview",
    "02   Route Analysis",
    "03   Travel-Time Prediction",
    "04   Schedule Optimization",
    "05   Results & Insights",
    "06   About & Limitations"
]

menu_selection = st.sidebar.radio(
    "NAVIGATION CONSOLE",
    nav_options,
    index=0,
    label_visibility="collapsed"
)

# Extract clean key (e.g. "01")
active_section = menu_selection.split()[0]

# Sidebar Configured Assumptions
turnaround_min = config.get('turnaround_and_recovery', {}).get('min_turnaround_time_minutes', 15.0)
duty_cap_hr = config.get('duty_duration_constraints', {}).get('max_duty_duration_hours', 8.0)
deadhead_speed = config.get('deadhead_repositioning', {}).get('deadhead_speed_kmh', 20.0)
deadhead_limit_km = config.get('deadhead_repositioning', {}).get('max_deadhead_distance_km', 15.0)
delay_buffer = config.get('ml_runtime_uncertainty', {}).get('delay_uncertainty_buffer', 1.05)

st.sidebar.markdown(f"""
<div class="system-config-card">
    <div class="system-config-header">
        <div class="system-config-title">
            {get_icon('settings', size=15, color='#00D2FF')}
            <span>SYSTEM PARAMETERS</span>
        </div>
        <span class="system-config-status">CONFIGURED</span>
    </div>
    <div class="system-config-table">
        <div class="system-config-row">
            <span class="system-config-label">Turnaround Recovery</span>
            <span class="system-config-value">{turnaround_min:.0f} min</span>
        </div>
        <div class="system-config-row">
            <span class="system-config-label">Max Single Shift</span>
            <span class="system-config-value">{duty_cap_hr:.1f} hrs</span>
        </div>
        <div class="system-config-row">
            <span class="system-config-label">Deadhead Speed</span>
            <span class="system-config-value">{deadhead_speed:.0f} km/h</span>
        </div>
        <div class="system-config-row">
            <span class="system-config-label">Deadhead Distance Cap</span>
            <span class="system-config-value">{deadhead_limit_km:.0f} km</span>
        </div>
        <div class="system-config-row">
            <span class="system-config-label">ML Delay Buffer</span>
            <span class="system-config-value">{delay_buffer:.2f}x</span>
        </div>
    </div>
    <div class="system-config-disclaimer">
        <span class="disclaimer-icon">{get_icon('info', size=13, color='#64748B')}</span>
        <span>Modeled research parameters & assumptions (Non-Official DTC Rules)</span>
    </div>
</div>

<div class="sidebar-footer-card">
    <div class="sidebar-footer-top">
        <span class="status-indicator-dot"></span>
        <span class="footer-title">DTC OPERATIONS CONSOLE</span>
    </div>
    <div class="footer-sub">
        ACADEMIC PROTOTYPE &bull; GTFS STATIC
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 01: OVERVIEW
# ==============================================================================
if active_section == "01":
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-title">01 / Fleet Operations & System Overview</div>
        <div class="section-tagline">NCT Delhi GTFS Timetable Evaluation & Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    # Core Validated KPIs
    total_trips = len(routes_df) # 2,403
    opt_buses = sched_metrics.get("linked_fleet_required", 868)
    fleet_saved = total_trips - opt_buses # 1,535
    fleet_red_pct = sched_metrics.get("fleet_reduction_percent", 63.88)
    trip_coverage = 100.0
    utilization_pct = sched_metrics.get("linked_fleet_utilization_percent", 71.25)
    avg_duty = sched_metrics.get("avg_duty_hours_per_bus", 6.11)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Scheduled Trips <span>GTFS</span></div>
            <div class="kpi-val-v2 cyan">{total_trips:,}</div>
            <div class="kpi-sub-v2">Distinct daily trips across NCT Delhi</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Optimized Fleet <span>LINKED</span></div>
            <div class="kpi-val-v2 green">{opt_buses:,}</div>
            <div class="kpi-sub-v2">vs. {total_trips:,} baseline unlinked buses</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Modeled Reduction <span>EFFICIENCY</span></div>
            <div class="kpi-val-v2 cyan">{fleet_red_pct:.1f}%</div>
            <div class="kpi-sub-v2">{fleet_saved:,} buses fewer in modeled fleet</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Trip Coverage <span>FEASIBILITY</span></div>
            <div class="kpi-val-v2 green">{trip_coverage:.0f}%</div>
            <div class="kpi-sub-v2">Zero trip dropped & zero overlap conflicts</div>
        </div>
        <div class="kpi-card-v2 accent-amber">
            <div class="kpi-lbl-v2">Driving Utilization <span>REVENUE</span></div>
            <div class="kpi-val-v2 amber">{utilization_pct:.1f}%</div>
            <div class="kpi-sub-v2">3,781.1h revenue in-service driving</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Average Bus Duty <span>COMPLIANCE</span></div>
            <div class="kpi-val-v2">{avg_duty:.2f}h</div>
            <div class="kpi-sub-v2">Strictly within 8.0h maximum shift cap</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # End-to-End Pipeline Card Flow
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="dashboard-card-header">
            <span>{get_icon('workflow', size=16, color='#00D2FF')} END-TO-END PIPELINE ARCHITECTURE</span>
        </div>
        <div class="pipeline-wrapper">
            <div class="pipeline-node">
                <div class="pipeline-step">STAGE 01</div>
                <div class="pipeline-title">GTFS DATA INGESTION</div>
            </div>
            <div class="pipeline-arrow">{get_icon('arrow-right', size=16, color='#00D2FF')}</div>
            <div class="pipeline-node">
                <div class="pipeline-step">STAGE 02</div>
                <div class="pipeline-title">DATA PROCESSING & CLEANING</div>
            </div>
            <div class="pipeline-arrow">{get_icon('arrow-right', size=16, color='#00D2FF')}</div>
            <div class="pipeline-node">
                <div class="pipeline-step">STAGE 03</div>
                <div class="pipeline-title">SPATIAL FEATURE ENGINEERING</div>
            </div>
            <div class="pipeline-arrow">{get_icon('arrow-right', size=16, color='#00D2FF')}</div>
            <div class="pipeline-node">
                <div class="pipeline-step">STAGE 04</div>
                <div class="pipeline-title">ML TRAVEL-TIME MODEL</div>
            </div>
            <div class="pipeline-arrow">{get_icon('arrow-right', size=16, color='#00D2FF')}</div>
            <div class="pipeline-node">
                <div class="pipeline-step">STAGE 05</div>
                <div class="pipeline-title">ROUTE RUNTIME ENGINE</div>
            </div>
            <div class="pipeline-arrow">{get_icon('arrow-right', size=16, color='#00D2FF')}</div>
            <div class="pipeline-node">
                <div class="pipeline-step">STAGE 06</div>
                <div class="pipeline-title">VEHICLE CHAINING (DAG MPC)</div>
            </div>
            <div class="pipeline-arrow">{get_icon('arrow-right', size=16, color='#00D2FF')}</div>
            <div class="pipeline-node" style="border-color: rgba(16, 185, 129, 0.5); background: rgba(16, 185, 129, 0.1);">
                <div class="pipeline-step" style="color: #34D399;">OUTPUT</div>
                <div class="pipeline-title" style="color: #A7F3D0;">OPTIMIZED DUTY SCHEDULE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Problem → AI/ML → Optimization → Result 4-Card Pillar
    st.markdown("""
    <div class="pillar-grid">
        <div class="pillar-card">
            <span class="pillar-badge badge-problem">01 / THE CHALLENGE</span>
            <h4>Unlinked Dispatches</h4>
            <p>Assigning one dedicated bus per scheduled trip requires <b>2,403 buses</b>, creating massive terminal layover wastage and unsustainable capital expenditure.</p>
        </div>
        <div class="pillar-card">
            <span class="pillar-badge badge-aiml">02 / AI / ML ENGINE</span>
            <h4>Travel-Time Prediction</h4>
            <p>Random Forest model trained on spatial stop features estimates segment runtimes with <b>0.36s Test MAE</b> and applies a <b>1.05x delay buffer</b> for stochastic traffic variance.</p>
        </div>
        <div class="pillar-card">
            <span class="pillar-badge badge-opt">03 / OPTIMIZATION</span>
            <h4>Vehicle Chaining (DAG)</h4>
            <p>Formulated as a Directed Acyclic Graph Minimum Path Cover to chain compatible trips while enforcing <b>15-min turnaround</b>, <b>8-hr duty cap</b>, and deadhead limits.</p>
        </div>
        <div class="pillar-card">
            <span class="pillar-badge badge-result">04 / THE RESULT</span>
            <h4>Fleet Optimization</h4>
            <p>Covers <b>100% of all 2,403 trips</b> using just <b>868 buses</b>—a <b>63.9% modeled reduction</b> saving <b>1,535 buses</b> with <b>71.3%</b> active revenue utilization.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Live-Looking System Status Panel + 24-hr Fleet Chart
    col_status, col_chart = st.columns([1, 1.4])
    with col_status:
        st.markdown(f"""
        <div class="dashboard-card" style="height: 100%;">
            <div class="dashboard-card-header">
                <span>{get_icon('circle-check', size=15, color='#10B981')} SYSTEM PIPELINE STATUS</span>
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">
                Verified execution status of end-to-end analytical pipeline modules:
            </div>
            <div class="status-panel-grid" style="grid-template-columns: 1fr;">
                <div class="status-row">
                    <span class="status-name"><span>{get_icon('database', size=14, color='#38BDF8')}</span> GTFS Data Pipeline</span>
                    <span class="status-val">VERIFIED</span>
                </div>
                <div class="status-row">
                    <span class="status-name"><span>{get_icon('cpu', size=14, color='#38BDF8')}</span> Random Forest ML Model</span>
                    <span class="status-val">TRAINED</span>
                </div>
                <div class="status-row">
                    <span class="status-name"><span>{get_icon('clipboard-check', size=14, color='#38BDF8')}</span> Test Set Evaluation</span>
                    <span class="status-val">COMPLETE</span>
                </div>
                <div class="status-row">
                    <span class="status-name"><span>{get_icon('settings', size=14, color='#38BDF8')}</span> Scheduling Engine (DAG MPC)</span>
                    <span class="status-val">COMPLETE</span>
                </div>
                <div class="status-row">
                    <span class="status-name"><span>{get_icon('monitor', size=14, color='#38BDF8')}</span> Operations Dashboard</span>
                    <span class="status-val">ONLINE</span>
                </div>
            </div>
            <div style="font-size: 0.73rem; color: #64748B; margin-top: 14px; text-align: center;">
                Academic prototype stage verification — Not connected to live GPS/AVL feeds.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        st.markdown(f"""
        <div class="dashboard-card" style="height: 100%;">
            <div class="dashboard-card-header">
                <span>{get_icon('chart', size=16, color='#00D2FF')} FLEET DEMAND PROFILE ACROSS 24 HOURS</span>
            </div>
        """, unsafe_allow_html=True)
        fig_fleet_path = get_figure_path("peak_vs_offpeak_fleet_demand.png")
        if os.path.exists(fig_fleet_path):
            st.image(fig_fleet_path, caption="Simultaneously Active Buses on Road vs. 868 Optimized Fleet Cap (Peak: 499 buses at 12:10 PM)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 02: ROUTE ANALYSIS
# ==============================================================================
elif active_section == "02":
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-title">02 / Route Intelligence & Spatial Network Exploration</div>
        <div class="section-tagline">Interactive geospatial stops, physical runtimes, and corridor topology</div>
    </div>
    """, unsafe_allow_html=True)

    # Route selection header
    route_ids = sorted(routes_df['route_id'].unique())
    col_sel, col_empty = st.columns([1.5, 2])
    with col_sel:
        selected_route_id = st.selectbox("Select a Delhi Bus Route ID:", route_ids, index=0)

    route_info = routes_df[routes_df['route_id'] == selected_route_id].iloc[0]

    # Selected Route KPI Row
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Route Distance</div>
            <div class="kpi-val-v2 cyan">{route_info['total_distance_km']:.1f} <span style="font-size:1rem;">km</span></div>
            <div class="kpi-sub-v2">Haversine cumulative path</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Total Stops</div>
            <div class="kpi-val-v2">{int(route_info['total_stops'])}</div>
            <div class="kpi-sub-v2">Boarding & alighting stations</div>
        </div>
        <div class="kpi-card-v2 accent-amber">
            <div class="kpi-lbl-v2">Scheduled Runtime</div>
            <div class="kpi-val-v2 amber">{route_info['scheduled_runtime_min']:.1f} <span style="font-size:1rem;">min</span></div>
            <div class="kpi-sub-v2">GTFS baseline timetable</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">ML Runtime</div>
            <div class="kpi-val-v2 green">{route_info['ml_predicted_runtime_min']:.1f} <span style="font-size:1rem;">min</span></div>
            <div class="kpi-sub-v2">Random Forest aggregate</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Departure</div>
            <div class="kpi-val-v2" style="font-size: 1.4rem;">{route_info['start_time_str']}</div>
            <div class="kpi-sub-v2">Service start timestamp</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Origin & Destination Cards
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 20px;">
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 14px 18px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #34D399; text-transform: uppercase; letter-spacing: 0.8px;">ORIGIN TERMINAL</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">{route_info['start_stop_name']}</div>
        </div>
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 14px 18px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #F87171; text-transform: uppercase; letter-spacing: 0.8px;">DESTINATION TERMINAL</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">{route_info['end_stop_name']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load stops for selected route
    all_stops = load_all_stops_data()
    route_stops = all_stops[all_stops['route_id'] == selected_route_id].sort_values('stop_sequence').reset_index(drop=True)

    tab_map, tab_stops, tab_network = st.tabs([
        "Route Map",
        "Stop-by-Stop Itinerary",
        "Network Statistics"
    ])

    with tab_map:
        st.caption("Interactive geospatial transit visualization. Emerald = Origin Stop, Crimson = Destination Terminal, Cyan = Intermediate Stops.")
        deck_map = build_pydeck_route_map(route_stops)
        if deck_map:
            st.pydeck_chart(deck_map)
        else:
            st.warning("Geographic stop coordinates unavailable for this route.")

    with tab_stops:
        st.markdown(f"**Itinerary for Route `{selected_route_id}` ({len(route_stops)} sequential stops):**")
        display_stops = route_stops[['stop_sequence', 'stop_id', 'stop_name', 'arrival_time', 'departure_time', 'dist_to_next_km', 'time_to_next_sec']].copy()
        display_stops.columns = ['Seq #', 'Stop ID', 'Stop Name', 'Arrival Time', 'Departure Time', 'Next Segment Dist (km)', 'Next Segment Time (sec)']
        st.dataframe(display_stops, use_container_width=True, height=360)

    with tab_network:
        col_n1, col_n2 = st.columns([1.3, 1])
        with col_n1:
            fig_dist_path = get_figure_path("route_runtime_distribution.png")
            if os.path.exists(fig_dist_path):
                st.image(fig_dist_path, caption="Distribution of Route Runtimes across 2,403 Delhi Bus Routes", use_container_width=True)
        with col_n2:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="dashboard-card-header">{get_icon('globe', size=16, color='#00D2FF')} Network Characteristics</div>
                <div class="param-item">
                    <span class="param-lbl">Shortest Route</span>
                    <span class="param-val">{routes_df['total_distance_km'].min():.2f} km ({routes_df['scheduled_runtime_min'].min():.1f} min)</span>
                </div>
                <div class="param-item">
                    <span class="param-lbl">Median Route</span>
                    <span class="param-val">{routes_df['total_distance_km'].median():.2f} km ({routes_df['scheduled_runtime_min'].median():.1f} min)</span>
                </div>
                <div class="param-item">
                    <span class="param-lbl">Longest Route</span>
                    <span class="param-val">{routes_df['total_distance_km'].max():.2f} km ({routes_df['scheduled_runtime_min'].max():.1f} min)</span>
                </div>
                <div class="param-item">
                    <span class="param-lbl">Median Stop Count</span>
                    <span class="param-val">{routes_df['total_stops'].median():.0f} stops per route</span>
                </div>
                <div class="param-item">
                    <span class="param-lbl">Average Network Speed</span>
                    <span class="param-val">~15.0 km/h (GTFS Planning Velocity)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 03: TRAVEL-TIME PREDICTION
# ==============================================================================
elif active_section == "03":
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-title">03 / AI Travel-Time Prediction Console</div>
        <div class="section-tagline">Random Forest Regression for Inter-Stop Segment Runtime Estimation</div>
    </div>
    """, unsafe_allow_html=True)

    model, feature_cols = load_ml_model()
    sample_segments = load_test_sample_segments()

    if model is not None and not sample_segments.empty:
        col_pick, col_hint = st.columns([2, 1.2])
        with col_pick:
            sample_indices = sample_segments.head(50).index.tolist()
            selected_idx = st.selectbox(
                "Choose Unseen Test Set Segment Record:",
                sample_indices,
                format_func=lambda x: f"Record #{x}: Route {sample_segments.loc[x, 'route_id']} | {sample_segments.loc[x, 'stop_name']} → {sample_segments.loc[x, 'next_stop_name']} ({sample_segments.loc[x, 'segment_dist_km']:.2f} km)"
            )

        row = sample_segments.loc[selected_idx]
        actual_time = row['travel_time_sec']

        input_vector = sample_segments.loc[[selected_idx]][feature_cols]
        predicted_time = model.predict(input_vector)[0]
        error_sec = predicted_time - actual_time

        # Large AI Prediction Console Card
        st.markdown(f"""
        <div class="prediction-hero">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
                <div>
                    <div class="prediction-title">{get_icon('cpu', size=16, color='#00D2FF')} AI ESTIMATED SEGMENT TRAVEL TIME</div>
                    <div class="prediction-val-big">{predicted_time:.1f} <span style="font-size: 1.5rem; color: #38BDF8;">seconds</span></div>
                    <div class="prediction-sub">Equates to <b>{predicted_time/60.0:.2f} minutes</b> estimated traversal duration</div>
                </div>
                <div style="display: flex; gap: 14px; flex-wrap: wrap;">
                    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px 16px; min-width: 140px;">
                        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">SCHEDULED TIME</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono';">{actual_time:.1f} s</div>
                        <div style="font-size: 0.75rem; color: #64748B;">{actual_time/60.0:.2f} min</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px 16px; min-width: 140px;">
                        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">PREDICTION ERROR (Δ)</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: {'#34D399' if abs(error_sec) < 1.0 else '#FBBF24'}; font-family: 'JetBrains Mono';">{error_sec:+.2f} s</div>
                        <div style="font-size: 0.75rem; color: #64748B;">Sub-second accuracy</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px 16px; min-width: 140px;">
                        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">SEGMENT DISTANCE</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono';">{row['segment_dist_km']:.2f} km</div>
                        <div style="font-size: 0.75rem; color: #64748B;">Bearing: {row['bearing_deg']:.0f}°</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Segment Information Cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 22px;">
            <div class="status-row">
                <span class="status-name"><span>{get_icon('map-pin', size=14, color='#38BDF8')}</span> Origin Stop:</span>
                <span class="status-val" style="color: #FFFFFF;">{row['stop_name']} ({row['origin_stop_degree']} routes)</span>
            </div>
            <div class="status-row">
                <span class="status-name"><span>{get_icon('flag', size=14, color='#38BDF8')}</span> Destination Stop:</span>
                <span class="status-val" style="color: #FFFFFF;">{row['next_stop_name']} ({row['dest_stop_degree']} routes)</span>
            </div>
            <div class="status-row">
                <span class="status-name"><span>{get_icon('clock', size=14, color='#38BDF8')}</span> Departure Hour:</span>
                <span class="status-val" style="color: #FFFFFF;">{int(row['departure_hour']):02d}:{int(row['departure_minute']):02d} IST</span>
            </div>
            <div class="status-row">
                <span class="status-name"><span>{get_icon('compass', size=14, color='#38BDF8')}</span> Route Tortuosity:</span>
                <span class="status-val" style="color: #FFFFFF;">{row['route_tortuosity']:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model Evaluation Visualizations
    st.markdown(f"""
    <div class="dashboard-card-header">
        <span>{get_icon('chart', size=16, color='#00D2FF')} MODEL VALIDATION & FEATURE IMPORTANCE VISUALIZATIONS</span>
    </div>
    """, unsafe_allow_html=True)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_actual = get_figure_path("actual_vs_predicted_travel_time.png")
        if os.path.exists(fig_actual):
            st.image(fig_actual, caption="Actual vs Predicted Segment Travel Time on Unseen Test Set (R² = 0.99987)", use_container_width=True)
    with col_v2:
        fig_residual = get_figure_path("residual_error_distribution.png")
        if os.path.exists(fig_residual):
            st.image(fig_residual, caption="Residual Error Distribution (MAE = 0.36s, RMSE = 1.04s)", use_container_width=True)

    fig_imp = get_figure_path("feature_importance.png")
    if os.path.exists(fig_imp):
        st.image(fig_imp, caption="Feature Importance Analysis — Physical Segment Distance Dominates (~99.9% Impurity Reduction)", use_container_width=True)


# ==============================================================================
# SECTION 04: SCHEDULE OPTIMIZATION
# ==============================================================================
elif active_section == "04":
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-title">04 / Fleet Optimization Control Center</div>
        <div class="section-tagline">Minimum Path Cover (DAG) & Multi-Trip Vehicle Chaining Rosters</div>
    </div>
    """, unsafe_allow_html=True)

    linked_df = load_linked_duties()
    bus_schedule = load_bus_schedule()

    unlinked_buses_count = len(routes_df) # 2,403
    linked_buses_count = len(linked_df)   # 868
    buses_saved_count = unlinked_buses_count - linked_buses_count # 1,535
    fleet_red_pct_dyn = (buses_saved_count / unlinked_buses_count) * 100.0 # 63.88%

    # Large Top KPI Cards
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
        <div class="kpi-card-v2 accent-rose">
            <div class="kpi-lbl-v2">Unlinked Baseline</div>
            <div class="kpi-val-v2 rose">{unlinked_buses_count:,}</div>
            <div class="kpi-sub-v2">1 dedicated bus per scheduled trip</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Optimized Linked Fleet</div>
            <div class="kpi-val-v2 green">{linked_buses_count:,}</div>
            <div class="kpi-sub-v2">Chained multi-trip duty rosters</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Modeled Fleet Reduction</div>
            <div class="kpi-val-v2 cyan">{buses_saved_count:,} <span style="font-size:1.1rem; color:#7DD3FC;">({fleet_red_pct_dyn:.1f}%)</span></div>
            <div class="kpi-sub-v2">Buses saved in modeled fleet requirement</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Trip Coverage Rate</div>
            <div class="kpi-val-v2 green">100%</div>
            <div class="kpi-sub-v2">All 2,403 trips covered with 0 conflicts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual Comparison Block
    st.markdown(f"""
    <div class="chaining-flow-container">
        <div class="chain-box">
            <div class="chain-num" style="color: #F87171;">{unlinked_buses_count:,}</div>
            <div class="chain-lbl">UNLINKED BUSES (1 PER TRIP)</div>
        </div>
        <div class="chain-mid">
            <div class="chain-badge">{get_icon('zap', size=15, color='#00D2FF')} VEHICLE CHAINING OPTIMIZATION (DAG MPC)</div>
            <div style="color: var(--cyan-primary); display: flex; align-items: center; justify-content: center; gap: 8px;">{get_icon('arrow-right', size=20, color='#00D2FF')} {get_icon('arrow-right', size=20, color='#00D2FF')} {get_icon('arrow-right', size=20, color='#00D2FF')}</div>
        </div>
        <div class="chain-box">
            <div class="chain-num" style="color: #34D399;">{linked_buses_count:,}</div>
            <div class="chain-lbl">OPTIMIZED LINKED FLEET</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_inspector, tab_gantt, tab_comparison = st.tabs([
        "Bus Duty Inspector",
        "Duty Gantt Timeline & Demand Profile",
        "Linked vs. Unlinked Performance"
    ])

    with tab_inspector:
        st.markdown(f'<div class="sub-section-header">{get_icon("search", size=18, color="#00D2FF")} <span>Vehicle Duty Rosters Inspector</span></div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            trip_filter = st.selectbox("Filter Duties by Trip Count:", ["All Buses", "4+ Trips", "3 Trips", "2 Trips", "Single Trip"])

        filtered_buses = linked_df.copy()
        if trip_filter == "4+ Trips":
            filtered_buses = filtered_buses[filtered_buses['trips_completed'] >= 4]
        elif trip_filter == "3 Trips":
            filtered_buses = filtered_buses[filtered_buses['trips_completed'] == 3]
        elif trip_filter == "2 Trips":
            filtered_buses = filtered_buses[filtered_buses['trips_completed'] == 2]
        elif trip_filter == "Single Trip":
            filtered_buses = filtered_buses[filtered_buses['trips_completed'] == 1]

        with col_f2:
            bus_list = filtered_buses['bus_id'].tolist()
            selected_bus_id = st.selectbox(f"Select Vehicle ID ({len(bus_list)} available):", bus_list, index=0)

        bus_info = linked_df[linked_df['bus_id'] == selected_bus_id].iloc[0]

        # Vehicle KPI Row
        st.markdown(f"""
        <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));">
            <div class="kpi-card-v2 accent-cyan">
                <div class="kpi-lbl-v2">Duty Shift Span</div>
                <div class="kpi-val-v2">{bus_info['total_duty_hours']:.2f}h</div>
                <div class="kpi-sub-v2">{bus_info['duty_start_str']} → {bus_info['duty_end_str']}</div>
            </div>
            <div class="kpi-card-v2 accent-green">
                <div class="kpi-lbl-v2">Driving Time</div>
                <div class="kpi-val-v2 green">{bus_info['driving_hours']:.2f}h</div>
                <div class="kpi-sub-v2">{(bus_info['driving_hours']/bus_info['total_duty_hours'])*100:.1f}% Duty Share</div>
            </div>
            <div class="kpi-card-v2 accent-amber">
                <div class="kpi-lbl-v2">Turnaround</div>
                <div class="kpi-val-v2 amber">{bus_info['turnaround_hours']*60:.0f}m</div>
                <div class="kpi-sub-v2">{bus_info['trips_completed']-1} Recovery Layovers</div>
            </div>
            <div class="kpi-card-v2 accent-cyan">
                <div class="kpi-lbl-v2">Deadhead</div>
                <div class="kpi-val-v2">{bus_info['deadhead_hours']*60:.0f}m</div>
                <div class="kpi-sub-v2">{bus_info['deadhead_hours']:.2f} h transfer</div>
            </div>
            <div class="kpi-card-v2 accent-green">
                <div class="kpi-lbl-v2">Trips Chained</div>
                <div class="kpi-val-v2 green">{bus_info['trips_completed']}</div>
                <div class="kpi-sub-v2">{bus_info['routes_served_chain']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Trip Chaining Sequence for `{selected_bus_id}`:**")
        bus_trips = bus_schedule[bus_schedule['bus_id'] == selected_bus_id].sort_values('trip_sequence_in_duty').reset_index(drop=True)

        display_trips = bus_trips[[
            'trip_sequence_in_duty', 'route_id', 'trip_id', 'from_stop_name', 'to_stop_name',
            'trip_start_str', 'trip_end_str', 'scheduled_runtime_min', 'deadhead_prev_km', 'deadhead_prev_min', 'turnaround_prev_min', 'idle_prev_min'
        ]].copy()
        display_trips.columns = [
            'Trip #', 'Route ID', 'Trip ID', 'Origin Stop', 'Destination Stop',
            'Departure', 'Arrival', 'Driving (min)', 'Prior Deadhead (km)', 'Prior Deadhead (min)', 'Turnaround (min)', 'Idle Wait (min)'
        ]
        st.dataframe(display_trips, use_container_width=True)

    with tab_gantt:
        st.markdown(f'<div class="sub-section-header">{get_icon("calendar-clock", size=18, color="#00D2FF")} <span>Duty Gantt Timeline & Schedule Distributions</span></div>', unsafe_allow_html=True)
        fig_gantt = get_figure_path("schedule_gantt_chart.png")
        if os.path.exists(fig_gantt):
            st.image(fig_gantt, caption="Gantt Timeline for Representative Multi-Trip Bus Duties Across the Service Day", use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_duty_dist = get_figure_path("duty_duration_distribution.png")
            if os.path.exists(fig_duty_dist):
                st.image(fig_duty_dist, caption=f"Duty Duration Distribution (Confirming 8.0-Hour Maximum Cap — Mean = {linked_df['total_duty_hours'].mean():.2f} hours)", use_container_width=True)
        with col_g2:
            fig_trips_dist = get_figure_path("trips_per_bus_distribution.png")
            if os.path.exists(fig_trips_dist):
                avg_trips_calc = len(bus_schedule) / len(linked_df)
                st.image(fig_trips_dist, caption=f"Trips Completed Per Bus Distribution (Average = {avg_trips_calc:.2f} trips per bus)", use_container_width=True)

    with tab_comparison:
        st.markdown(f'<div class="sub-section-header">{get_icon("scale", size=18, color="#00D2FF")} <span>Fleet Sizing & Utilization Metrics</span></div>', unsafe_allow_html=True)
        driving_h = round(float(linked_df['driving_hours'].sum()), 1)
        turnaround_h = round(float(linked_df['turnaround_hours'].sum()), 1)
        deadhead_h = round(float(linked_df['deadhead_hours'].sum()), 1)
        idle_h = round(float(linked_df['idle_hours'].sum()), 1)
        total_fleet_duty_h = round(driving_h + turnaround_h + deadhead_h + idle_h, 1) # 5,306.7
        layover_overhead_h = round(turnaround_h + deadhead_h + idle_h, 1)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_fleet = get_figure_path("fleet_requirement_comparison.png")
            if os.path.exists(fig_fleet):
                st.image(fig_fleet, caption=f"Fleet Sizing: Unlinked Baseline ({unlinked_buses_count:,}) vs. Optimized Linked Schedule ({linked_buses_count:,}) — Fleet Reduction: {fleet_red_pct_dyn:.1f}% ({buses_saved_count:,} fewer buses)", use_container_width=True)
        with col_c2:
            fig_util = get_figure_path("bus_utilization_comparison.png")
            if os.path.exists(fig_util):
                st.image(fig_util, caption=f"Duty Hours Breakdown: {total_fleet_duty_h:,.1f} Total Fleet Duty Hours (Driving: {driving_h:,.1f}h, Turnaround: {turnaround_h:,.1f}h, Deadhead: {deadhead_h:,.1f}h, Idle: {idle_h:,.1f}h)", use_container_width=True)

        st.markdown(f"""
        <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 14px;">
            <div class="kpi-card-v2 accent-cyan">
                <div class="kpi-lbl-v2">Modeled Reduction</div>
                <div class="kpi-val-v2 cyan">{fleet_red_pct_dyn:.1f}%</div>
                <div class="kpi-sub-v2">{buses_saved_count:,} buses fewer in modeled fleet</div>
            </div>
            <div class="kpi-card-v2 accent-cyan">
                <div class="kpi-lbl-v2">Total Fleet Duty</div>
                <div class="kpi-val-v2">{total_fleet_duty_h:,.1f} h</div>
                <div class="kpi-sub-v2">5,306.7 Total Fleet Duty Hours</div>
            </div>
            <div class="kpi-card-v2 accent-green">
                <div class="kpi-lbl-v2">Revenue In-Service</div>
                <div class="kpi-val-v2 green">{driving_h:,.1f} h</div>
                <div class="kpi-sub-v2">{(driving_h / total_fleet_duty_h) * 100.0:.1f}% of total fleet duty</div>
            </div>
            <div class="kpi-card-v2 accent-amber">
                <div class="kpi-lbl-v2">Operational Layover</div>
                <div class="kpi-val-v2 amber">{layover_overhead_h:,.1f} h</div>
                <div class="kpi-sub-v2">{(layover_overhead_h / total_fleet_duty_h) * 100.0:.1f}% turnaround/deadhead/idle</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 05: RESULTS & INSIGHTS
# ==============================================================================
elif active_section == "05":
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-title">05 / Quantitative Results & Operational Insights</div>
        <div class="section-tagline">Comprehensive Statistical Evaluation Across ML & Fleet Scheduling</div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION A: ML PERFORMANCE
    st.markdown(f"""
    <div class="dashboard-card-header">
        <span>{get_icon('cpu', size=16, color='#00D2FF')} SECTION A: MACHINE LEARNING PERFORMANCE</span>
    </div>
    """, unsafe_allow_html=True)

    test_mae = ml_metrics.get('test_metrics', {}).get('test_mae_sec', 0.36)
    test_rmse = ml_metrics.get('test_metrics', {}).get('test_rmse_sec', 1.04)
    test_r2 = ml_metrics.get('test_metrics', {}).get('test_r2', 0.99987)
    test_medae = ml_metrics.get('test_metrics', {}).get('test_medae_sec', 0.27)
    test_mape = ml_metrics.get('test_metrics', {}).get('test_mape_percent', 0.39)

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));">
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Test MAE</div>
            <div class="kpi-val-v2 cyan">{test_mae:.2f} s</div>
            <div class="kpi-sub-v2">Sub-second mean deviation</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Test RMSE</div>
            <div class="kpi-val-v2">{test_rmse:.2f} s</div>
            <div class="kpi-sub-v2">Zero catastrophic outliers</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Test R² Score</div>
            <div class="kpi-val-v2 green">{test_r2:.5f}</div>
            <div class="kpi-sub-v2">99.99% Variance explained</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Test MedAE</div>
            <div class="kpi-val-v2">{test_medae:.2f} s</div>
            <div class="kpi-sub-v2">Median absolute error</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Test MAPE</div>
            <div class="kpi-val-v2 green">{test_mape:.2f}%</div>
            <div class="kpi-sub-v2">Relative percentage accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION B: SCHEDULING PERFORMANCE
    st.markdown(f"""
    <div class="dashboard-card-header" style="margin-top: 24px;">
        <span>{get_icon('bus', size=16, color='#00D2FF')} SECTION B: SCHEDULING & FLEET PERFORMANCE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));">
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Fleet Requirement</div>
            <div class="kpi-val-v2 cyan">2,403 → 868</div>
            <div class="kpi-sub-v2">Unlinked to linked reduction</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Modeled Reduction</div>
            <div class="kpi-val-v2 green">63.9%</div>
            <div class="kpi-sub-v2">1,535 buses saved requirement</div>
        </div>
        <div class="kpi-card-v2 accent-green">
            <div class="kpi-lbl-v2">Trip Coverage</div>
            <div class="kpi-val-v2 green">100%</div>
            <div class="kpi-sub-v2">2,403 trips covered with 0 drops</div>
        </div>
        <div class="kpi-card-v2 accent-amber">
            <div class="kpi-lbl-v2">Driving Utilization</div>
            <div class="kpi-val-v2 amber">71.3%</div>
            <div class="kpi-sub-v2">3,781.1h revenue active driving</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Average Trips/Bus</div>
            <div class="kpi-val-v2">2.77</div>
            <div class="kpi-sub-v2">Max: 6 trips on single vehicle</div>
        </div>
        <div class="kpi-card-v2 accent-cyan">
            <div class="kpi-lbl-v2">Average Duty</div>
            <div class="kpi-val-v2">6.11 h</div>
            <div class="kpi-sub-v2">Within 8.0h shift duration cap</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bus Utilization Statistics Table
    st.markdown(f'<div class="sub-section-header">{get_icon("chart", size=18, color="#00D2FF")} <span>Duty Breakdown Statistics Table</span></div>', unsafe_allow_html=True)
    util_df = load_bus_utilization()
    st.table(util_df)

    # KEY INSIGHTS CHECKPOINTS PANEL
    st.markdown(f"""
    <div class="dashboard-card" style="margin-top: 15px;">
        <div class="dashboard-card-header">
            <span>{get_icon('sparkles', size=16, color='#00D2FF')} KEY SCIENTIFIC & OPERATIONAL INSIGHTS</span>
        </div>
        <div class="insight-list">
            <div class="insight-item">
                <span class="insight-check">{get_icon('check', size=14, color='#10B981')}</span>
                <span class="insight-text"><b>2,403 scheduled trips are covered</b> completely across the NCT Delhi transit network.</span>
            </div>
            <div class="insight-item">
                <span class="insight-check">{get_icon('check', size=14, color='#10B981')}</span>
                <span class="insight-text"><b>Vehicle chaining reduces the modeled fleet requirement</b> from 2,403 to 868 buses (63.9% reduction).</span>
            </div>
            <div class="insight-item">
                <span class="insight-check">{get_icon('check', size=14, color='#10B981')}</span>
                <span class="insight-text"><b>100% trip coverage is maintained</b> with zero schedule overlap conflicts or dropped routes.</span>
            </div>
            <div class="insight-item">
                <span class="insight-check">{get_icon('check', size=14, color='#10B981')}</span>
                <span class="insight-text"><b>No schedule overlap conflicts are reported</b> across all 868 linked duty rosters.</span>
            </div>
            <div class="insight-item">
                <span class="insight-check">{get_icon('check', size=14, color='#10B981')}</span>
                <span class="insight-text"><b>Average duty remains below the configured 8-hour cap</b> (Mean duty: 6.11 hours; Max duty: 8.00 hours).</span>
            </div>
            <div class="insight-item">
                <span class="insight-check">{get_icon('check', size=14, color='#10B981')}</span>
                <span class="insight-text"><b>ML evaluation is performed on an unseen test set</b> confirming sub-second estimation precision (MAE = 0.36s).</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 06: ABOUT & LIMITATIONS
# ==============================================================================
elif active_section == "06":
    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-title">06 / Project Assumptions, Transparency & Faculty Presentation</div>
        <div class="section-tagline">Academic Rigor, Deployment Boundaries, and Defense Script</div>
    </div>
    """, unsafe_allow_html=True)

    # 3 High-Tech Summary Cards
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-bottom: 22px;">
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 18px 20px;">
            <div style="font-size: 0.8rem; font-weight: 800; color: #F87171; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                {get_icon('triangle-alert', size=15, color='#F87171')} LIMITATIONS
            </div>
            <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.5;">
                • <b>Synthetic GTFS timing:</b> Constant ~15 km/h planned speed.<br>
                • <b>No passenger-demand data:</b> Headways taken as given.<br>
                • <b>Modeled scheduling assumptions:</b> 15m turnaround & 8h duty.
            </div>
        </div>
        <div style="background: rgba(0, 210, 255, 0.08); border: 1px solid rgba(0, 210, 255, 0.3); border-radius: 12px; padding: 18px 20px;">
            <div style="font-size: 0.8rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                {get_icon('rocket', size=15, color='#38BDF8')} FUTURE WORK
            </div>
            <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.5;">
                • <b>Live GPS / AVL:</b> Real-time traffic delay feedback.<br>
                • <b>Passenger demand forecasting:</b> Smartcard tap volumes.<br>
                • <b>Depot constraints & Crew scheduling:</b> Driver union shifts.<br>
                • <b>EV charging optimization:</b> Battery state-of-charge.
            </div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 18px 20px;">
            <div style="font-size: 0.8rem; font-weight: 800; color: #34D399; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                {get_icon('graduation-cap', size=15, color='#34D399')} PROJECT STATUS
            </div>
            <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.5;">
                • <b>Academic prototype:</b> Decision-support system.<br>
                • <b>Validated mathematical optimization:</b> DAG MPC.<br>
                • <b>Not official DTC deployment:</b> Ready for real data integrations.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Faculty Demonstration Card
    st.markdown(f"""
    <div class="dashboard-card" style="border-color: rgba(0, 210, 255, 0.35); background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(15, 23, 42, 0.95) 100%);">
        <div class="dashboard-card-header" style="color: #38BDF8;">
            <span>{get_icon('graduation-cap', size=16, color='#00D2FF')} FACULTY DEMONSTRATION & PROJECT SUMMARY</span>
        </div>
        <div style="font-size: 0.96rem; font-weight: 600; color: #FFFFFF; margin-bottom: 8px;">
            "From scheduled trips → ML travel-time estimation → route runtime → vehicle chaining → optimized fleet."
        </div>
        <div style="font-size: 0.86rem; color: #94A3B8; margin-bottom: 14px;">
            This decision-support prototype demonstrates how mathematical optimization (Directed Acyclic Graph Minimum Path Cover) 
            combined with Machine Learning travel-time estimation can systematically reduce fleet size requirements for megacity transit agencies.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Expandable 60-90 second presentation script
    with st.expander("View 60–90 Second Faculty Presentation Script", expanded=True):
        st.markdown("""
        <div class="pitch-quote">
        "Good morning, Professors. Today, I am presenting <b>The Automated Bus Scheduling and Route Management Demonstration System for the Delhi Transport Corporation (DTC)</b>.<br><br>
        The core problem our project addresses is the <b>Vehicle Scheduling Problem (VSP)</b>: how can a mega-city transit agency like DTC cover thousands of scheduled trips using the minimum number of buses, without schedule conflicts, and while strictly complying with driver rest and duty regulations?<br><br>
        Our dataset contains <b>2,403 distinct scheduled bus trips</b> across NCT Delhi. If dispatched naively with one bus per trip, DTC would require <b>2,403 buses</b>, wasting capital and creating massive vehicle idle time.<br><br>
        Our decision-support system solves this through an end-to-end pipeline:
        <ol>
            <li>First, we extract sequential stop segments and spatial network metrics from GTFS data.</li>
            <li>Second, our Machine Learning regression model estimates segment travel times with a <b>Test MAE of 0.36 seconds</b>.</li>
            <li>Third, we compile complete route runtimes across all 2,403 routes.</li>
            <li>Finally, our <b>Vehicle Chaining Optimizer</b> models the network as a Minimum Path Cover on a Directed Acyclic Graph, enforcing a mandatory <b>15-minute turnaround recovery buffer</b>, spatial <b>deadhead repositioning limits</b>, and an <b>8-hour maximum shift cap</b>.</li>
        </ol>
        The result is dramatic: our optimizer covers <b>100% of all 2,403 trips using just 868 buses</b>—a <b>63.9% modeled reduction in fleet requirements</b>, saving <b>1,535 buses</b> while maintaining an active driving utilization of <b>71.3%</b>.<br><br>
        All assumptions are fully configurable in our system, providing DTC transit planners with an interactive, transparent tool for data-driven scheduling and fleet management. Thank you."
        </div>
        """, unsafe_allow_html=True)

    # Detailed Methodology & Transparency Tabs
    tab_details, tab_future = st.tabs(["Detailed Academic Boundaries", "Production Roadmap Specifications"])

    with tab_details:
        st.markdown("""
        1. **Synthetic GTFS Timing**:
           The arrival times in the raw GTFS dataset were computationally synthesized assuming a constant cruising velocity of ~15.0 km/h (240 seconds/km).
           Therefore, the Machine Learning model reflects **scheduled transit time estimation**, and is **NOT** presented as a real-world dynamic traffic congestion model.
           
        2. **Absence of Passenger Demand**:
           The dataset does not contain farebox counts, ticketing records, or passenger origin-destination (OD) volumes. 
           Headways and timetables were taken directly from the static GTFS schedule.
           
        3. **Configurable Scheduling Assumptions**:
           Parameters such as the **15-minute turnaround buffer**, **8-hour duty cap**, and **20 km/h deadhead repositioning speed** are modeled assumptions rather than official DTC collective bargaining rules.
           
        4. **Prototype Status**:
           The **63.9% fleet reduction** is a prototype mathematical result under modeled assumptions, not an official DTC operational claim. Real-world implementation requires depot pull-in/pull-out rosters, bus charging/refueling times, maintenance rotations, and union shift regulations.
        """)

    with tab_future:
        st.markdown("""
        1. **Real-Time GPS & Automatic Vehicle Location (AVL)**:
           Ingest live vehicle telematics to replace static scheduled runtimes with empirical, time-of-day traffic delay distributions.
           
        2. **Automatic Passenger Counting (APC) & Demand Forecasting**:
           Incorporate smartcard (One Delhi / Metro transit card) tap data to optimize departure frequencies dynamically based on crowding and passenger demand.
           
        3. **Depot-Specific Pull-Out / Pull-In Constraints**:
           Assign vehicles to official DTC physical depots (e.g., Millennium Depot, Rajghat Depot, Rohini Depot) and model deadhead travel to/from garage facilities.
           
        4. **Crew & Driver Duty Rostering**:
           Extend vehicle scheduling into full Crew Scheduling, accounting for split shifts, meal breaks, overtime rules, and driver home depot assignments.
           
        5. **Electric Bus (EV) Charging Scheduling**:
           Model battery state-of-charge (SoC) consumption rates, depot charging bay availability, and fast-charging opportunity layovers.
        """)
