"""
Main Streamlit application entry point.
Redirects to / runs the redesigned DTC Bus Scheduling & Route Management Dashboard.
"""
import sys
import runpy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(1, str(APP_DIR))

# Execute the redesigned dashboard module dynamically on every run/rerun
dashboard_path = APP_DIR / "dashboard.py"
runpy.run_path(str(dashboard_path), run_name="__main__")
