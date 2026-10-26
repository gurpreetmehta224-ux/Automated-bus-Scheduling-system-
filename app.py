"""
Main Streamlit application entry point.
Automated Bus Scheduling and Route Management System.
"""
import sys
import runpy
from pathlib import Path

# Resolve base project directory
BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(1, str(APP_DIR))

# Execute the DTC Bus Scheduling & Route Management Dashboard
dashboard_path = APP_DIR / "dashboard.py"
runpy.run_path(str(dashboard_path), run_name="__main__")
