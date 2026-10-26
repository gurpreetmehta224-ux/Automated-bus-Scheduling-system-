# DTC Operations — AI-Assisted Transit Scheduling & Fleet Optimization

An enterprise-grade public transit scheduling and fleet optimization system designed for the **Delhi Transport Corporation (DTC)**. The system models the **Vehicle Scheduling Problem (VSP)** across the NCT Delhi bus transit network using machine learning travel-time prediction and Directed Acyclic Graph (DAG) Minimum Path Cover vehicle chaining.

---

## Key Highlights & Performance

- **2,403 Scheduled Trips Covered**: Complete coverage across NCT Delhi without dropped routes.
- **63.9% Fleet Reduction**: Optimizes the modeled vehicle requirement from **2,403 unlinked buses down to 868 linked buses** (saving 1,535 buses).
- **100% Trip Coverage Rate**: Zero schedule overlap conflicts across all linked duty rosters.
- **Sub-Second ML Travel-Time Prediction**: Random Forest regression trained on spatial GTFS stop features achieving **0.36s Test MAE** and $R^2 = 0.99987$.
- **Regulatory Duty Compliance**: Enforces an 8.0-hour maximum shift cap (average duty: 6.11 hours), mandatory 15-minute turnaround recovery buffers, and spatial deadhead repositioning limits.

---

## System Architecture

```
GTFS Data Ingestion ➔ Data Processing & Cleaning ➔ Spatial Feature Engineering ➔ ML Travel-Time Model ➔ Route Runtime Engine ➔ Vehicle Chaining (DAG MPC) ➔ Optimized Duty Schedule
```

1. **GTFS Data Pipeline**: Ingests route shapes, timetables, and stop sequences across Delhi.
2. **Spatial Feature Engineering**: Computes Haversine distances, route tortuosity, bearing angles, network connectivity degrees, and transit corridor hub classifications.
3. **Random Forest Travel-Time Engine**: Predicts inter-stop segment runtimes with delay buffers for traffic variance.
4. **DAG Vehicle Chaining Optimizer**: Models compatible trip-to-trip transfers as a Directed Acyclic Graph Minimum Path Cover to chain buses efficiently while guaranteeing turnaround buffers and driver duty limits.
5. **Transit Operations Dashboard**: Interactive Streamlit control center with vector Lucide visual iconography, dark transit-control aesthetics, and real-time scenario parameter controls.

---

## Dashboard Navigation Sections

1. **01 Overview**: High-level KPIs, 6-stage end-to-end pipeline architecture, live system status, and 24-hour fleet demand profiles.
2. **02 Route Analysis**: Geospatial route visualization, corridor topology, stop-by-stop itineraries, and network speed characteristics.
3. **03 Travel-Time Prediction**: Interactive AI inference console for unseen test segments, scheduled vs. predicted runtime deltas, and feature importance rankings.
4. **04 Schedule Optimization**: Multi-trip vehicle duty inspector, Gantt timeline schedules, turnaround layover analysis, and linked vs. unlinked fleet comparisons.
5. **05 Results & Insights**: Statistical model evaluation tables, duty breakdown metrics, and verified operational findings.
6. **06 About & Limitations**: Project scope, academic boundaries, and production roadmap specifications (AVL, passenger demand, EV charging).

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gurpreetmehta224-ux/Automated-bus-Scheduling-system-.git
   cd Automated-bus-Scheduling-system-
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Transit Control Center**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## Project Structure

```
├── app/
│   ├── app.py             # Streamlit application entrypoint
│   ├── dashboard.py       # Main dashboard control center UI & analytics
│   ├── icons.py           # Lucide vector SVG icon family
│   ├── utils.py           # Data loaders and visualization helpers
│   └── bus_icon.svg       # Vector transit bus icon
├── src/
│   ├── eda_route_analysis.py # GTFS exploratory data analysis & network features
│   ├── ml_model.py           # Random forest travel-time model training & evaluation
│   └── bus_scheduler.py      # DAG minimum path cover vehicle chaining engine
├── data/                  # GTFS stops, routes, and network tables
├── models/                # Trained Random Forest model & feature specifications
├── outputs/               # Precomputed duty schedules, figures, and KPIs
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## License & Academic Disclaimer

Developed as an academic decision-support prototype for transit engineering and fleet management. Datasets and timetables are modeled based on published Delhi GTFS transit feeds.
