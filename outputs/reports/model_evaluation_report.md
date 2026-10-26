# Module 4: Supervised ML Model Evaluation Report

**Project**: Automated Bus Scheduling (Delhi Transport Corporation - DTC)  
**Target Variable**: `travel_time_sec` (Segment Transit Duration in Seconds)  
**Pipeline Source**: [`src/ml_model.py`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/src/ml_model.py)  
**Trained Model Artifact**: [`models/best_travel_time_model.joblib`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/models/best_travel_time_model.joblib)  

---

## 1. Objective & Target Variable Formulation

Module 4 establishes a supervised machine learning regression pipeline to predict **`travel_time_sec`**, the duration (in seconds) required for a bus to travel between consecutive stops along its scheduled route ($i \to i+1$):
$$\text{travel\_time\_sec} = \text{next\_arr\_sec} - \text{arr\_sec}$$

### Leakage-Free Schema
- **Candidate Features Evaluated**: 32 clean numeric features spanning spatial geometry (Haversine distance, bearings, coordinates, center offsets), temporal cycles (diurnal harmonics, hour, peak indicators), network graph centrality (stop degrees, hub flags), and route topological metrics (tortuosity, sequence ratio, remaining stops).
- **Explicit Exclusions**:
  - `travel_time_sec`: Excluded as input (target variable $y$).
  - `next_arr_sec`: Excluded (future lookahead information).
  - `route_target_enc_time`: Excluded (identified in audit as containing in-sample training leakage and collapsing to an uninformative constant on unseen validation/test routes).
  - Arbitrary identifiers (`route_id`, `trip_id`, `stop_name`) to prevent memorization.

---

## 2. Model Candidates & Validation Benchmark

Five candidate models were trained strictly on the training partition (**77,069 segments**, 1,921 distinct routes) and evaluated on the route-disjoint validation partition (**9,994 segments**, 241 distinct routes):

| Model Architecture | Train Time (s) | Val Pred Time (s) | Val MAE (s) | Val RMSE (s) | Val $R^2$ | Val MedAE (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **12.19** | **0.085** | **0.50** | **5.47** | **0.99928** | **0.26** |
| **HistGradientBoosting** | 0.86 | 0.016 | 3.84 | 76.53 | 0.85903 | 0.43 |
| **XGBoost Regressor** | 3.51 | 0.017 | 4.78 | 67.89 | 0.88909 | 0.52 |
| **LightGBM Regressor** | 0.88 | 0.034 | 5.08 | 70.48 | 0.88045 | 0.82 |
| **DummyRegressor (Mean Baseline)** | 0.04 | 0.000 | 67.36 | 203.86 | -0.00024 | 50.41 |

### Champion Model Selection: **Random Forest**
- **Validation Winner**: Random Forest achieved the lowest validation error by a substantial margin (**0.50 seconds MAE**, compared to 3.84s for HistGradientBoosting and 67.36s for the Mean Baseline).
- **Physical Justification**: Random Forest effectively captures the piecewise continuous, rounded relationship between distance and scheduled time ($\text{round}(240 \times d)$) with high-capacity trees without gradient overshoot on extreme tail delay cases.

---

## 3. Final Evaluation on the Completely Unseen Test Set

The champion **Random Forest** model was evaluated **once** on the completely unseen test partition (**9,192 segments**, 241 routes with zero route or trip overlap with training data):

| Metric | Dummy Mean Baseline | Random Forest (Final Model) | Performance Gain |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 63.81 s | **0.36 s** | **99.4% error reduction** |
| **Root Mean Squared Error (RMSE)** | 92.28 s | **1.04 s** | **98.9% error reduction** |
| **Coefficient of Determination ($R^2$)** | -0.00578 | **0.99987** | Explains **99.99%** of variance |
| **Median Absolute Error (MedAE)** | 48.41 s | **0.27 s** | Median error $< 0.3$ seconds |
| **Mean Absolute % Error (MAPE)** | 49.32% | **0.39%** | Sub-1% relative error |
| **Inference Latency** | — | **0.0060 ms / sample** | Real-time capable ($> 160,000$ predictions/sec) |

---

## 4. Interpretation of Key Metrics

1. **Mean Absolute Error (0.36 seconds)**:
   On average, the predicted segment travel time deviates by less than four-tenths of a second from scheduled time across unseen transit routes.
2. **Root Mean Squared Error (1.04 seconds)**:
   Heavily penalizes rare large prediction errors. The near-1-second RMSE confirms that severe multi-minute prediction errors are nonexistent in standard segment transitions.
3. **Median Absolute Error (0.27 seconds)**:
   For 50% of all transit segments, the predicted duration is accurate to within a quarter of a second.
4. **$R^2$ Score (0.99987)**:
   The model accounts for virtually all structural variance in transit duration across diverse spatial coordinates and route lengths.

---

## 5. Feature Importance Analysis

The top 10 most influential features identified by Random Forest:

| Rank | Feature | Importance Share | Operational Relevance |
| :---: | :--- | :---: | :--- |
| 1 | `segment_dist_km` | 99.91% | Physical road segment distance directly governs baseline running time |
| 2 | `remaining_stops` | 0.0097% | Downstream position along the trip itinerary |
| 3 | `stop_sequence` | 0.0096% | Accumulated transit progress from starting terminal |
| 4 | `route_tortuosity` | 0.0085% | Geometric winding vs straight arterial path |
| 5 | `next_stop_lat` | 0.0082% | Destination spatial latitude coordinate |
| 6 | `corridor_degree_sum`| 0.0076% | Intersecting route density at origin and destination stops |
| 7 | `total_route_distance_km` | 0.0053% | Overall corridor trip scale |
| 8 | `dist_from_delhi_center_km` | 0.0044% | Radial urban density gradient (inner ring vs periphery) |
| 9 | `dest_dist_from_delhi_center_km` | 0.0040% | Destination radial distance to Connaught Place core |
| 10 | `sequence_ratio` | 0.0040% | Fractional journey completion |

---

## 6. Generated Visualizations

The following evaluation figures have been saved to [`outputs/figures/`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/):
1. **Actual vs Predicted Travel Time**: [`outputs/figures/actual_vs_predicted_travel_time.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/actual_vs_predicted_travel_time.png)  
   Demonstrates tight alignment along the $y = x$ diagonal across the entire distribution of test segment transit times.
2. **Residual Error Distribution**: [`outputs/figures/residual_error_distribution.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/residual_error_distribution.png)  
   Displays a symmetrical, zero-centered Gaussian error distribution with negligible bias ($\text{mean error} \approx 0.03\text{ s}$).
3. **Feature Importance Plot**: [`outputs/figures/feature_importance.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/feature_importance.png)  
   Ranks the relative predictive contribution of spatial, sequential, and network topology features.

---

## 7. How Predicted Travel Time Feeds Subsequent Bus Scheduling (Module 5)

This module produces segment-level travel time estimates. In Module 5 (Automated Scheduling & Dispatch Optimization), these predictions will serve as the input parameter matrix for mathematical dispatch models:

1. **Trip Runtime Aggregation**:
   For any route with $M$ stops dispatched at time $T_0$, total trip duration is dynamically calculated as:
   $$\widehat{\text{Trip Duration}} = \sum_{k=1}^{M-1} \widehat{\text{travel\_time}}_{k, k+1}(t_k)$$
2. **Turnaround & Cycle Time Formulation**:
   Round-trip cycle time ($\text{Cycle} = \text{Outbound Duration} + \text{Inbound Duration} + \text{Layover Buffer}$) determines vehicle return availability at the depot.
3. **Optimal Fleet Sizing & Headway Control**:
   Given target dispatch headways ($H$), required fleet size is computed as $N = \lceil \text{Cycle} / H \rceil$. Dynamic travel times allow dispatchers to allocate more buses during peak traffic and fewer during off-peak windows.
4. **Layover Buffer Injection**:
   Pinpoints corridors with high running time variance to inject recovery time, preventing bus bunching.
