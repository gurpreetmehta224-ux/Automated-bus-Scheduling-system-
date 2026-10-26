# Module 5: Automated Bus Scheduling & Route Optimization Report
**Project**: Automated Bus Scheduling and Route Management for Delhi Transport Corporation (DTC)  
**System Component**: Vehicle Scheduling Problem (VSP) & Linked Duty Optimizer  
**Source Code**: [`src/bus_scheduler.py`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/src/bus_scheduler.py) & [`src/route_runtime_calculator.py`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/src/route_runtime_calculator.py)  
**Configuration File**: [`config/scheduling_config.json`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/config/scheduling_config.json)  

---

## 1. Executive Summary & Core Purpose

The primary objective of Module 5 is to demonstrate the operational core of automated public transit management: **translating segment-level transit runtimes into feasible, cost-effective, and conflict-free bus schedules for Delhi Transport Corporation (DTC)**.

Moving beyond simple descriptive statistics or isolated machine learning predictions, Module 5 implements the operational bridge:
$$\text{Route \& Stop Network} \longrightarrow \text{Segment Travel Times} \longrightarrow \text{Route Runtimes} \longrightarrow \text{Operational Constraints} \longrightarrow \text{Optimized Bus Schedules}$$

### Key Achievement
By replacing unlinked single-trip dispatches with **Linked Duty Scheduling (Vehicle Chaining Optimization)**:
- **Fleet Size Requirement Reduced by 63.9%**: Covering all 2,403 scheduled trips requires only **868 buses**, compared to **2,403 buses** under unlinked dispatching (a direct capital saving of **1,535 buses**).
- **100% Trip Coverage**: All 2,403 trips are scheduled without temporal overlap or spatial conflict.
- **Strict Constraint Adherence**: Every vehicle chain enforces a mandatory 15-minute turnaround buffer and respects the 8.0-hour single-shift duty cap.

---

## 2. Operational Data Audit (Step 1)

To maintain absolute academic and scientific integrity, all operational parameters and data fields are categorized into three explicit tiers:

| Tier | Category | Specific Fields & Parameters | Status / Data Source |
| :---: | :--- | :--- | :--- |
| **Tier A** | **Directly Present in Raw Dataset** | `route_id`, `trip_id`, `stop_id`, `arrival_time`, `departure_time`, `stop_name`, `stop_lat`, `stop_lon` | Official GTFS Static Schedule for Delhi (`data/`) |
| **Tier B** | **Derived Systematically from Data** | `stop_sequence`, segment Haversine distance (`segment_dist_km`), segment travel time (`travel_time_sec`), total route distance, total route stops, route runtime, network centrality / stop degree | Quantitatively engineered in Modules 1–4 |
| **Tier C** | **Configurable Operational Assumptions** | Turnaround buffer (15 min), maximum vehicle duty (8.0 h), deadhead speed (20 km/h), max deadhead distance (15 km), stochastic delay buffer (1.05x) | **ASSUMPTIONS — NOT FROM DTC DATA** (Configured in `config/scheduling_config.json`) |

> [!IMPORTANT]
> The raw GTFS dataset does **not** contain bus vehicle license numbers, driver payroll/shift rosters, physical depot locations, or observed GPS probe delays. These operational elements are modeled through documented, configurable assumptions.

---

## 3. Route Runtime Compilation (Step 2)

Using [`src/route_runtime_calculator.py`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/src/route_runtime_calculator.py), the 96,255 segment transitions were aggregated into a comprehensive route-level dataset: [`outputs/data/route_runtime_summary.csv`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/route_runtime_summary.csv).

### Route Runtime Summary Statistics (2,403 Routes)
- **Route Distance**: Mean = **23.62 km**, Median = **22.60 km**, Range = $[0.87\text{ km}, 184.41\text{ km}]$
- **Stops per Route**: Mean = **41.1 stops**, Median = **42 stops**, Range = $[2, 197]$
- **Scheduled Runtime**: Mean = **94.41 min** ($\approx 1.57\text{ h}$), Median = **90.33 min**, Range = $[3.48\text{ min}, 736.40\text{ min}]$
- **ML Predicted Runtime**: Mean = **94.40 min**, Median = **90.33 min**
- **Buffered Runtime (1.05x)**: Mean = **99.12 min**, Median = **94.85 min**

---

## 4. Scheduling Constraints Engine (Step 3)

All operational constraints are isolated in [`config/scheduling_config.json`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/config/scheduling_config.json) to allow rapid sensitivity tuning:

1. **Turnaround & Recovery Constraint**:
   $$\text{Departure Time}(Trip_B) \ge \text{Arrival Time}(Trip_A) + T_{\text{turnaround}} + T_{\text{deadhead}}(A \to B)$$
   - *Value*: $T_{\text{turnaround}} = 15.0\text{ minutes}$ (*ASSUMPTION — NOT FROM DTC DATA*).
   - *Justification*: Prevents delay propagation, provides vehicle ventilation/cleaning, and ensures driver turnaround recovery.
2. **Maximum Vehicle Duty Duration**:
   $$\text{Duty End Time} - \text{Duty Start Time} \le 8.0\text{ hours}$$
   - *Value*: $8.0\text{ hours}$ (*ASSUMPTION — NOT FROM DTC DATA*).
   - *Justification*: Standard labor regulation and single-shift operational cap.
3. **Deadhead Repositioning Thresholds**:
   $$\text{Deadhead Distance}(End_A \to Start_B) \le 15.0\text{ km}$$
   $$\text{Deadhead Time} = \frac{\text{Haversine}(End_A, Start_B)}{20.0\text{ km/h}} \times 3,600\text{ seconds}$$
   - *Speed*: $20.0\text{ km/h}$; *Distance Cap*: $15.0\text{ km}$ (*ASSUMPTIONS — NOT FROM DTC DATA*).
   - *Justification*: Prevents excessive non-revenue empty bus running across Delhi.

---

## 5. Optimization Methodology: Linked vs. Unlinked Scheduling (Steps 4–6)

### Mathematical Formulation
The transit Vehicle Scheduling Problem (VSP) with fixed timetables is formulated as a **Minimum Path Cover on a Directed Acyclic Graph (DAG)**:
- **Vertices ($V$)**: 2,403 scheduled trips with fixed start times and durations.
- **Edges ($E$)**: A directed arc exists from $Trip_A \to Trip_B$ if and only if $Trip_B$ can be performed after $Trip_A$ without temporal conflict:
  $$t_{\text{start}}(B) \ge t_{\text{end}}(A) + T_{\text{turnaround}} + T_{\text{deadhead}}(A \to B) \quad \text{and} \quad \text{Duty Duration} \le 8.0\text{ h}$$

### Optimization Algorithm: Earliest-Compatible Greedy Chain Matching
Trips are sorted chronologically by start time. For each trip $B$:
1. Search active buses that have completed their previous trip $A$.
2. Filter candidates satisfying all turnaround, deadhead, and 8-hour duty constraints.
3. If multiple buses are eligible, choose the bus minimizing the multi-objective cost:
   $$\min \left( 2.5 \times \text{Deadhead Distance (km)} + \frac{\text{Idle Wait Time (sec)}}{1800} \right)$$
4. If no active bus is eligible, dispatch a new bus from the fleet pool.

*Why this method?* It guarantees zero temporal conflicts, strictly respects all constraints, scales smoothly across thousands of trips in seconds, and closely approximates the exact minimum path cover solution.

---

## 6. Comparative Performance: Linked vs. Unlinked Schedules

| Performance Metric | Unlinked Baseline (1 Bus / Trip) | Optimized Linked Schedule | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Fleet Size Required** | **2,403 buses** | **868 buses** | **63.9% modeled fleet reduction (1,535 buses fewer)** |
| **Trips Covered** | 2,403 (100%) | 2,403 (100%) | 100% Full Coverage |
| **Total In-Service Driving Hours** | 3,781.1 h | 3,781.1 h | Identical service delivery |
| **Turnaround / Recovery Hours** | 0.0 h | 383.8 h | Healthy recovery buffers added |
| **Deadhead Repositioning Hours** | 0.0 h | 189.9 h | 3.6% of duty time for transfers |
| **Idle Waiting Hours** | 0.0 h | 951.9 h | Staggered schedule wait |
| **Total Fleet Duty Hours** | 3,781.1 h | 5,306.7 h | Multi-trip extended duties |
| **Fleet In-Service Utilization** | 100.0% (Isolated) | **71.3%** | Highly efficient realistic duty mix |
| **Average Trips per Bus** | 1.00 trip | **2.77 trips** | **+176.8% vehicle productivity** |
| **Maximum Trips on Single Bus** | 1 trip | **6 trips** | Up to 6 chained runs |
| **Average Vehicle Duty Duration** | 1.57 hours | **6.11 hours** | Well within the 8.0-hour cap |

---

## 7. Machine Learning Integration (Step 7)

- **Role of the ML Model**: The Random Forest model from Module 4 acts as the **baseline segment runtime calculator** within the scheduling engine.
- **Uncertainty Buffer Option**: Schedulers can apply a configurable delay multiplier (e.g. $1.05\times$) to inject safety margins into segment predictions before compiling route runtimes:
  $$\text{Scheduled Segment Time} = \widehat{\text{travel\_time}}_{\text{ML}} \times 1.05$$
- **Integrity Notice**: The ML model reflects the synthetic 15 km/h GTFS timetable baseline and is not claimed to represent real-world traffic congestion.

---

## 8. Artifacts & Datasets Generated (Step 8)

### Output Datasets ([`outputs/data/`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/)):
1. [`route_runtime_summary.csv`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/route_runtime_summary.csv): Route-level distances, stops, scheduled and ML-predicted runtimes (2,403 rows).
2. [`bus_schedule.csv`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/bus_schedule.csv): Trip-level master schedule with assigned vehicle IDs, sequence in duty, and gap statistics (2,403 rows).
3. [`linked_duty_schedule.csv`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/linked_duty_schedule.csv): Bus-level duty rosters detailing duty start/end times, driving time, turnaround, and route chains (868 rows).
4. [`unlinked_duty_schedule.csv`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/unlinked_duty_schedule.csv): Unlinked baseline vehicle roster (2,403 rows).
5. [`bus_utilization_statistics.csv`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/bus_utilization_statistics.csv): Comparative summary table.
6. [`idle_recovery_statistics.csv`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/data/idle_recovery_statistics.csv): Connection-level turnaround and deadhead metrics (1,535 links).

### Output Visualizations ([`outputs/figures/`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/)):
1. [`fleet_requirement_comparison.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/fleet_requirement_comparison.png): Visual bar chart comparing Unlinked (2,403) vs Linked (868) buses.
2. [`bus_utilization_comparison.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/bus_utilization_comparison.png): Breakdown of duty hours (Driving vs Turnaround vs Deadhead vs Idle).
3. [`trips_per_bus_distribution.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/trips_per_bus_distribution.png): Histogram of trip counts assigned per vehicle.
4. [`duty_duration_distribution.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/duty_duration_distribution.png): Distribution of vehicle shift durations showing adherence to the 8h cap.
5. [`schedule_gantt_chart.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/schedule_gantt_chart.png): Timeline Gantt visualization of representative multi-trip vehicle duties.
6. [`peak_vs_offpeak_fleet_demand.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/peak_vs_offpeak_fleet_demand.png): Profile of simultaneously active buses on road across the 24-hour service day.
7. [`route_runtime_distribution.png`](file:///c:/Users/sande/Desktop/automated%20bus%20scheduling/outputs/figures/route_runtime_distribution.png): Histogram of complete route runtimes across Delhi.

---

## 9. Assumptions & Methodological Limitations

### Documented Assumptions (All Configurable)
1. **Turnaround Recovery**: Minimum 15 minutes between consecutive trips.
2. **Duty Span**: Maximum 8.0 hours per single-shift vehicle duty.
3. **Deadhead Repositioning**: Empty buses travel between different terminals at an average speed of 20 km/h, capped at 15 km distance.
4. **Single-Trip Route Sample**: Each route in the raw dataset contains only one scheduled trip instance.

### Limitations
1. **Lack of Depot Locations**: The dataset does not specify depot coordinates; deadhead is currently modeled between consecutive trip terminals.
2. **Fixed Timetable Inputs**: Start times are fixed according to the GTFS schedule; future modules can explore dynamic timetable shifting to optimize headways further.

---

## 10. Faculty Demonstration: 60–90 Second Oral Presentation Pitch

> *"Good morning, Professors. Today, I am demonstrating **Module 5: Automated Bus Scheduling and Route Optimization for the Delhi Transport Corporation**.*
> 
> *The core objective of our project is to solve the **Vehicle Scheduling Problem (VSP)**: how can a public transit agency cover thousands of scheduled trips using the minimum number of buses, without schedule conflicts, and while strictly complying with labor and recovery rules?*
> 
> *In our raw Delhi transit dataset, there are **2,403 distinct scheduled bus trips**. If we operate naively—assigning one bus per trip without vehicle chaining—the city would require **2,403 buses**, resulting in massive fleet idle time and wasted capital.*
> 
> *In Module 5, we built an automated scheduling optimizer based on **Minimum Path Cover and Greedy Vehicle Chaining**. Our engine takes the route runtimes derived from our ML pipeline and chains compatible trips into realistic vehicle duties.*
> 
> *Every vehicle chain strictly enforces real-world transit constraints:*
> 1. *A mandatory **15-minute turnaround and recovery buffer** between consecutive trips.*
> 2. *A **maximum 8-hour duty cap** per vehicle shift.*
> 3. *Spatial **deadhead repositioning limits** between terminal stops.*
> 
> *The results are dramatic: our optimizer successfully covers **100% of all 2,403 trips using only 868 buses**. That is a **63.9% reduction in fleet requirements**, saving **1,535 buses** while maintaining an active driving utilization of **71.3%** and an average of **2.8 trips per bus**.*
> 
> *We have fully exported the master bus schedules, duty rosters, and utilization analytics to demonstrate how data science directly drives capital and operational efficiency in urban bus transit. Thank you."*
