# Gridlock: AI-Driven Illegal-Parking Intelligence System

Gridlock is a predictive parking intelligence platform and dashboard designed to assist municipal authorities (such as the Bengaluru Traffic Authority) in identifying and predicting illegal parking hotspots. By utilizing time-decay weighting, spatial grid clustering, and trend detection, Gridlock visualizes historical density maps and forecasts violation hotspots.

---

## Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed.

### Setup Instructions

1. **Clone the repository** (or navigate to the workspace directory):

   ```bash
   git clone https://github.com/Thushar1108/WildPark_CN.git
   ```

   ```bash
   cd Gridlock
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preprocessing Pipeline

Before running the web application, you must execute the data pipeline to load, clean, engineer features, and calculate baseline impact scores.


### How to Preprocess the Data:
To run the preprocessing script and generate the structured dataset (`data/processed_data.csv`), execute:
```bash
python main.py
```

#### Verbose Logging Mode
If you wish to see detailed debugging logs and execution trace metrics, run the pipeline with the `--log-level` parameter:
```bash
python main.py --log-level DEBUG
```

#### Pipeline Stages:
1. **Load**: Imports the raw spatial data from `data/data_grid.csv` (cleaning missing fields and dropping admin identifiers).
2. **Preprocess**: Cleans coordinates, parses timestamps, and structures categorical attributes.
3. **Feature Engineering**: Calculates spatial grid boundaries (down to 11m resolution), computes peak-hour traffic overlap, maps vehicle and violation weights, and generates the final **Violation Impact Score**.

---

## Running the Web Dashboard

Once `data/processed_data.csv` has been generated, launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to view the interface.

---

## Key Features

- **Historical Hotspot Heatmap**: Highly localized, high-contrast Folium heatmap displaying violation density based on historical impact scores.
- **Predictive Mode (Forecast)**:
  - Toggle forecast mode to query future dates/time ranges.
  - Predicts violation hotspots for a selected day of the week and hour window (between 1 and 4 hours).
  - Implements a **Time-Decay model** (2.0x weight for records in the last 7 days of the dataset).
- **Patrol Routing**:
   - The dashboard features an automated patrol routing engine that assigns high-priority violation hotspots to available patrol units. It is designed for maximum resilience and smart resource distribution.
   - divide the top-N grid cells across patrol units using a greedy round-robin assignment ranked by impact_score (highest to lowest)
- **Trend Detection**: Compares recent 7-day averages against a prior 21-day baseline to display trend directions (`▲ Increasing`, `— Stable`, `▼ Decreasing`).
- **Confidence Scoring**: Categorizes predictions into `High`, `Medium`, or `Low` confidence levels based on historical sample size.
- **Interactive KPI Cards**: Custom HTML cards styled in a premium green aesthetic, featuring smooth hover expansion effects.
