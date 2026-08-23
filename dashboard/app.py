"""
MLOps Sentiment Analysis Dashboard

Real-time monitoring of prediction distribution, drift detection,
model registry, and retraining controls.

Run: streamlit run dashboard/app.py
"""
import json
import os
import subprocess
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
DRIFT_HISTORY_PATH = os.getenv("DRIFT_HISTORY_PATH", "monitoring/drift_history.jsonl")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_COLORS = {
    "sadness": "#636EFA",
    "joy": "#FFA15A",
    "love": "#EF553B",
    "anger": "#AB63FA",
    "fear": "#00CC96",
    "surprise": "#19D3F3",
}

TRAINING_REFERENCE = {
    "sadness": 0.28,
    "joy": 0.28,
    "love": 0.08,
    "anger": 0.14,
    "fear": 0.12,
    "surprise": 0.10,
}

st.set_page_config(
    page_title="Sentiment MLOps Dashboard",
    page_icon="📊",
    layout="wide",
)

# ───────────────────── Custom CSS ─────────────────────
st.markdown("""
<style>
    .drift-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .drift-ok { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .drift-warn { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .drift-danger { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .scenario-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def api_get(path: str):
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def load_drift_history():
    entries = []
    api_data = api_get("/drift-history?limit=100")
    if api_data:
        return api_data
    if os.path.exists(DRIFT_HISTORY_PATH):
        with open(DRIFT_HISTORY_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries[-100:]


def get_mlflow_models():
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW_URL)
        models = []
        for rm in client.search_registered_models():
            for mv in client.search_model_versions(f"name='{rm.name}'"):
                run = client.get_run(mv.run_id)
                metrics = run.data.metrics
                models.append({
                    "Model": rm.name,
                    "Version": mv.version,
                    "Stage": mv.current_stage,
                    "Accuracy": metrics.get("accuracy", "N/A"),
                    "Macro F1": metrics.get("macro_f1", "N/A"),
                    "Created": mv.creation_timestamp,
                })
        return models
    except Exception as e:  # noqa: BLE001 - MLflow may be unreachable; degrade gracefully
        st.sidebar.warning(f"MLflow unavailable: {e}")
        return []


# ═══════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════
st.title("Sentiment Analysis -- MLOps Monitoring Dashboard")
st.markdown(
    "Real-time monitoring of X/Twitter post sentiment predictions, "
    "drift detection, and automated retraining pipeline."
)

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)
if auto_refresh:
    time.sleep(10)
    st.rerun()

# ═══════════════════════════════════════════════════════
#  SECTION 1: DRIFT STATUS
# ═══════════════════════════════════════════════════════
st.header("1. Drift Detection Status")

drift_data = api_get("/drift-status")

if drift_data and drift_data.get("drift_result"):
    dr = drift_data["drift_result"]
    psi = dr.get("psi", 0.0)
    threshold = dr.get("psi_threshold", 0.2)
    detected = dr.get("drift_detected", False)
    ts = dr.get("timestamp", "N/A")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PSI Score", f"{psi:.4f}", delta=f"threshold: {threshold}")
    col2.metric("Drift Detected", "YES" if detected else "NO")
    col3.metric("Samples Analyzed", dr.get("sample_count", 0))
    col4.metric("Last Check", ts[:19] if ts != "N/A" else "N/A")

    if detected:
        if psi > threshold * 2:
            st.markdown(
                '<div class="drift-alert drift-danger">'
                'CRITICAL DRIFT DETECTED -- Significant distribution shift observed. '
                'Scenario: Gen-Z hate-speech / social unrest surge detected in incoming '
                'posts. Anger and fear predictions dominate, deviating sharply from '
                'normal sentiment patterns. Immediate retraining recommended.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="drift-alert drift-warn">'
                'WARNING: Moderate drift detected. Prediction distribution is shifting '
                'from the training baseline. Monitor closely.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="drift-alert drift-ok">'
            'All clear -- No significant drift detected. Prediction distribution '
            'aligns with training data.</div>',
            unsafe_allow_html=True,
        )
else:
    st.info(
        "No drift data available yet. Run the drift detector or send predictions "
        "first, then click 'Run Drift Check' below."
    )

# ═══════════════════════════════════════════════════════
#  SECTION 2: DISTRIBUTION COMPARISON
# ═══════════════════════════════════════════════════════
st.header("2. Prediction Distribution vs Training Reference")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Training Reference")
    ref_df = pd.DataFrame({
        "Emotion": list(TRAINING_REFERENCE.keys()),
        "Proportion": list(TRAINING_REFERENCE.values()),
    })
    fig_ref = px.bar(
        ref_df, x="Emotion", y="Proportion",
        color="Emotion", color_discrete_map=LABEL_COLORS,
        title="Training Data Distribution",
    )
    fig_ref.update_layout(showlegend=False, yaxis_range=[0, 0.6])
    st.plotly_chart(fig_ref, use_container_width=True)

with col_chart2:
    st.subheader("Current Predictions (last 200)")
    if drift_data and drift_data.get("live_distribution"):
        live = drift_data["live_distribution"]
        live_df = pd.DataFrame({
            "Emotion": LABEL_NAMES,
            "Proportion": [live.get(l, 0.0) for l in LABEL_NAMES],
        })
        fig_live = px.bar(
            live_df, x="Emotion", y="Proportion",
            color="Emotion", color_discrete_map=LABEL_COLORS,
            title=f"Live Predictions (n={drift_data.get('total_predictions', 0)})",
        )
        fig_live.update_layout(showlegend=False, yaxis_range=[0, 0.6])
        st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.info("No predictions yet.")

# Side-by-side overlay
if drift_data and drift_data.get("live_distribution"):
    st.subheader("Overlay Comparison")
    live = drift_data["live_distribution"]
    overlay_df = pd.DataFrame({
        "Emotion": LABEL_NAMES * 2,
        "Proportion": (
            [TRAINING_REFERENCE.get(l, 0) for l in LABEL_NAMES]
            + [live.get(l, 0) for l in LABEL_NAMES]
        ),
        "Source": ["Training Reference"] * len(LABEL_NAMES) + ["Current Predictions"] * len(LABEL_NAMES),
    })
    fig_overlay = px.bar(
        overlay_df, x="Emotion", y="Proportion", color="Source",
        barmode="group", title="Distribution Comparison",
        color_discrete_map={
            "Training Reference": "#636EFA",
            "Current Predictions": "#EF553B",
        },
    )
    fig_overlay.update_layout(yaxis_range=[0, 0.7])
    st.plotly_chart(fig_overlay, use_container_width=True)

# ═══════════════════════════════════════════════════════
#  SECTION 3: DRIFT TREND
# ═══════════════════════════════════════════════════════
st.header("3. Drift Score Over Time")

history = load_drift_history()
if history:
    hist_df = pd.DataFrame(history)
    if "timestamp" in hist_df.columns and "psi" in hist_df.columns:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        threshold_val = history[0].get("psi_threshold", 0.2)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["psi"],
            mode="lines+markers", name="PSI Score",
            line={"color": "#EF553B", "width": 2},
        ))
        fig_trend.add_hline(
            y=threshold_val, line_dash="dash", line_color="red",
            annotation_text=f"Threshold ({threshold_val})",
        )
        fig_trend.update_layout(
            title="Population Stability Index (PSI) Over Time",
            xaxis_title="Time", yaxis_title="PSI",
            yaxis_range=[0, max(hist_df["psi"].max() * 1.3, 0.5)],
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Drift history format not recognized.")
else:
    st.info("No drift history available yet. Run drift checks to populate.")

# ═══════════════════════════════════════════════════════
#  SECTION 4: MODEL REGISTRY
# ═══════════════════════════════════════════════════════
st.header("4. MLflow Model Registry")
models = get_mlflow_models()
if models:
    models_df = pd.DataFrame(models)
    for col_name in ["Accuracy", "Macro F1"]:
        if col_name in models_df.columns:
            models_df[col_name] = models_df[col_name].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else x
            )
    st.dataframe(models_df, use_container_width=True, hide_index=True)
else:
    st.info("MLflow not reachable or no models registered.")

# ═══════════════════════════════════════════════════════
#  SECTION 5: ACTIONS PANEL
# ═══════════════════════════════════════════════════════
st.header("5. Demo Actions")

st.markdown(
    '<div class="scenario-box">'
    "<b>Drift Scenario Story:</b> Your sentiment analysis API monitors X/Twitter posts "
    "for emotion trends. During normal operation, emotions follow a balanced distribution "
    "(~28% sadness, ~28% joy, etc.). The <b>drift scenario</b> simulates a sudden surge "
    "of Gen-Z style posts dominated by <b>anger and fear</b> -- representing a trending "
    "controversy, social unrest, or potential insurgency discourse. This shifts the "
    "prediction distribution far from the training baseline, triggering drift detection "
    "and automated retraining.</div>",
    unsafe_allow_html=True,
)

col_a1, col_a2, col_a3, col_a4 = st.columns(4)

with col_a1:
    normal_count = st.number_input("Normal traffic count", 50, 500, 150, step=50)
    if st.button("Send Normal Traffic", type="secondary"):
        with st.spinner("Sending balanced emotion texts..."):
            result = subprocess.run(
                ["python", "scripts/simulate_drift.py",
                 "--mode", "normal", "--count", str(normal_count),
                 "--api-url", f"{API_URL}/predict", "--delay", "0.05"],
                capture_output=True, text=True, timeout=120, check=False,
            )
            if result.returncode == 0:
                st.success(f"Sent {normal_count} normal predictions")
            else:
                st.error(f"Failed: {result.stderr[:200]}")

with col_a2:
    drift_count = st.number_input("Drift traffic count", 50, 500, 200, step=50)
    if st.button("Send Drift Traffic (GenZ Surge)", type="primary"):
        with st.spinner("Sending anger/fear surge texts..."):
            result = subprocess.run(
                ["python", "scripts/simulate_drift.py",
                 "--mode", "drift", "--count", str(drift_count),
                 "--api-url", f"{API_URL}/predict", "--delay", "0.05"],
                capture_output=True, text=True, timeout=120, check=False,
            )
            if result.returncode == 0:
                st.success(f"Sent {drift_count} drift predictions")
            else:
                st.error(f"Failed: {result.stderr[:200]}")

with col_a3:
    if st.button("Run Drift Check"):
        with st.spinner("Running drift detector..."):
            result = subprocess.run(
                ["python", "monitoring/drift_detector.py"],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if result.returncode == 0:
                st.success("Drift check complete!")
                st.code(result.stdout)
            else:
                st.error(f"Failed: {result.stderr[:200]}")

with col_a4:
    if st.button("Trigger Retraining"):
        st.info(
            "Triggering retrain DAG via Airflow REST API..."
        )
        try:
            resp = requests.post(
                f"{AIRFLOW_URL}/api/v1/dags/retrain_dag/dagRuns",
                json={"conf": {}},
                auth=("admin", "admin"),
                timeout=10,
            )
            if resp.status_code in (200, 201):
                st.success("Retrain DAG triggered successfully!")
            else:
                st.warning(f"Airflow response: {resp.status_code} - {resp.text[:200]}")
        except requests.RequestException as e:
            st.warning(
                f"Could not reach Airflow ({e}). "
                "You can trigger manually from the Airflow UI at "
                f"{AIRFLOW_URL}"
            )

# ═══════════════════════════════════════════════════════
#  SECTION 6: RETRAINING LOG
# ═══════════════════════════════════════════════════════
st.header("6. Retraining History")
retrain_path = "monitoring/retrain_history.jsonl"
api_retrain_path = os.getenv("RETRAIN_HISTORY_PATH", retrain_path)
retrain_entries = []
for path in [api_retrain_path, retrain_path]:
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    retrain_entries.append(json.loads(line))
        break

if retrain_entries:
    rt_df = pd.DataFrame(retrain_entries)
    st.dataframe(rt_df, use_container_width=True, hide_index=True)
else:
    st.info("No retraining events recorded yet.")

# ───────────────────── Footer ─────────────────────
st.markdown("---")
st.caption(
    "Team 42 -- DA5402W MLOps Course End Project | "
    "Sentiment Analysis Pipeline with Drift Detection & Automated Retraining"
)
