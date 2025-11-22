import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Note: These imports are required to run the original logic,
# but the model file ('rockfall_lstm.h5') is assumed to be missing in this environment,
# triggering the simulated prediction fallback.
try:
    from tensorflow.keras.models import load_model
except ImportError:
    st.warning(
        "TensorFlow/Keras not found. The application will use simulated predictions instead of loading an actual LSTM model.")

    def load_model(path):
        raise ImportError("Keras model loading is skipped.")

try:
    # Used for automatic page refresh to simulate real-time data
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.warning("`streamlit_autorefresh` not installed. Time step will not advance automatically.")

    def st_autorefresh(**kwargs):
        pass  # Placeholder function

# --- Configuration ---
SEQ_LENGTH = 10
AUTOREFRESH_INTERVAL_MS = 2000  # 2 seconds (customizable in sidebar)

# Vibrant Color Palette
COLORS = {
    'vibration': '#FF6B6B',  # Vibrant Red-Orange
    'strain': '#4ECDC4',     # Teal
    'rainfall': '#45B7D1',   # Sky Blue
    'risk': '#FECA57',       # Sunny Yellow
    'threshold': '#96CEB4',  # Mint Green
    'high_risk': '#FF4757',  # Bright Red
    'low_risk': '#2ED573',   # Spring Green
    'background': '#F8F9FA', # Light Gray
}

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(
    page_title="🌈 AI Rockfall Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🪨"
)

# Custom CSS for colorful styling
st.markdown("""
    <style>
    .main {background-color: #F8F9FA;}
    .stMetric {background: linear-gradient(90deg, #4ECDC4, #45B7D1);}
    .stAlert {border-radius: 10px;}
    h1 {color: #2E86AB !important;}
    h2 {color: #A23B72 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("🪨🌈 Vibrant Real-Time Rockfall Risk Dashboard")
st.markdown(
    "**Monitor rockfall risks with colorful, interactive visualizations!** "
    "Simulated sensor data (vibration, strain, rainfall) updates every few seconds. "
    "Toggle controls to customize your view. 🚀")

# -------------------------------
# Initialize session state
# -------------------------------
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'predictions_ready' not in st.session_state:
    st.session_state.predictions_ready = False
if 'simulation_paused' not in st.session_state:
    st.session_state.simulation_paused = False

# -------------------------------
# Generate simulated data (cached)
# -------------------------------
@st.cache_data(show_spinner="🎨 Generating colorful simulated data...")
def generate_simulated_data(n_samples=500):
    """Generates synthetic sensor data and scales it for model input."""
    np.random.seed(42)
    time_axis = np.linspace(0, n_samples / 10, n_samples)

    # Simulated Sensor Data Generation (with more variation for colorfulness)
    vibration = np.sin(2 * np.pi * 2 * time_axis) * (1 + 0.5 * time_axis / n_samples) + np.random.normal(0, 0.2, n_samples)
    strain = 0.001 * time_axis + np.random.normal(0, 0.0005, n_samples)
    pore_pressure = 5 + 0.5 * np.sin(0.2 * time_axis) + 0.1 * time_axis / n_samples + np.random.normal(0, 0.1, n_samples)
    displacement = np.cumsum(np.random.normal(0.005, 0.01, n_samples))
    rainfall = np.maximum(0, 10 * np.sin(2 * np.pi * 0.5 * time_axis) + np.random.normal(0, 2, n_samples))
    ndvi = 0.8 - 0.0005 * time_axis + np.random.normal(0, 0.01, n_samples)
    change_in_ndvi = np.diff(ndvi, prepend=ndvi[0])
    blast_vibration = np.zeros(n_samples)
    # Simulate a few blast events
    blast_indices = np.random.choice(n_samples, size=min(10, n_samples // 50), replace=False)
    blast_vibration[blast_indices] = np.random.uniform(1, 3, len(blast_indices))
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

    sim_data = pd.DataFrame({
        'Timestamp': timestamps,
        'Blast_Vibration': blast_vibration,
        'Change_in_NDVI': change_in_ndvi,
        'Date': timestamps.astype(int) / 10 ** 9,
        'NDVI': ndvi,
        'Rainfall': rainfall,
        'Vibration': vibration,
        'Strain': strain,
        'Pore_Pressure': pore_pressure,
        'Displacement': displacement
    })

    feature_cols = [col for col in sim_data.columns if col not in ['Timestamp', 'Date']]  # Exclude Timestamp/Date
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(sim_data[feature_cols])

    return sim_data, scaled_features, feature_cols

# -------------------------------
# Sidebar for controls (Enhanced with more options)
# -------------------------------
st.sidebar.header("⚙️ 🎨 Dashboard Controls")

# Data parameters
n_samples = st.sidebar.slider("📊 Number of Samples (History)", 100, 2000, 500, 50, key='n_samples')
risk_threshold = st.sidebar.slider("⚠️ Risk Threshold", 0.1, 1.0, 0.5, 0.05, key='risk_threshold')
show_all_features = st.sidebar.checkbox("🔍 Show All Sensor Features", value=False, key='show_all_features')
model_path = st.sidebar.text_input("🧠 LSTM Model Path", value="rockfall_lstm.h5", key='model_path')

# New: Simulation controls
st.sidebar.subheader("⏱️ Simulation")
autorefresh_interval = st.sidebar.slider("Refresh Interval (ms)", 1000, 5000, AUTOREFRESH_INTERVAL_MS, 500)
pause_sim = st.sidebar.checkbox("⏸️ Pause Simulation", value=False, key='pause_sim')
if st.sidebar.button("▶️ Play/Pause", key='play_pause'):
    st.session_state.simulation_paused = not st.session_state.simulation_paused

# New: Chart customization
st.sidebar.subheader("🎨 Chart Colors")
use_vibrant_colors = st.sidebar.checkbox("Use Vibrant Colors", value=True, key='vibrant_colors')
selected_sensors = st.sidebar.multiselect("Select Sensors to Plot",
                                          ['Vibration', 'Strain', 'Rainfall', 'Displacement', 'Pore_Pressure'],
                                          default=['Vibration', 'Strain', 'Rainfall'], key='sensors')

# Load Data based on N_samples
sim_data, scaled_np, feature_cols = generate_simulated_data(n_samples)
N_SAMPLES = n_samples

# Model loading logic (unchanged, but with colorful feedback)
if st.sidebar.button("🧠 Load LSTM Model"):
    if os.path.exists(model_path):
        try:
            st.session_state.lstm_model = load_model(model_path)
            st.session_state.predictions_ready = False
            st.session_state.current_step = 0
            st.sidebar.success("✅ Model loaded! 🌟")
            st.info("🔄 Recomputing predictions...")
        except Exception as e:
            st.session_state.lstm_model = None
            st.session_state.predictions_ready = False
            st.sidebar.error(f"❌ Load failed: {e}")
    else:
        st.session_state.lstm_model = None
        st.session_state.predictions_ready = False
        st.sidebar.warning("⚠️ Model not found. Using simulations! 🎲")

# Reset button
if st.sidebar.button("🔄 Reset & Refresh", type='primary'):
    st.session_state.current_step = 0
    st.session_state.predictions_ready = False
    st.session_state.simulation_paused = False
    st.cache_data.clear()
    st.rerun()

# -------------------------------
# Precompute predictions (or use simulation) - Enhanced fallback
# -------------------------------
if not st.session_state.predictions_ready:
    num_features = len(feature_cols)
    num_predictions = max(0, N_SAMPLES - SEQ_LENGTH + 1)

    if st.session_state.lstm_model is not None and num_predictions > 0:
        st.info("🔄 Computing predictions with model... (Colorful vibes incoming!)")
        try:
            sequences = np.array([scaled_np[i - SEQ_LENGTH + 1: i + 1] for i in range(SEQ_LENGTH - 1, N_SAMPLES)])
            model_input_shape = st.session_state.lstm_model.input_shape
            if len(model_input_shape) != 3 or model_input_shape[1] != SEQ_LENGTH or model_input_shape[2] != num_features:
                raise ValueError(f"Shape mismatch: Expected (None, {SEQ_LENGTH}, {num_features})")

            with st.spinner("Running inference..."):
                batch_preds = st.session_state.lstm_model.predict(sequences, verbose=0, batch_size=32)
                predictions = batch_preds.flatten().tolist()

            pad_length = SEQ_LENGTH - 1
            st.session_state.predictions = [0.0] * pad_length + predictions
            st.session_state.predictions_ready = True
            st.success("✅ Predictions ready! 🎉")

        except Exception as e:
            st.error(f"⚠️ Model error: {e}. Using colorful simulations!")
            st.session_state.lstm_model = None
            st.session_state.predictions = [0.5] * N_SAMPLES
            st.session_state.predictions_ready = True

    # FALLBACK: Enhanced simulated predictions with more color-influencing logic
    if st.session_state.lstm_model is None:
        st.info("🎨 Generating vibrant simulated predictions! (No model needed)")

        try:
            vib_idx = feature_cols.index('Vibration')
            strain_idx = feature_cols.index('Strain')
            rainfall_idx = feature_cols.index('Rainfall')
        except ValueError:
            predictions = np.clip(np.random.normal(0.5, 0.1, N_SAMPLES), 0, 1).tolist()
        else:
            vib_norm = scaled_np[:, vib_idx]
            strain_norm = scaled_np[:, strain_idx]
            rain_norm = scaled_np[:, rainfall_idx]

            # Enhanced risk: Add sinusoidal waves for more dynamic/colorful patterns
            base_risk = 0.2 + 0.3 * vib_norm + 0.3 * rain_norm + 0.2 * strain_norm + 0.1 * np.sin(0.1 * np.arange(N_SAMPLES))
            noise = np.random.normal(0, 0.05, N_SAMPLES)
            predictions = np.clip(base_risk + noise, 0, 1).tolist()

        st.session_state.predictions = predictions
        st.session_state.predictions_ready = True
        st.success("✅ Simulations ready! Let's make it colorful! 🌈")

    if st.session_state.predictions_ready:
        st.rerun()

# --- Check if data/predictions are ready ---
if not st.session_state.predictions_ready:
    st.stop()

# -------------------------------
# Auto-advancing Time Step (with Pause Control)
# -------------------------------
st.subheader("⏱️ 🎮 Simulation Time Control")

# Auto-refresh only if not paused
if not st.session_state.simulation_paused:
    st_autorefresh(interval=autorefresh_interval, limit=None, key="time_refresh")

# Update current step
if not st.session_state.simulation_paused:
    st.session_state.current_step = (st.session_state.current_step + 1) % N_SAMPLES
current_step = st.session_state.current_step

# Interactive slider (now enabled for manual control if paused)
disabled = st.session_state.simulation_paused
st.slider(
    "Current Time Step",
    min_value=0,
    max_value=N_SAMPLES - 1,
    value=current_step,
    step=1,
    disabled=disabled,
    help="Drag to navigate (unpause for auto-advance)."
)

progress = st.progress(current_step / (N_SAMPLES - 1))
st.write(f"**Progress:** {current_step} / {N_SAMPLES - 1} | **Time:** {sim_data['Timestamp'].iloc[current_step]} | "
         f"**Status:** {'⏸️ Paused' if st.session_state.simulation_paused else '▶️ Running'}")

# -------------------------------
# Metrics (Colorful with Icons)
# -------------------------------
st.subheader("📊 Current Readings")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔊 Vibration (G)", f"{sim_data['Vibration'].iloc[current_step]:.3f}",
              delta=f"{sim_data['Vibration'].iloc[current_step] - sim_data['Vibration'].iloc[current_step-1] if current_step > 0 else 0:+.3f}")

with col2:
    st.metric("🧬 Strain (µε)", f"{sim_data['Strain'].iloc[current_step]:.3f}",
              delta=f"{sim_data['Strain'].iloc[current_step] - sim_data['Strain'].iloc[current_step-1] if current_step > 0 else 0:+.3f}")

with col3:
    st.metric("🌧️ Rainfall (mm)", f"{sim_data['Rainfall'].iloc[current_step]:.2f}",
              delta=f"{sim_data['Rainfall'].iloc[current_step] - sim_data['Rainfall'].iloc[current_step-1] if current_step > 0 else 0:+.2f}")

with col4:
    pred_prob = st.session_state.predictions[current_step]
    prev_prob = st.session_state.predictions[current_step - 1] if current_step > 0 else st.session_state.predictions[-1]
    delta = pred_prob - prev_prob
    st.metric("🪨 Rockfall Risk", f"{pred_prob:.3f}", delta=f"{delta:+.3f}", delta_color="inverse")

# -------------------------------
# Risk Alert (Colorful with Background)
# -------------------------------
st.subheader("🚨 Risk Alert")
pred_prob = st.session_state.predictions[current_step]
if pred_prob >= risk_threshold:
    st.error(
        f"**🔴 HIGH ROCKFALL RISK!** Probability: **{pred_prob:.3f}** (Threshold: {risk_threshold:.3f}) "
        f"– Evacuate area immediately! ⚠️", icon="🚨")
else:
    st.success(
        f"**🟢 RISK NORMAL.** Probability: **{pred_prob:.3f}** (Threshold: {risk_threshold:.3f}) "
        f"– All clear! 👍", icon="✅")

# -------------------------------
# Charts (Enhanced with Subplots and Vibrant Colors)
# -------------------------------
history_steps = list(range(current_step + 1))
history_data = sim_data.iloc[history_steps].copy()
pred_history = [st.session_state.predictions[i] for i in history_steps]

st.subheader("📈 Interactive Sensor Data & Predictions History")
col_chart1, col_chart2 = st.columns(2)

# --- Chart 1: Multi-Sensor Plot (Subplots for Vibrancy) ---
with col_chart1:
    fig1 = make_subplots(specs=[[{"secondary_y": False}]], subplot_titles=["Vibration & Strain Over Time"])

    # Add traces for selected sensors
    for sensor in selected_sensors:
        if sensor in history_data.columns:
            color = COLORS.get(sensor.lower(), '#7D5FFF')  # Default purple
            fig1.add_trace(
                go.Scatter(
                    x=history_data['Timestamp'],
                    y=history_data[sensor],
                    mode='lines+markers',
                    name=sensor,
                    line=dict(color=color if use_vibrant_colors else 'gray', width=3),
                    marker=dict(size=4)
                ),
                secondary_y=False
            )

    fig1.update_layout(
        title="🔥 Sensor Trends",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400,
        hovermode='x unified',
        template="plotly_white" if use_vibrant_colors else "plotly_dark",
        showlegend=True
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

# --- Chart 2: Predicted Rockfall Risk (Vibrant with Highlights) ---
with col_chart2:
    # Color each marker by whether the associated risk is above the threshold
    marker_colors = [COLORS['high_risk'] if p >= risk_threshold else COLORS['low_risk'] for p in pred_history]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=history_data['Timestamp'],
        y=pred_history,
        mode='lines+markers',
        name='Risk Probability',
        line=dict(color=COLORS['risk'] if use_vibrant_colors else 'gray', width=3),
        marker=dict(size=6, color=marker_colors)
    ))

    # threshold line
    fig2.add_hline(y=risk_threshold, line=dict(color=COLORS['threshold'], dash='dash'), annotation_text="Threshold", annotation_position="top left")

    # latest value annotation
    fig2.add_annotation(
        x=history_data['Timestamp'].iloc[-1],
        y=pred_history[-1],
        text=f"Latest: {pred_history[-1]:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40
    )

    fig2.update_layout(
        title="🛡️ Predicted Rockfall Risk Over Time",
        xaxis_title="Time",
        yaxis_title="Risk Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        hovermode='x unified',
        template="plotly_white" if use_vibrant_colors else "plotly_dark",
        showlegend=False
    )
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')

    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

# (You can continue your dashboard below — this fixes the plotting error and safely finishes the risk chart.)
