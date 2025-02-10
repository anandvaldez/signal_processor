import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# Function to apply a low-pass filter
def lowpass_filter(signal, cutoff, fs, order=5):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    except Exception as e:
        st.error(f"Error in filtering: {e}")
        return signal

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

# Streamlit UI
st.title("📊 Signal Preprocessing & Comparison Tool")
st.write("Upload multiple signal files, apply preprocessing, and compare.")

# File upload (multiple files allowed)
uploaded_files = st.file_uploader("📁 Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = {}  # Dictionary to store dataframes
    raw_data = {}  # Dictionary to store raw data for comparison
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        if 'amplitude' not in df.columns:
            st.error(f"❌ {uploaded_file.name} is missing the 'amplitude' column.")
            continue
        df = df.dropna(subset=['amplitude'])  # Remove NaN values
        dfs[uploaded_file.name] = df  # Store dataframe
        raw_data[uploaded_file.name] = df.copy()  # Store raw data

    # Dataset selection
    selected_datasets = st.multiselect("📌 Select datasets to process", list(dfs.keys()), default=list(dfs.keys()))

    if selected_datasets:
        # Step 1: Preprocessing
        for name in selected_datasets:
            df = dfs[name]

            with st.expander(f"⚙️ Preprocessing: {name}"):
                cutoff = st.slider(f"Low-pass filter cutoff (Hz) for {name}:", 1, 100, 10, key=f"cutoff_{name}")
                fs = st.number_input(f"Sampling frequency (Hz) for {name}:", 1, 500, 100, key=f"fs_{name}")

                if st.button(f"Apply Filter for {name}"):
                    df['filtered'] = lowpass_filter(df['amplitude'], cutoff, fs)
                    st.session_state.processed_data[name] = df  # Save to session state
                    st.success(f"Filter applied to {name}!")

        # Step 2: Comparison
        with st.expander("📊 Compare Signals", expanded=True):
            # Options for comparison
            comparison_options = st.multiselect(
                "Select data to compare",
                ["Raw Data", "Processed Data"],
                default=["Raw Data", "Processed Data"]
            )

            fig = go.Figure()

            if "Raw Data" in comparison_options:
                for name in selected_datasets:
                    raw_df = raw_data[name]
                    fig.add_trace(go.Scatter(y=raw_df['amplitude'], mode='lines', name=f'Raw: {name}'))

            if "Processed Data" in comparison_options:
                for name in st.session_state.processed_data:
                    processed_df = st.session_state.processed_data[name]
                    fig.add_trace(go.Scatter(y=processed_df['filtered'], mode='lines', name=f'Processed: {name}'))

            fig.update_layout(
                title="Signal Comparison",
                xaxis_title="Index",
                yaxis_title="Amplitude",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download Processed Data
        with st.expander("⬇️ Download Processed Data"):
            for name in st.session_state.processed_data:
                csv = st.session_state.processed_data[name].to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {name}", data=csv, file_name=f"processed_{name}.csv", mime="text/csv")
