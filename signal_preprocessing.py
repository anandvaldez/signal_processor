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

# Streamlit UI
st.title("📊 Signal Preprocessing & Comparison Tool")
st.write("Upload one or multiple signal files, apply preprocessing, and compare them.")

# File upload (multiple files allowed)
uploaded_files = st.file_uploader("📁 Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = {}  # Dictionary to store dataframes
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        if 'amplitude' not in df.columns:
            st.error(f"❌ {uploaded_file.name} is missing the 'amplitude' column.")
            continue
        df = df.dropna(subset=['amplitude'])  # Remove NaN values
        dfs[uploaded_file.name] = df  # Store dataframe

    # Dataset selection
    selected_datasets = st.multiselect("📌 Select datasets to process", list(dfs.keys()), default=list(dfs.keys()))

    if selected_datasets:
        # Data Trimming UI
        with st.expander("✂️ Data Trimming"):
            dataset_to_trim = st.selectbox("Select dataset to trim", selected_datasets)
            df_trim = dfs[dataset_to_trim]

            # Trim range selection
            start_idx, end_idx = st.slider(
                "Select data range",
                0, len(df_trim) - 1, (0, len(df_trim) - 1)
            )

            # Trimmed data
            df_trim = df_trim.iloc[start_idx:end_idx]
            st.write("### Trimmed Data Preview")
            st.dataframe(df_trim.head())

            # Save trimmed dataset
            dfs[dataset_to_trim] = df_trim

        # Preprocessing UI
        with st.expander("⚙️ Preprocessing"):
            cutoff = st.slider("Low-pass filter cutoff frequency (Hz):", min_value=1, max_value=100, value=10)
            fs = st.number_input("Sampling frequency (Hz):", min_value=1, value=100)

            apply_filter = st.button("Apply Low-pass Filter")
            if apply_filter:
                for name in selected_datasets:
                    dfs[name]['filtered'] = lowpass_filter(dfs[name]['amplitude'], cutoff, fs)

        # Data Overlapping & Comparison
        with st.expander("📊 Compare Signals", expanded=True):
            fig = go.Figure()
            for name in selected_datasets:
                df = dfs[name]
                fig.add_trace(go.Scatter(y=df['amplitude'], mode='lines', name=f'Raw: {name}'))
                if 'filtered' in df.columns:
                    fig.add_trace(go.Scatter(y=df['filtered'], mode='lines', name=f'Filtered: {name}', line=dict(dash='dot')))

            fig.update_layout(
                title="Signal Comparison",
                xaxis_title="Index",
                yaxis_title="Amplitude",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download Processed Data
        with st.expander("⬇️ Download Processed Data"):
            for name in selected_datasets:
                csv = dfs[name].to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {name}", data=csv, file_name=f"processed_{name}.csv", mime="text/csv")
