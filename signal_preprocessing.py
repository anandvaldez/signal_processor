import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, resample
from scipy.fftpack import fft, fftfreq

# Function to apply a filter
def apply_filter(signal, cutoff, fs, order=5, filter_type='low'):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = np.array(cutoff) / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        return filtfilt(b, a, signal)
    except Exception as e:
        st.error(f"Error in filtering: {e}")
        return signal

# Function for time-domain analysis
def time_domain_analysis(signal):
    return {
        "Mean": np.mean(signal),
        "Standard Deviation": np.std(signal),
        "RMS": np.sqrt(np.mean(signal**2)),
        "Peak-to-Peak": np.ptp(signal),
        "Zero-Crossing Rate": np.mean(np.diff(np.sign(signal)) != 0)
    }

# Function for FFT analysis
def compute_fft(signal, fs):
    signal = np.asarray(signal)  # Ensure NumPy array
    N = len(signal)
    freq = fftfreq(N, d=1/fs)[:N//2]  # Only positive frequencies
    fft_values = np.abs(fft(signal)[:N//2])  # Magnitude spectrum
    return freq, fft_values

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

st.title("📊 Signal Processing & Comparison Tool")
st.write("Upload multiple signal files, apply preprocessing, and compare.")

# File upload
uploaded_files = st.file_uploader("📁 Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs, raw_data = {}, {}
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        if 'amplitude' not in df.columns:
            st.error(f"❌ {uploaded_file.name} is missing the 'amplitude' column.")
            continue
        df = df.dropna(subset=['amplitude'])
        dfs[uploaded_file.name] = df
        raw_data[uploaded_file.name] = df.copy()

    selected_datasets = st.multiselect("📌 Select datasets to process", list(dfs.keys()), default=list(dfs.keys()))

    if selected_datasets:
        for name in selected_datasets:
            df = dfs[name]
            with st.expander(f"⚙️ Preprocessing & Analysis: {name}"):
                fs = st.number_input(f"Sampling frequency (Hz) for {name}", 1, 500, 100, key=f"fs_{name}")
                df['normalized'] = (df['amplitude'] - df['amplitude'].min()) / (df['amplitude'].max() - df['amplitude'].min())

                # Time-domain analysis
                st.subheader("📈 Time-Domain Analysis (Raw Data)")
                analysis = time_domain_analysis(df['amplitude'])
                st.json(analysis)

                # Resampling
                new_fs = st.slider(f"Resampling Frequency (Hz) for {name}", 1, fs, fs, key=f"resample_{name}")
                if new_fs != fs:
                    resampled_signal = resample(df['amplitude'], int(len(df) * new_fs / fs))
                    resampled_df = pd.DataFrame({'amplitude': resampled_signal})
                    df = df.iloc[:len(resampled_signal)]
                    df['resampled'] = resampled_signal

                # Filtering
                filter_type = st.selectbox(f"Filter Type for {name}", ["low", "high", "band"], key=f"filter_type_{name}")
                if filter_type == "band":
                    cutoff = st.slider(f"Band-pass Cutoff (Hz) for {name}", 1, fs // 2, (10, 50), key=f"cutoff_{name}")
                else:
                    cutoff = st.slider(f"Cutoff Frequency (Hz) for {name}", 1, fs // 2, 10, key=f"cutoff_{name}")

                if st.button(f"Apply Filter for {name}"):
                    df['filtered'] = apply_filter(df['amplitude'], cutoff, fs, filter_type=filter_type)
                    st.session_state.processed_data[name] = df
                    st.success(f"Filter applied to {name}!")

                    # Time-domain analysis for filtered signal
                    st.subheader("📉 Time-Domain Analysis (Filtered Data)")
                    filtered_analysis = time_domain_analysis(df['filtered'])
                    st.json(filtered_analysis)

                # FFT Analysis
                st.subheader("🔍 Frequency-Domain Analysis (FFT)")
                freq, fft_values = compute_fft(df['amplitude'], fs)
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(x=freq, y=fft_values, mode='lines', name=f'FFT: {name}'))
                fig_fft.update_layout(title="FFT Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", hovermode="x unified")
                st.plotly_chart(fig_fft, use_container_width=True)

        # Comparison
        with st.expander("📊 Compare Signals", expanded=True):
            comparison_options = st.multiselect("Select data to compare", ["Raw Data", "Filtered Data"], default=["Raw Data", "Filtered Data"])
            fig = go.Figure()

            if "Raw Data" in comparison_options:
                for name in selected_datasets:
                    fig.add_trace(go.Scatter(y=raw_data[name]['amplitude'], mode='lines', name=f'Raw: {name}'))

            if "Filtered Data" in comparison_options:
                for name in st.session_state.processed_data:
                    fig.add_trace(go.Scatter(y=st.session_state.processed_data[name]['filtered'], mode='lines', name=f'Filtered: {name}'))
            
            fig.update_layout(title="Signal Comparison", xaxis_title="Index", yaxis_title="Amplitude", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        # Download Processed Data
        with st.expander("⬇️ Download Processed Data"):
            for name in st.session_state.processed_data:
                csv = st.session_state.processed_data[name].to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {name}", data=csv, file_name=f"processed_{name}.csv", mime="text/csv")
