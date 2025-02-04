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
if 'saved_trimmed_data' not in st.session_state:
    st.session_state.saved_trimmed_data = {}

# Streamlit UI
st.title("📊 Signal Preprocessing & Comparison Tool")
st.write("Upload multiple signal files, apply preprocessing, trim data, and compare.")

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

        # Step 2: Data Trimming
        selected_trim_dataset = st.selectbox("Select dataset for trimming", selected_datasets)
        if selected_trim_dataset in st.session_state.processed_data:
            df = st.session_state.processed_data[selected_trim_dataset]

            with st.expander(f"✂️ Data Trimming: {selected_trim_dataset}", expanded=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df['filtered'], mode='lines', name=f'Filtered: {selected_trim_dataset}'))
                fig.update_layout(
                    title=f"Trim Data for {selected_trim_dataset}",
                    xaxis_title="Index",
                    yaxis_title="Amplitude",
                    xaxis=dict(rangeslider=dict(visible=True)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                start_idx, end_idx = st.slider(
                    f"Select trimming range for {selected_trim_dataset}",
                    0, len(df) - 1, (0, len(df) - 1),
                    key=f"trim_{selected_trim_dataset}"
                )
                trimmed_df = df.iloc[start_idx:end_idx]

                if st.button("Add Data for Comparison"):
                    st.session_state.saved_trimmed_data[selected_trim_dataset] = trimmed_df  # Save to session state
                    st.success(f"{selected_trim_dataset} added for comparison!")

        # Step 3: Comparison
        with st.expander("📊 Compare Signals", expanded=True):
            # Options for comparison
            comparison_options = st.multiselect(
                "Select data to compare",
                ["Raw Data", "Processed Data", "Trimmed Data", "Processed & Trimmed Data"],
                default=["Raw Data", "Processed Data", "Trimmed Data", "Processed & Trimmed Data"]
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

            if "Trimmed Data" in comparison_options:
                for name in st.session_state.saved_trimmed_data:
                    trimmed_df = st.session_state.saved_trimmed_data[name]
                    fig.add_trace(go.Scatter(y=trimmed_df['filtered'], mode='lines', name=f'Trimmed: {name}'))

            if "Processed & Trimmed Data" in comparison_options:
                for name in st.session_state.saved_trimmed_data:
                    trimmed_df = st.session_state.saved_trimmed_data[name]
                    fig.add_trace(go.Scatter(y=trimmed_df['filtered'], mode='lines', name=f'Processed & Trimmed: {name}'))

            fig.update_layout(
                title="Signal Comparison",
                xaxis_title="Index",
                yaxis_title="Amplitude",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download Processed Data
        with st.expander("⬇️ Download Processed Data"):
            for name in st.session_state.saved_trimmed_data:
                csv = st.session_state.saved_trimmed_data[name].to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {name}", data=csv, file_name=f"processed_{name}.csv", mime="text/csv")