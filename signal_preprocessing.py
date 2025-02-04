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
        return signal  # Return unfiltered signal if error occurs

# Streamlit UI
st.title("📊 Signal Preprocessing Tool")
st.write("Upload a signal file and apply preprocessing techniques.")

# File upload
uploaded_file = st.file_uploader("📁 Upload a CSV file containing signal data", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.write("### 📋 Uploaded Data Preview")
    st.dataframe(df.head())

    # Check for required column
    if 'amplitude' not in df.columns:
        st.error("❌ The uploaded file must contain a column named 'amplitude'.")
    else:
        # Handle missing values
        df = df.dropna(subset=['amplitude'])

        # Plot raw signal
        with st.expander("📈 View Raw Signal", expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df['amplitude'], mode='lines', name='Raw Signal'))
            fig.update_layout(title="Raw Signal", xaxis_title="Index", yaxis_title="Amplitude")
            st.plotly_chart(fig, use_container_width=True)

        # Preprocessing options
        with st.expander("⚙️ Preprocessing Options"):
            cutoff = st.slider("Low-pass filter cutoff frequency (Hz):", min_value=1, max_value=100, value=10)
            fs = st.number_input("Sampling frequency (Hz):", min_value=1, value=100)

            # Apply preprocessing
            if st.button("Apply Low-pass Filter"):
                df['filtered'] = lowpass_filter(df['amplitude'], cutoff, fs)

                # Plot filtered signal
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df['amplitude'], mode='lines', name='Raw Signal', opacity=0.5))
                fig.add_trace(go.Scatter(y=df['filtered'], mode='lines', name='Filtered Signal', line=dict(color='orange')))

                fig.update_layout(
                    title="Filtered Signal",
                    xaxis_title="Index",
                    yaxis_title="Amplitude",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

        # Option to download processed data
        with st.expander("⬇️ Download Processed Data"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="processed_signal.csv", mime="text/csv")
