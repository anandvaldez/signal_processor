import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Function to apply a low-pass filter
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# Streamlit UI
st.title("Signal Preprocessing Tool")
st.write("Upload a signal file and apply basic preprocessing techniques.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file containing signal data", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data:")
    st.write(df.head())

    # Check for required column
    if 'amplitude' not in df.columns:
        st.error("The uploaded file must contain a column named 'amplitude'.")
    else:
        # Plot raw signal
        st.write("### Raw Signal:")
        st.line_chart(df['amplitude'])

        # Preprocessing options
        st.write("### Preprocessing Options:")
        cutoff = st.slider("Low-pass filter cutoff frequency (Hz):", min_value=1, max_value=100, value=10)
        fs = st.number_input("Sampling frequency (Hz):", min_value=1, value=100)

        # Apply preprocessing
        if st.button("Apply Low-pass Filter"):
            filtered_signal = lowpass_filter(df['amplitude'], cutoff, fs)

            # Add filtered signal to dataframe
            df['filtered'] = filtered_signal

            # Plot filtered signal
            st.write("### Filtered Signal:")
            fig, ax = plt.subplots()
            ax.plot(df.index, df['amplitude'], label='Raw Signal', alpha=0.5)
            ax.plot(df.index, df['filtered'], label='Filtered Signal', color='orange')
            ax.set_title("Signal Before and After Filtering")
            ax.legend()
            st.pyplot(fig)

        # Option to download processed data
        st.write("### Download Processed Data:")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="processed_signal.csv", mime="text/csv")
