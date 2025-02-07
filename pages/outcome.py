import streamlit as st
import pandas as pd
import os

# Title of the app
st.title("Prediction Outcomes")

# Path to the predictions CSV file
predictions_file_path = '/content/stemi_prediction/logs/predictions.csv'

# Function to load predictions
def load_predictions():
    if os.path.exists(predictions_file_path):
        df = pd.read_csv(predictions_file_path)
        return df
    else:
        st.error("Predictions file not found.")
        return None

# Load the predictions
predictions_df = load_predictions()

# Display the predictions if available
if predictions_df is not None:
    st.subheader("Predictions Data")
    st.write(predictions_df)

    # Optionally, display some statistics
    st.subheader("Statistics")
    st.write(f"Total Predictions: {len(predictions_df)}")
    st.write(f"Unique IDs: {predictions_df['ids'].nunique()}")
    
    # Display a sample of the predictions
    st.subheader("Sample Predictions")
    st.write(predictions_df.sample(min(5, len(predictions_df))))  # Show up to 5 random samples

    # Optionally, allow users to download the predictions
    st.download_button(
        label="Download Predictions CSV",
        data=predictions_df.to_csv(index=False).encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv'
    )