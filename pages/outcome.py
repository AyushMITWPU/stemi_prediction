import streamlit as st
import pandas as pd
import subprocess
import os

# Title of the app
st.title("Outcome Prediction")

# Button to run predictions
if st.button("Run Predictions"):
    # Run the prediction.py script
    try:
        # Call the prediction script
        result = subprocess.run(
            ['python3', '/content/stemi_prediction/prediction.py', '--input_data', '/content/stemi_prediction/data/test_data.h5'],
            capture_output=True,
            text=True
        )
        
        # Check if the script ran successfully
        if result.returncode == 0:
            st.success("Predictions completed successfully.")
            st.text(result.stdout)  # Display any output from the script
        else:
            st.error("Error running predictions.")
            st.text(result.stderr)  # Display any error messages
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Load and display predictions from the CSV file
output_file = '/content/stemi_prediction/logs/predictions.csv'

if os.path.exists(output_file):
    predictions_df = pd.read_csv(output_file)
    st.subheader("Predictions Data")
    st.write(predictions_df)

    # Optionally, display some statistics
    st.subheader("Statistics")
    st.write(f"Total Predictions: {len(predictions_df)}")
    st.write(f"Unique IDs: {predictions_df['ids'].nunique()}")

    # Display a sample of the predictions
    st.subheader("Sample Predictions")
    sample_df = predictions_df.sample(min(5, len(predictions_df)))  # Show up to 5 random samples
    st.write(sample_df)

    # Optionally, allow users to download the predictions
    st.download_button(
        label="Download Predictions CSV",
        data=predictions_df.to_csv(index=False).encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv'
    )
else:
    st.warning("No predictions found. Please run the predictions first.")
