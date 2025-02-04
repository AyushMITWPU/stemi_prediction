import pandas as pd
import os
import random
import streamlit as st  # Import Streamlit

def get_next_id_rnd(df):
    """Get the next id_rnd based on the existing DataFrame."""
    if df.empty:
        return 1  # Start from 1 if the DataFrame is empty
    else:
        return df['id_rnd'].max() + 1  # Increment the maximum id_rnd

def generate_unique_patient_id(existing_ids):
    """Generate a unique 5-digit patient_id that is not in existing_ids."""
    while True:
        patient_id = random.randint(10000, 99999)  # Generate a random 5-digit number
        if patient_id not in existing_ids:
            return patient_id

def gather_user_input(metadata_file):
    """Gather user input for label, age, and sex using Streamlit."""
    # Check if the file exists and read it
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
    else:
        df = pd.DataFrame(columns=["id_rnd", "label", "patient_id", "path", "age", "sex"])

    # Get the next id_rnd
    next_id_rnd = get_next_id_rnd(df)

    # Gather additional input from the user using Streamlit
    label = st.text_input("Enter label (e.g., normal, mi, nstemi):")
    age = st.number_input("Enter age:", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Select sex:", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")

    return next_id_rnd, label, age, sex