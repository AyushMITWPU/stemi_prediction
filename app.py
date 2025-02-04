import streamlit as st
import pandas as pd
import os
import random
from input import gather_user_input  # Import the function from input.py

def save_files(uploaded_files, base_path, next_id_rnd):
    """Save uploaded .dat and .hea files to the specified directory."""
    new_dir = os.path.join(base_path, str(next_id_rnd))
    os.makedirs(new_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.dat') or uploaded_file.name.endswith('.hea'):
            new_file_name = f"{next_id_rnd}.{uploaded_file.name.split('.')[-1]}"
            with open(os.path.join(new_dir, new_file_name), "wb") as f:
                f.write(uploaded_file.getbuffer())

def main():
    st.title("ECG Data Upload")

    uploaded_files = st.file_uploader("Upload .dat and .hea files", type=['dat', 'hea'], accept_multiple_files=True)

    if st.button("Submit"):
        if uploaded_files:
            base_path = "data/Test_data/"  # Adjust the path as necessary
            metadata_file = os.path.join(base_path, "Test_metadata.csv")

            # Gather user input using the function from input.py
            next_id_rnd, label, age, sex = gather_user_input(metadata_file)

            save_files(uploaded_files, base_path, next_id_rnd)

            # Generate a unique patient_id
            df = pd.read_csv(metadata_file) if os.path.exists(metadata_file) else pd.DataFrame(columns=["id_rnd", "label", "patient_id", "path", "age", "sex"])
            existing_patient_ids = df['patient_id'].unique() if not df.empty else []
            patient_id = generate_unique_patient_id(existing_patient_ids)

            # Construct the new record as a DataFrame
            new_record = pd.DataFrame({
                "id_rnd": [next_id_rnd],
                "label": [label],
                "patient_id": [patient_id],
                "path": [f"{next_id_rnd}/{next_id_rnd}"],
                "age": [age],
                "sex": [sex]
            })

            # Append the new record to the DataFrame
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(metadata_file, index=False)

            st.success(f"New record added: {new_record.iloc[0].to_dict()}")
        else:
            st.warning("Please upload at least one .dat or .hea file.")

if __name__ == "__main__":
    main()