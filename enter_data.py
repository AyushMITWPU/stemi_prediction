import streamlit as st
import pandas as pd
import os
import random

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

    # File uploader for .dat and .hea files
    uploaded_files = st.file_uploader("Upload .dat and .hea files", type=['dat', 'hea'], accept_multiple_files=True)

    # Gather user input using Streamlit
    with st.form(key='input_form'):
        label = st.text_input("Enter label (e.g., normal, mi, nstemi):")
        age = st.number_input("Enter age:", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Select sex:", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        
        # Submit button for the form
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if uploaded_files:
            base_path = "/content/stemi_prediction/data/Test_data/"  # Adjust the path as necessary
            metadata_file = os.path.join("/content/stemi_prediction/data", "Test_metadata.csv")

            # Read existing metadata
            df = pd.read_csv(metadata_file) if os.path.exists(metadata_file) else pd.DataFrame(columns=["id_rnd", "label", "patient_id", "path", "age", "sex"])

            # Get the next id_rnd
            next_id_rnd = get_next_id_rnd(df)

            # Generate a unique patient_id
            existing_patient_ids = df['patient_id'].unique() if not df.empty else []
            patient_id = generate_unique_patient_id(existing_patient_ids)

            # Save the uploaded files
            save_files(uploaded_files, base_path, next_id_rnd)

            # Construct the new record as a DataFrame
            new_record = pd.DataFrame({
                "id_rnd": [next_id_rnd],
                "label": [label],
                "patient_id": [patient_id],
                "path": [f"/{next_id_rnd}/{next_id_rnd}"],
                "age": [age],
                "sex": [sex]
            })

            # Append the new record to the DataFrame
            df = pd.concat([df, new_record], ignore_index=True)
            # Save the updated DataFrame back to the metadata file
            df.to_csv(metadata_file, index=False)

            st.success("Files uploaded and metadata saved successfully!")
        else:
            st.warning("Please upload at least one .dat or .hea file.")

if __name__ == "__main__":
    main()