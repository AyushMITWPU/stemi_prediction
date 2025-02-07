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

def main():
    # Define the path to the Test_metadata.csv file
    metadata_file = "data/Test_metadata.csv"  # Adjust the path as necessary

    # Check if the file exists and read it
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
    else:
        # Create an empty DataFrame with the required columns if the file does not exist
        df = pd.DataFrame(columns=["id_rnd", "label", "patient_id", "path", "age", "sex"])

    # Get the next id_rnd
    next_id_rnd = get_next_id_rnd(df)

    # Prompt user for input
    label = input("Enter label (e.g., normal, mi, nstemi): ")
    age = int(input("Enter age: "))
    sex = int(input("Enter sex (0 for male, 1 for female): "))

    # Generate a unique patient_id
    existing_patient_ids = df['patient_id'].unique() if not df.empty else []
    patient_id = generate_unique_patient_id(existing_patient_ids)

    # Construct the new record as a DataFrame
    new_record = pd.DataFrame({
        "id_rnd": [next_id_rnd],
        "label": [label],
        "patient_id": [patient_id],
        "path": [f"/{next_id_rnd}/{next_id_rnd}"],
        "age": [age],
        "sex": [sex]
    })

    # Append the new record to the DataFrame using pd.concat
    df = pd.concat([df, new_record], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(metadata_file, index=False)

    print(f"New record added: {new_record.iloc[0].to_dict()}")

if __name__ == "__main__":
    main()