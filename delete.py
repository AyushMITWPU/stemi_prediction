import pandas as pd
import os

def main():
    # Define the path to the Test_metadata.csv file
    metadata_file = "data/Test_metadata.csv"  # Adjust the path as necessary

    # Check if the file exists and read it
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
    else:
        print("The metadata file does not exist.")
        return

    # Prompt user for id_rnd to delete
    id_rnd_to_delete = int(input("Enter id_rnd of the record to delete: "))

    # Check if the id_rnd exists in the DataFrame
    if id_rnd_to_delete not in df['id_rnd'].values:
        print(f"No record found with id_rnd: {id_rnd_to_delete}")
        return

    # Delete the record with the specified id_rnd
    df = df[df['id_rnd'] != id_rnd_to_delete]

    # Save the updated DataFrame back to the CSV file
    df.to_csv(metadata_file, index=False)

    print(f"Record with id_rnd {id_rnd_to_delete} has been deleted.")

if __name__ == "__main__":
    main()