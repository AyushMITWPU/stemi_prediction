import argparse
from argparse import Namespace
import json
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from dataloader import MyECGDataset
from model import EnsembleECGModel

if __name__ == "__main__":
    # System arguments
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument(
        "--input_data", type=str, default="data/test_data.h5", help="Path to data."
    )
    sys_parser.add_argument(
        "--log_dir", type=str, default="logs/", help="Path to dir model weights"
    )
    settings, _ = sys_parser.parse_known_args()

    # Read config file
    file_path = os.path.join(os.getcwd(), settings.log_dir, "config.json")
    with open(file_path) as json_file:
        mydict = json.load(json_file)
    config = Namespace(**mydict)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------------
    dataset = MyECGDataset(settings.input_data)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False
    )

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    model = EnsembleECGModel(config, settings.log_dir)
    model.eval()  # Set the model to evaluation mode

    # -----------------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------------
    all_probs = []
    all_labels = []
    all_ids = []
    
    for batch_idx, batch in enumerate(test_loader):
        # Extract data from batch
        traces, labels, ids, age, sex = batch
        traces = traces.to(device=config.device)
        labels = labels.to(device=config.device)
        age_sex = torch.stack([sex, age]).t().to(device=config.device)
        
        # Forward pass
        with torch.no_grad():
            inp = traces, age_sex
            logits = model(inp)
            probs = F.softmax(logits, dim=-1)
            
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_ids.append(ids)

        # Print true labels and predicted outputs
        for i in range(len(labels)):
            true_label = labels[i].item()
            predicted_label = torch.argmax(probs[i]).item()
            print(f"True Label: {true_label}, Predicted Label: {predicted_label}")

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    
    # Prepare DataFrame for saving
    df_new = pd.DataFrame({
        "ids": np.asarray(all_ids),
        "labels": np.asarray(all_labels),
        "logits": list(np.asarray(all_probs))
    })

    # Save (ids, labels, logits) to csv file, appending if it already exists
    output_file = os.path.join(settings.log_dir, "predictions.csv")
    
    if os.path.exists(output_file):
        # Read existing predictions
        df_existing = pd.read_csv(output_file)
        
        # Filter out duplicates
        df_new = df_new[~df_new['ids'].isin(df_existing['ids'])]
        
        # Append new records if any
        if not df_new.empty:
            df_new.to_csv(output_file, mode='a', header=False, index=False)  # Append without header
            print(f"Appended {len(df_new)} new records to predictions.csv.")
        else:
            print("No new records to append.")
    else:
        # Create new file if it doesn't exist
        df_new.to_csv(output_file, index=False)  # Create new file
        print("Created predictions.csv with new records.")

    print("Predictions saved to csv file.")