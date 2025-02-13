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

def parse_probs(s):
    """
    Parse a probability string into a NumPy array.
    First, try to use json.loads. If that fails, fall back to manual parsing.
    """
    try:
        return np.array(json.loads(s))
    except json.decoder.JSONDecodeError:
        s = s.strip("[]")
        parts = s.replace(",", " ").split()
        return np.array([float(part) for part in parts])

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
    config_path = os.path.join(os.getcwd(), settings.log_dir, "config.json")
    with open(config_path) as json_file:
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
    # We'll ignore the provided ids and generate new incremental ones.
    for batch_idx, batch in enumerate(test_loader):
        traces, labels, _, age, sex = batch  # Ignore dataset ids here.
        traces = traces.to(device=config.device)
        labels = labels.to(device=config.device)
        age_sex = torch.stack([sex, age]).t().to(device=config.device)

        with torch.no_grad():
            inp = (traces, age_sex)
            logits = model(inp)
            probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        for i in range(len(labels)):
            true_label = labels[i].item()
            predicted_label = torch.argmax(probs[i]).item()
            sample_probs = probs[i].detach().cpu().numpy()
            print(f"True Label: {true_label}, Predicted Label: {predicted_label}, Prediction: {sample_probs}")

    all_probs = np.vstack(all_probs)  # Shape: (N, num_classes)
    all_labels = np.concatenate(all_labels, axis=0)

    # Create DataFrame using our own (to-be-generated) ids, labels, and probability predictions.
    df_new = pd.DataFrame({
        "labels": all_labels,
        "probs": [json.dumps(prob.tolist()) for prob in all_probs]
    })

    output_file = os.path.join(settings.log_dir, "predictions.csv")

    # Function to check if a new record is duplicate based on probability vector.
    def is_duplicate(row, existing_probs):
        new_prob = np.array(json.loads(row["probs"]))
        for p in existing_probs:
            if np.allclose(new_prob, p, atol=1e-6):
                return True
        return False

    if os.path.exists(output_file):
        # Read existing predictions
        df_existing = pd.read_csv(output_file)
        if "probs" not in df_existing.columns and "logits" in df_existing.columns:
            df_existing.rename(columns={"logits": "probs"}, inplace=True)

        existing_probs = []
        if "probs" in df_existing.columns:
            existing_probs = df_existing["probs"].apply(parse_probs).tolist()
        else:
            print("Warning: No probabilities column found in existing CSV.")

        # Remove new records that duplicate existing predictions (based on probabilities).
        df_new_filtered = df_new[~df_new.apply(lambda row: is_duplicate(row, existing_probs), axis=1)]
        
        if not df_new_filtered.empty:
            # Generate new incremental ids: start from max id in existing file (or 0 if none).
            if "ids" in df_existing.columns:
                max_id = df_existing["ids"].max()
            else:
                max_id = -1
            new_ids = np.arange(max_id + 1, max_id + 1 + len(df_new_filtered))
            df_new_filtered = df_new_filtered.copy()
            df_new_filtered["ids"] = new_ids
            # Reorder columns so that ids come first.
            df_new_filtered = df_new_filtered[["ids", "labels", "probs"]]
            df_new_filtered.to_csv(output_file, mode="a", header=False, index=False)
            print(f"Appended {len(df_new_filtered)} new records to predictions.csv.")
        else:
            print("No new records to append.")
    else:
        # File does not exist: generate new incremental ids starting at 0.
        new_ids = np.arange(len(df_new))
        df_new = df_new.copy()
        df_new["ids"] = new_ids
        df_new = df_new[["ids", "labels", "probs"]]
        df_new.to_csv(output_file, index=False)
        print("Created predictions.csv with new records.")

    print("Predictions saved to CSV file.")
