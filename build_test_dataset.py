import wfdb
import pandas as pd
import numpy as np
import scipy.signal as sgn
import h5py
import os

# Leads available in the GE MUSE format.
leads_used = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

def remove_baseline_filter(sample_rate):
    """For a given sampling rate"""
    fc = 0.8  # [Hz], cutoff frequency
    fst = 0.2  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)

    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(
        filterorder, wn, rp, rs, btype="high", ftype="ellip", output="sos"
    )

    return sos

def normalize(ecg, sample_rate):
    """Take a stacked array with all lead data, remove the baseline, resample to 400Hz, and zero pad to length 4096."""
    # Remove baseline.
    sos = remove_baseline_filter(sample_rate)
    ecg_nobaseline = sgn.sosfiltfilt(sos, ecg, padtype="constant", axis=-1)

    # Resample to 400Hz (4000 samplings).
    new_freq = 400
    ecg_resampled = sgn.resample_poly(
        ecg_nobaseline, up=new_freq, down=sample_rate, axis=-1
    )

    # Zero pad from 4000 to a length of 4096 to match the CNN design used.
    new_len = 4096
    n_leads, len = ecg_resampled.shape
    ecg_zeropadded = np.zeros([n_leads, new_len])
    pad = (new_len - len) // 2
    ecg_zeropadded[..., pad : len + pad] = ecg_resampled

    return ecg_zeropadded

def main():
    base_path = "/content/stemi_prediction/data/Test_data"  # Base path for data files
    out_path = "/content/stemi_prediction/data/"

    # Load the metadata
    test_records = pd.read_csv(os.path.join(out_path, "Test_metadata.csv"))

    # Check if there are any records
    if test_records.empty:
        print("No records found in Test_metadata.csv.")
        return

    # Process only the last entry in the metadata
    last_record = test_records.iloc[-1]
    raw_file = last_record['path']  # This should contain the relative path

    # Construct the full path to the .dat and .hea files
    dat_file_path = os.path.join(base_path, raw_file + ".dat")  # Constructing path for .dat file
    hea_file_path = os.path.join(base_path, raw_file + ".hea")  # Constructing path for .hea file

    # Print the constructed paths for debugging
    print(f"Constructed .dat file path: {dat_file_path}")
    print(f"Constructed .hea file path: {hea_file_path}")

    # Check if the .dat and .hea files exist
    if not os.path.exists(dat_file_path):
        print(f"File {dat_file_path} not found.")
        return
    if not os.path.exists(hea_file_path):
        print(f"File {hea_file_path} not found.")
        return

    # Read the ECG signal
    try:
        signal, meta = wfdb.rdsamp(dat_file_path)
    except Exception as e:
        print(f"Error reading {dat_file_path}: {e}")
        return

    # Normalize the ECG signal
    normalized_ecg = normalize(signal, 500)  # Assuming a sample rate of 500 Hz

    # Create HDF5 file and datasets
    with h5py.File(os.path.join(out_path, "test_data.h5"), "w") as f:
        set_name = "test"
        x_ecg = f.create_dataset("x_ecg_{}".format(set_name), data=normalized_ecg, dtype="f8")
        x_age = f.create_dataset("x_age_{}".format(set_name), data=np.array([last_record["age"]]), dtype="i4")
        x_is_male = f.create_dataset("x_is_male_{}".format(set_name), data=np.array([1 if last_record["gender"] == "male" else 0]), dtype="i4")

if __name__ == "__main__":
    main()