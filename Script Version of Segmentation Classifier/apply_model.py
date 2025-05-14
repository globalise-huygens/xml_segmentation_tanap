#!/usr/bin/env python3
"""
apply_model.py

Application script that:
1. Reads XML files from a directory, sorted by their trailing digit (e.g. _0001.xml before _0002.xml).
2. Creates a CSV [xml_file_name, xml_data] in that sorted order.
3. Applies the extended feature-extraction script (xml_to_vectors_extended.py) to generate page features.
4. Loads a trained segmentation model (and label encoder) from joblib.
5. Predicts the segmentation label ("NONE", "START", etc.) for each page.
6. Writes an intermediate CSV with [xml_file_name, output].
7. Appends a scan_url column matching the logic from segmentation_gui.py so the final CSV contains [xml_file_name, output, scan_url].
"""

import sys
import os
import re
import glob
import joblib
# Pandas is essential for data manipulation; fail fast if it's missing.
try:
    import pandas as pd
except ImportError as e:
    print("ERROR: The 'pandas' package is required but not installed. "
          "Please install it with 'pip install pandas' and rerun the script.")
    sys.exit(1)

def extract_trailing_number(filename: str) -> int:
    """Extract the trailing numeric portion of the filename (before .xml).

    Examples
    --------
    'NL-HaNA_1.04.02_1120_0001.xml' -> 1
    'NL-HaNA_1.04.02_1120_0002.xml' -> 2

    Returns
    -------
    int
        The trailing number or 999999999 if the pattern is not found (puts unsorted files last).
    """
    match = re.search(r'_(\d+)\.xml$', filename)
    return int(match.group(1)) if match else 999999999


def append_scan_urls(csv_path: str) -> None:
    """Append a `scan_url` column to the results CSV.

    The URL format follows the original logic used in *segmentation_gui.py* /
    *get_scan_urls.py*:

        https://www.nationaalarchief.nl/onderzoeken/archief/{{archive_code}}/invnr/{{inventory_number}}/file/{{base_name}}

    Parameters
    ----------
    csv_path : str
        Path to the CSV that already contains at least the `xml_file_name` column.
    """
    df = pd.read_csv(csv_path)

    # Strip ".xml" suffix to get the base name.
    df["base_name"] = df["xml_file_name"].str.replace(r"\.xml$", "", regex=True)

    # The base name is of the form 'NL-HaNA_<archive_code>_<inventory_number>_<page>'
    # so splitting on "_" gives us the archive code and inventory number.
    parts = df["base_name"].str.split("_", expand=True)
    df["archive_code"] = parts[1]
    df["inventory_number"] = parts[2]

    # Build the full URL.
    df["scan_url"] = (
        "https://www.nationaalarchief.nl/onderzoeken/archief/"
        + df["archive_code"]
        + "/invnr/"
        + df["inventory_number"]
        + "/file/"
        + df["base_name"]
    )

    # Remove helper columns and overwrite the same CSV path.
    df.drop(columns=["base_name", "archive_code", "inventory_number"], inplace=True)
    df.to_csv(csv_path, index=False)


def main(xml_dir: str, model_path: str, label_encoder_path: str, output_csv: str) -> None:
    """Run the end‑to‑end inference pipeline and save results to *output_csv*."""

    # 1. Collect XML files and sort them numerically on the trailing page number.
    xml_files = sorted(
        glob.glob(os.path.join(xml_dir, "*.xml")),
        key=lambda f: extract_trailing_number(os.path.basename(f)),
    )

    if not xml_files:
        print(f"No .xml files found in directory: {xml_dir}")
        sys.exit(1)

    # 2. Create an intermediate CSV [xml_file_name, xml_data] for feature extraction.
    rows = []
    for xml_file in xml_files:
        with open(xml_file, "r", encoding="utf-8") as f:
            rows.append([os.path.basename(xml_file), f.read()])

    input_df = pd.DataFrame(rows, columns=["xml_file_name", "xml_data"])
    temp_input_csv = "temp_input_files.csv"
    input_df.to_csv(temp_input_csv, index=False)
    print(f"Created intermediate CSV: {temp_input_csv}")

    # 3. Run feature extraction (53‑feature version by default).
    try:
        from xml_to_vectors_53_features import process_dataset
    except ImportError:
        print("ERROR: Could not import 'process_dataset' from 'xml_to_vectors_53_features.py'.")
        sys.exit(1)

    # --- Robustly call process_dataset no matter the signature -----------------
    import inspect

    temp_features_csv = "temp_features.csv"
    sig = inspect.signature(process_dataset)
    param_names = list(sig.parameters)

    # Build arguments based on parameter names.
    # We assume only positional parameters are used.
    arg_values = {}
    for name in param_names:
        lname = name.lower()
        if "input" in lname:
            arg_values[name] = temp_input_csv
        elif "xml" in lname and "column" in lname:
            arg_values[name] = "xml_data"
        elif "feature" in lname:
            arg_values[name] = temp_features_csv
        elif "output" in lname:
            arg_values[name] = temp_features_csv
        else:
            # Default anything unrecognised to the feature CSV
            arg_values[name] = temp_features_csv

    # Preserve original parameter order in the call.
    ordered_args = [arg_values[n] for n in param_names]

    try:
        process_dataset(*ordered_args)
    except Exception as err:
        print(f"ERROR: Failed while running process_dataset(): {err}")
        print(f"Tried calling with parameters: {ordered_args}")
        sys.exit(1)

    print(f"Extracted features CSV: {temp_features_csv}")

    # 4. Load model and label encoder.
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # 5. Prepare inference matrix.
    features_df = pd.read_csv(temp_features_csv)
    drop_cols = ["xml_file_name", "xml_data", "output"]
    X_infer = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns], errors="ignore")

    # 6. Predict segmentation labels.
    preds_numeric = model.predict(X_infer)
    preds_labels = label_encoder.inverse_transform(preds_numeric)

    # 7. Build results DataFrame and save.
    result_df = pd.DataFrame({"xml_file_name": features_df["xml_file_name"], "output": preds_labels})
    result_df.to_csv(output_csv, index=False)

    # 8. Append scan URLs so the final CSV matches the GUI output.
    append_scan_urls(output_csv)
    print(f"Inference complete! Saved results with scan URLs to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python apply_model.py <XML_DIR> <MODEL_PATH> <LABEL_ENCODER_PATH> <OUTPUT_CSV>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
