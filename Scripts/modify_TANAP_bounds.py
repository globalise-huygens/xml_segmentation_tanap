import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("/Users/gavinl/Desktop/XML_segmentation/Datasets/Intermediate Datasets/renate_data_with_xml.csv",sep=",")

# Flag to track whether we are within a document
in_document = False

for i, row in df.iterrows():
    current_value = row["TANAP Boundaries"]
    if current_value == "START":
        in_document = True
    elif current_value == "END":
        in_document = False
    elif pd.isna(current_value):
        # If within a document, mark as "MIDDLE"; if not, mark as "NONE"
        df.at[i, "TANAP Boundaries"] = "MIDDLE" if in_document else "NONE"

# Save the modified dataset
df.to_csv("/Users/gavinl/Desktop/XML_segmentation/Datasets/Training dataset/final_dataset_xml.csv", index=False)