import os
import pandas as pd

def merge_csv_files_with_column(input_directory, output_file):
    """
    1. Merges all (semicolon-separated) CSV files in input_directory into one long CSV.
    2. Adds a new column called 'partOfDocument'. If the 'TANAP ID' cell is NaN or empty,
       'partOfDocument' = 'false'. Otherwise, 'true'.
    """
    all_dfs = []  # Will collect valid DataFrames here

    # Gather all semicolon-separated CSV files in the directory
    csv_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.csv')]
    # Sort the files for a predictable merge order (optional)
    csv_files.sort()

    for csv_file in csv_files:
        csv_path = os.path.join(input_directory, csv_file)
        
        # Read the CSV with semicolon as separator. 
        # Using dtype=str keeps all data as strings (important for consistent processing).
        try:
            df = pd.read_csv(csv_path, sep=';', dtype=str, engine='python')
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        
        # Strip whitespace from column names to handle accidental leading/trailing spaces
        df.columns = df.columns.str.strip()

        # Look for a column that matches "TANAP ID" (case-insensitive)
        matching_cols = [col for col in df.columns if col.lower() == "tanap id"]
        
        if not matching_cols:
            print(f"Skipping {csv_file}: No column named 'TANAP ID' (case-insensitive).")
            continue
        else:
            tanap_col = matching_cols[0]  # The actual column name in this CSV

        # Replace empty strings or obvious "NaN" strings with proper NaN so .notna() will catch them
        df[tanap_col] = df[tanap_col].replace({'': pd.NA, 'NaN': pd.NA, 'nan': pd.NA})
        
        # Create the new 'partOfDocument' column based on TANAP ID presence
        df['partOfDocument'] = df[tanap_col].notna().map({True: 'true', False: 'false'})
        
        # Collect this DataFrame
        all_dfs.append(df)

    if not all_dfs:
        print("No valid CSV files found or all skipped due to missing 'TANAP ID'. No output generated.")
        return

    # Concatenate all DataFrames into one
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Write out to CSV. Default separator is comma, but if you prefer semicolons, specify sep=';'
    merged_df.to_csv(output_file, index=False, sep=',')
    print(f"Merged data saved to {output_file}")


if __name__ == "__main__":
    # Example usage:
    input_dir = "/Users/gavinl/Desktop/XML_segmentation/Datasets/Annotations"
    output_csv = "/Users/gavinl/Desktop/XML_segmentation/Datasets/Intermediate Datasets/renate_data.csv"
    
    merge_csv_files_with_column(input_dir, output_csv)