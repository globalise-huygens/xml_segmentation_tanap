import os
import pandas as pd

def load_xml_files(root_dir):
    """
    Walk through all subdirectories of root_dir and build a dictionary 
    mapping XML file names to their content.
    """
    xml_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.xml'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        xml_content = file.read()
                    xml_dict[filename] = xml_content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return xml_dict

# Define the root directory that contains the XML files
root_directory = '/Users/gavinl/Desktop/XML_segmentation/Datasets/Intermediate Datasets/xml_data'

# Load all XML file contents into a dictionary
xml_data_dict = load_xml_files(root_directory)
print("Loaded XML file keys:", list(xml_data_dict.keys()))

# Read the CSV file
csv_file = '/Users/gavinl/Desktop/XML_segmentation/Datasets/Intermediate Datasets/renate_data.csv'
df = pd.read_csv(csv_file)

# Debug: print CSV columns and unique file names before processing
print("CSV columns:", df.columns)
print("Unique 'Scan File_Name' values before processing:", df['Scan File_Name'].unique())

# Normalize the CSV 'Scan File_Name' values: convert to string, strip whitespace, and lower-case them.
df['Scan File_Name'] = df['Scan File_Name'].astype(str).str.strip().str.lower()

# Append the '.xml' extension if not present.
df['Scan File_Name'] = df['Scan File_Name'].apply(lambda x: x if x.endswith('.xml') else f"{x}.xml")

print("Unique 'Scan File_Name' values after processing:", df['Scan File_Name'].unique())

# Create a normalized version of the XML dictionary keys (lower-case and stripped)
xml_data_dict_lower = {k.lower().strip(): v for k, v in xml_data_dict.items()}
print("Processed XML file keys:", list(xml_data_dict_lower.keys()))

# Map the CSV file names to the XML content
df['xml_data'] = df['Scan File_Name'].map(xml_data_dict_lower)

# Fill in rows where there was no matching XML file
df['xml_data'] = df['xml_data'].fillna('XML file not found')

# Print out how many rows did not find a matching XML file
print("Rows with XML file not found:", df[df['xml_data'] == 'XML file not found'].shape[0])
print(df.head())

# Save the updated CSV to disk
new_csv_file = "/Users/gavinl/Desktop/XML_segmentation/Datasets/Intermediate Datasets/renate_data_with_xml.csv"
df.to_csv(new_csv_file, index=False)
print("CSV file updated successfully.")