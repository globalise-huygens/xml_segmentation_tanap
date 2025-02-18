import os
import re
import xml.etree.ElementTree as ET

def get_region_types_from_directory(dir_path):
    # Namespace used in PAGE XML
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Set to keep track of unique types
    all_region_types = set()

    # Regex to capture what's inside "type:XYZ;"
    # Adjust the pattern if your "custom" attribute format differs
    pattern = re.compile(r"type:\s*([^;]+);")

    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".xml"):
            xml_file_path = os.path.join(dir_path, filename)
            try:
                tree = ET.parse(xml_file_path)
                root = tree.getroot()

                # Find all TextRegion elements
                for region in root.findall(".//pc:TextRegion", ns):
                    custom_attr = region.get("custom")
                    if custom_attr:
                        match = pattern.search(custom_attr)
                        if match:
                            region_type = match.group(1).strip()
                            all_region_types.add(region_type)
            except ET.ParseError:
                print(f"Warning: Could not parse '{xml_file_path}' as valid XML.")
            except Exception as e:
                print(f"Error processing '{xml_file_path}': {e}")

    return all_region_types


if __name__ == "__main__":
    # Example usage
    directory_path = "/Users/gavinl/Desktop/Inventory 1120/page"
    region_types = get_region_types_from_directory(directory_path)

    print("Found region types:")
    for rt in sorted(region_types):
        print(f" - {rt}")
