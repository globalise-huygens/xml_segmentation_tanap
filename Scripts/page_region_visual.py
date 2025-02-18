#!/usr/bin/env python3
"""
This script reads a JPG image and its corresponding PAGE-format XML file,
parses the XML to extract text regions and their polygon coordinates, draws
overlays (filled and outlined polygons) on the image (using different colors per region type),
and saves the resulting image with the overlays.
"""

import os
import sys
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageColor

# Directories for images and XML files.
IMAGES_DIR = "/Users/gavinl/Desktop/Inventory 1120/1120"  # Update this path if necessary
XML_DIR = "/Users/gavinl/Desktop/Inventory 1120/page"      # Update this path if necessary

def extract_region_type(custom_attr):
    """
    Extract the region type from the custom attribute string.
    The custom attribute is expected to be in a format such as:
        "structure {type:header;}"
    This function uses a regular expression to find the word after 'type:'.
    
    Args:
        custom_attr (str): The custom attribute string.
        
    Returns:
        str or None: The region type (e.g., 'header') if found, else None.
    """
    match = re.search(r'type\s*:\s*(\w+)', custom_attr)
    if match:
        return match.group(1).lower()
    return None

def parse_coords(points_str):
    """
    Convert a points string from the XML into a list of (x, y) tuples.
    The points string is expected to be in the format:
        "x1,y1 x2,y2 x3,y3 ..."
        
    Args:
        points_str (str): The space-separated string of x,y coordinates.
        
    Returns:
        list: A list of (x, y) tuples as integers.
    """
    points = []
    for point in points_str.strip().split():
        try:
            x_str, y_str = point.split(',')
            points.append((int(x_str), int(y_str)))
        except ValueError:
            # Skip malformed coordinate pairs
            continue
    return points

def main(base_name):
    """
    Main processing function.
    
    Args:
        base_name (str): The base name for the image and XML file (without extension).
    """
    # Build full file paths for the image and XML file.
    image_path = os.path.join(IMAGES_DIR, base_name + ".jpg")
    xml_path = os.path.join(XML_DIR, base_name + ".xml")

    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(xml_path):
        print(f"XML file not found: {xml_path}")
        sys.exit(1)

    # Open the image using Pillow and convert to RGBA (to support transparency).
    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)
    
    # Create a separate overlay image (with a transparent background) to draw the polygons.
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Parse the XML file.
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        sys.exit(1)
    
    # Define the namespace used in the PAGE XML. This is needed to find the elements.
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Define a mapping of region types to colors (as basic color names).
    type_colors = {
        'header': 'red',
        'paragraph': 'blue',
        'catch-word': 'green',
        'page-number': 'yellow',
        'marginalia': 'purple',
        'signature-mark': 'orange'
    }
    default_color = 'white'  # Color to use if no type is found or type is unrecognized.

    # Iterate over each <TextRegion> element in the XML.
    # The XPath './/pc:TextRegion' finds all TextRegion elements in the document.
    for text_region in root.findall('.//pc:TextRegion', ns):
        # Retrieve the 'custom' attribute which may contain region type info.
        custom_attr = text_region.get('custom', '')
        region_type = extract_region_type(custom_attr) if custom_attr else None
        base_color = type_colors.get(region_type, default_color)

        # Convert the base color name to an RGB tuple.
        try:
            rgb = ImageColor.getrgb(base_color)
        except Exception:
            rgb = (255, 255, 255)  # fallback to white if conversion fails

        # Set fill color with transparency (alpha value 100 out of 255) and full opaque outline.
        fill_color = (rgb[0], rgb[1], rgb[2], 100)
        outline_color = (rgb[0], rgb[1], rgb[2], 255)

        # Find the <Coords> element within this TextRegion.
        coords_elem = text_region.find('.//pc:Coords', ns)
        if coords_elem is not None:
            points_str = coords_elem.get('points', '')
            points = parse_coords(points_str)
            
            if points:
                # Draw the filled polygon with the semi-transparent color and the opaque outline.
                draw_overlay.polygon(points, fill=fill_color, outline=outline_color)
                # Optionally, write the region type text near the first point.
                if region_type:
                    draw_overlay.text(points[0], region_type, fill=outline_color)
    
    # Composite the overlay onto the original image.
    result_image = Image.alpha_composite(image, overlay)
    # Convert back to RGB before saving as JPEG (which doesn't support alpha channels).
    result_image = result_image.convert("RGB")

    # Save the image with overlays.
    output_filename = base_name + "_overlay.jpg"
    output_path = os.path.join("/Users/gavinl/Desktop/Inventory 1120/Inventory 1120 Overlays", output_filename)
    try:
        result_image.save(output_path)
        print(f"Overlay image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    # The script expects one argument: the base filename (without extension).
    if len(sys.argv) != 2:
        print("Usage: python script.py <base_filename_without_extension>")
        sys.exit(1)
    
    base_name = sys.argv[1]
    main(base_name)