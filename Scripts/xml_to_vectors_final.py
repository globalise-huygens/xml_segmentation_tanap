import pandas as pd
import xml.etree.ElementTree as ET
import re
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

def create_document_structure_features() -> Dict[str, Any]:
    """Create specialized features for document structure classification"""
    features = {
        # General emptiness indicators (for "Outside" classification)
        'has_any_regions': 0,  # Binary: 1 if any regions exist, 0 if completely empty
        'total_region_count': 0,  # Count of all regions on the page
        'content_density_score': 0,  # Ratio of filled area to total page area
        
        # Document start indicators
        'has_header_region': 0,  # Binary: 1 if header region exists
        'header_size_ratio': 0,  # Size of header relative to page (larger = more likely a title)
        'header_top_position': 0,  # Vertical position of header (0=top of page)
        'prev_page_has_signature': 0,  # Binary: 1 if previous page has signature
        'prev_page_emptiness': 0,  # How empty the previous page is (0-1)
        'content_increase_from_prev': 0,  # Content increase from previous page
        'first_content_vertical_position': 0,  # Where content starts on page (normalized 0-1)
        
        # Middle page indicators
        'paragraph_to_region_ratio': 0,  # Ratio of paragraphs to total regions
        'has_no_signature': 0,  # Binary: 1 if no signature present
        'has_small_header_only': 0,  # Binary: 1 if only small headers exist
        'surrounded_by_content': 0,  # Binary: 1 if both prev and next pages have content
        'content_continuity_score': 0,  # Measure of text flow from previous page
        'consistent_layout_with_neighbors': 0,  # Layout similarity with adjacent pages
        
        # End page indicators
        'has_signature': 0,  # Binary: 1 if signature mark is present
        'signature_at_bottom': 0,  # Binary: 1 if signature is at bottom of page
        'next_page_emptiness': 0,  # How empty the next page is (0-1)
        'content_decrease_to_next': 0,  # Content decrease to next page
        'bottom_region_is_final': 0,  # Features suggesting bottom content completes a section
        
        # Transition indicators
        'is_structural_boundary': 0,  # Indicates a major change in document structure
        'content_discontinuity_score': 0,  # Measures breaks in content flow
        
        # Relative page features
        'regions_vs_document_avg': 0,  # Page's region count relative to document average
        'is_outlier_in_sequence': 0  # Whether page has unusual characteristics in sequence
    }
    return features

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_region_info(tr, coords_elem, page_width, page_height, page_area, ns):
    """Extract region information including coordinates, size, and text content"""
    region_info = {}
    
    points_str = coords_elem.get('points', '')
    if not points_str:
        return None
        
    # Extract points
    try:
        points = []
        for pair_str in points_str.split():
            if ',' not in pair_str:
                continue
            x_str, y_str = pair_str.split(',', 1)
            try:
                x_val = float(x_str)
                y_val = float(y_str)
                points.append((x_val, y_val))
            except ValueError:
                continue
            
        if not points:
            return None
            
        # Calculate bounding box
        all_x = [p[0] for p in points]
        all_y = [p[1] for p in points]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        
        # Calculate center point
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Normalize if page dimensions exist
        if page_width > 0 and page_height > 0:
            x_norm = min_x / page_width
            y_norm = min_y / page_height
            w_norm = width / page_width
            h_norm = height / page_height
            area_norm = area / page_area
            center_x_norm = center_x / page_width
            center_y_norm = center_y / page_height
        else:
            x_norm, y_norm = min_x, min_y
            w_norm, h_norm = width, height
            area_norm = area
            center_x_norm, center_y_norm = center_x, center_y
        
        # Extract text content
        text_content = ""
        text_lines = []
        
        for text_line in tr.findall('.//page:TextLine', ns):
            line_text = ""
            for word_elem in text_line.findall('.//page:Word', ns):
                text_equiv = word_elem.find('.//page:TextEquiv', ns)
                if text_equiv is not None:
                    unicode_elem = text_equiv.find('.//page:Unicode', ns)
                    if unicode_elem is not None and unicode_elem.text:
                        if line_text:
                            line_text += " "
                        line_text += unicode_elem.text
            
            if line_text:
                text_lines.append(line_text)
                
        if text_lines:
            text_content = "\n".join(text_lines)
        
        # Store region information
        region_info = {
            'x': min_x, 'y': min_y,
            'width': width, 'height': height,
            'area': area,
            'center_x': center_x, 'center_y': center_y,
            'x_norm': x_norm, 'y_norm': y_norm,
            'w_norm': w_norm, 'h_norm': h_norm,
            'area_norm': area_norm,
            'center_x_norm': center_x_norm, 'center_y_norm': center_y_norm,
            'aspect_ratio': width / height if height else 0,
            'text_content': text_content,
            'text_length': len(text_content),
            'line_count': len(text_lines),
            'char_density': len(text_content) / area if area else 0,
            'page_width': page_width,
            'page_height': page_height
        }
        
        return region_info
    
    except Exception as e:
        print(f"Warning: Error extracting region info: {e}")
        return None

def calculate_layout_similarity(page1_regions, page2_regions):
    """Calculate a similarity score between the layouts of two pages"""
    if not page1_regions or not page2_regions:
        return 0
    
    # Create grid representation for each page (5x5 grid)
    grid_size = 5
    grid1 = np.zeros((grid_size, grid_size))
    grid2 = np.zeros((grid_size, grid_size))
    
    def populate_grid(grid, regions):
        for region in regions:
            # Determine grid cells that this region covers
            x_start = min(int(region['x_norm'] * grid_size), grid_size - 1)
            y_start = min(int(region['y_norm'] * grid_size), grid_size - 1)
            x_end = min(int((region['x_norm'] + region['w_norm']) * grid_size), grid_size - 1)
            y_end = min(int((region['y_norm'] + region['h_norm']) * grid_size), grid_size - 1)
            
            # Mark grid cells
            for i in range(y_start, y_end + 1):
                for j in range(x_start, x_end + 1):
                    grid[i, j] = 1
    
    # Populate grids
    all_regions1 = []
    all_regions2 = []
    
    for regions in page1_regions.values():
        all_regions1.extend(regions)
    
    for regions in page2_regions.values():
        all_regions2.extend(regions)
    
    populate_grid(grid1, all_regions1)
    populate_grid(grid2, all_regions2)
    
    # Calculate similarity (intersection over union)
    intersection = np.sum(np.logical_and(grid1, grid2))
    union = np.sum(np.logical_or(grid1, grid2))
    
    return intersection / union if union > 0 else 0

def calculate_page_content_score(regions_by_type):
    """Calculate a score representing the amount of content on a page"""
    total_area = 0
    total_text_length = 0
    
    for region_type, regions in regions_by_type.items():
        for region in regions:
            total_area += region.get('area_norm', 0)
            total_text_length += region.get('text_length', 0)
    
    # Combine area and text length for a comprehensive content score
    # We normalize text length by dividing by 1000 to keep it in a similar range to area_norm
    return total_area + (total_text_length / 1000.0)

def parse_pagexml_to_features(xml_string: str, prev_page_info=None, next_page_info=None, 
                              page_position=None, doc_avg_regions=None) -> Dict[str, Any]:
    """
    Parses PAGE XML and extracts features specifically for document structure classification.
    
    Args:
        xml_string: The PAGE XML content as a string
        prev_page_info: Information about the previous page (if available)
        next_page_info: Information about the next page (if available)
        page_position: Dictionary with page position info
        doc_avg_regions: Average number of regions across the document
        
    Returns:
        Dictionary containing document structure classification features
    """
    # Initialize with empty features
    features = create_document_structure_features()
    
    # If empty or invalid input, return empty dictionary with "Outside" indicators
    if not isinstance(xml_string, str) or not xml_string.strip():
        # An empty XML string is likely an outside page
        features['is_outlier_in_sequence'] = 1
        return features

    # Attempt to parse XML
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        print("Warning: Failed to parse XML")
        features['is_outlier_in_sequence'] = 1
        return features

    # Namespace for PAGE XML
    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Region types we care about
    region_types = ["paragraph", "header", "page-number", "marginalia", "signature-mark", "catch-word"]
    
    # Find the <Page> element
    page_elem = root.find('.//page:Page', ns)
    if page_elem is None:
        features['is_outlier_in_sequence'] = 1
        return features  # No page element found

    # Read page dimensions
    try:
        page_width = float(page_elem.get('imageWidth', '0'))
        page_height = float(page_elem.get('imageHeight', '0'))
        page_area = page_width * page_height if page_width and page_height else 0
    except ValueError:
        page_width, page_height, page_area = 0, 0, 0
    
    # Initialize data structures for region information
    regions_by_type = {rt: [] for rt in region_types}
    
    # Find all <TextRegion> elements
    text_regions = page_elem.findall('.//page:TextRegion', ns)
    
    # Track the total content on the page
    total_region_count = len(text_regions)
    features['total_region_count'] = total_region_count
    features['has_any_regions'] = 1 if total_region_count > 0 else 0
    
    # Track content positions
    min_y_position = float('inf')
    total_region_area = 0
    
    # Process each text region
    for tr in text_regions:
        region_id = tr.get('id', '')
        custom_attr = tr.get('custom', '')
        
        # Extract region type
        region_type = None
        if 'type:' in custom_attr:
            match = re.search(r'type:([^;}\s]+)', custom_attr)
            if match:
                region_type = match.group(1).strip()
        
        if region_type not in region_types:
            continue
            
        # Extract region information
        coords_elem = tr.find('.//page:Coords', ns)
        if coords_elem is None:
            continue
            
        region_info = extract_region_info(tr, coords_elem, page_width, page_height, page_area, ns)
        if region_info is None:
            continue
        
        # Add region ID for reference
        region_info['id'] = region_id
        region_info['type'] = region_type
        
        # Track the top-most content position
        if region_info['y_norm'] < min_y_position:
            min_y_position = region_info['y_norm']
        
        # Track total content area
        total_region_area += region_info['area_norm']
        
        # Store region info
        regions_by_type[region_type].append(region_info)
    
    # Update emptiness indicators
    features['content_density_score'] = total_region_area
    features['first_content_vertical_position'] = min_y_position if min_y_position != float('inf') else 1.0
    
    # Extract header features
    header_regions = regions_by_type.get('header', [])
    features['has_header_region'] = 1 if header_regions else 0
    
    if header_regions:
        header_areas = [r['area_norm'] for r in header_regions]
        largest_header_area = max(header_areas)
        features['header_size_ratio'] = largest_header_area / total_region_area if total_region_area > 0 else 0
        
        # Find the topmost header position
        header_top_positions = [r['y_norm'] for r in header_regions]
        features['header_top_position'] = min(header_top_positions)
        
        # Check if all headers are small (for middle page detection)
        avg_header_size = sum(header_areas) / len(header_areas)
        features['has_small_header_only'] = 1 if avg_header_size < 0.05 else 0  # Threshold for "small" header
    
    # Extract paragraph features
    paragraph_regions = regions_by_type.get('paragraph', [])
    features['paragraph_to_region_ratio'] = len(paragraph_regions) / total_region_count if total_region_count > 0 else 0
    
    # Extract signature features
    signature_regions = regions_by_type.get('signature-mark', [])
    features['has_signature'] = 1 if signature_regions else 0
    features['has_no_signature'] = 1 if not signature_regions else 0
    
    if signature_regions:
        # Check if signature is at the bottom of the page
        signature_y_positions = [r['y_norm'] + r['h_norm'] for r in signature_regions]  # Bottom edge of signature
        max_signature_position = max(signature_y_positions)
        features['signature_at_bottom'] = 1 if max_signature_position > 0.8 else 0  # Bottom 20% of page
    
    # Extract page context features
    if prev_page_info:
        # Check for signature in previous page
        prev_signatures = prev_page_info.get('signature-mark', [])
        features['prev_page_has_signature'] = 1 if prev_signatures else 0
        
        # Calculate previous page emptiness
        prev_total_area = 0
        for regions in prev_page_info.values():
            prev_total_area += sum(r.get('area_norm', 0) for r in regions)
        
        features['prev_page_emptiness'] = 1 - prev_total_area  # 1 means completely empty
        
        # Calculate content continuity between pages
        # Compare content amount and distribution between pages
        prev_content_score = calculate_page_content_score(prev_page_info)
        current_content_score = calculate_page_content_score(regions_by_type)
        
        # For content change calculations
        if prev_content_score > 0:
            change_ratio = (current_content_score - prev_content_score) / prev_content_score
            features['content_increase_from_prev'] = max(0, change_ratio)  # Only positive changes
        else:
            # If previous page was empty, this is a big increase
            features['content_increase_from_prev'] = 1 if current_content_score > 0 else 0
        
        # Calculate layout similarity for continuity
        layout_similarity = calculate_layout_similarity(regions_by_type, prev_page_info)
        features['content_continuity_score'] = layout_similarity
    
    if next_page_info:
        # Calculate next page emptiness
        next_total_area = 0
        for regions in next_page_info.values():
            next_total_area += sum(r.get('area_norm', 0) for r in regions)
        
        features['next_page_emptiness'] = 1 - next_total_area  # 1 means completely empty
        
        # Calculate content decrease to next page
        next_content_score = calculate_page_content_score(next_page_info)
        current_content_score = calculate_page_content_score(regions_by_type)
        
        if current_content_score > 0:
            change_ratio = (next_content_score - current_content_score) / current_content_score
            features['content_decrease_to_next'] = max(0, -change_ratio)  # Only negative changes (make positive)
        else:
            features['content_decrease_to_next'] = 0
    
    # Check if surrounded by content (middle page indicator)
    if prev_page_info and next_page_info:
        prev_has_content = any(len(regions) > 0 for regions in prev_page_info.values())
        next_has_content = any(len(regions) > 0 for regions in next_page_info.values())
        features['surrounded_by_content'] = 1 if prev_has_content and next_has_content else 0
        
        # Calculate consistency with neighboring pages
        prev_layout_similarity = calculate_layout_similarity(regions_by_type, prev_page_info)
        next_layout_similarity = calculate_layout_similarity(regions_by_type, next_page_info)
        avg_similarity = (prev_layout_similarity + next_layout_similarity) / 2
        features['consistent_layout_with_neighbors'] = avg_similarity
    
    # Calculate structural boundary features
    if prev_page_info:
        # A structural boundary is indicated by a significant change in layout or content
        prev_layout_similarity = calculate_layout_similarity(regions_by_type, prev_page_info)
        features['is_structural_boundary'] = 1 if prev_layout_similarity < 0.3 else 0  # Low similarity = boundary
        features['content_discontinuity_score'] = 1 - prev_layout_similarity
    
    # Check if the bottom region completes a section (end page indicator)
    all_regions = []
    for regions in regions_by_type.values():
        all_regions.extend(regions)
    
    if all_regions:
        # Sort regions by vertical position (bottom to top)
        sorted_regions = sorted(all_regions, key=lambda r: -(r['y_norm'] + r['h_norm']))
        
        # Check if the bottommost region is a signature, catch-word, or similar
        bottommost_type = sorted_regions[0]['type']
        features['bottom_region_is_final'] = 1 if bottommost_type in ['signature-mark', 'catch-word'] else 0
    
    # Calculate relative metrics if document average is provided
    if doc_avg_regions is not None and doc_avg_regions > 0:
        features['regions_vs_document_avg'] = total_region_count / doc_avg_regions
        
        # A page with unusually few or many regions compared to document average is an outlier
        ratio = total_region_count / doc_avg_regions
        features['is_outlier_in_sequence'] = 1 if ratio < 0.3 or ratio > 2.0 else 0
    
    return features

def process_dataset(input_csv_path: str, xml_column: str, output_csv_path: str, verbose: bool = True):
    """
    Process a dataset containing PAGE XML data and extract document structure features
    
    Args:
        input_csv_path: Path to input CSV
        xml_column: Name of column containing the XML data
        output_csv_path: Path to save the output CSV
        verbose: Whether to print progress information
    """
    # Read the input CSV
    if verbose:
        print(f"Reading input CSV: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path)
    
    if xml_column not in df.columns:
        raise ValueError(f"XML column '{xml_column}' not found in the CSV file.")
    
    # Get total row count for progress tracking
    total_rows = len(df)
    if verbose:
        print(f"Processing {total_rows} rows...")
    
    # First pass: extract basic region information for page context and calculate document averages
    if verbose:
        print("First pass: Extracting basic region information for document context...")
    
    # Store region info for all pages to use for context
    page_info = []
    total_regions_in_document = 0
    
    for idx, xml_str in enumerate(df[xml_column]):
        if verbose and idx % 100 == 0:
            print(f"First pass: Processing row {idx+1}/{total_rows} ({idx/total_rows*100:.1f}%)")
        
        # Parse XML but only extract region info, not full features
        try:
            root = ET.fromstring(xml_str)
            ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
            
            page_elem = root.find('.//page:Page', ns)
            if page_elem is None:
                page_info.append({})
                continue
                
            # Read page dimensions
            try:
                page_width = float(page_elem.get('imageWidth', '0'))
                page_height = float(page_elem.get('imageHeight', '0'))
                page_area = page_width * page_height if page_width and page_height else 0
            except ValueError:
                page_width, page_height, page_area = 0, 0, 0
            
            # Initialize data structures for region information
            region_types = ["paragraph", "header", "page-number", "marginalia", "signature-mark", "catch-word"]
            regions_by_type = {rt: [] for rt in region_types}
            
            # Find all <TextRegion> elements
            text_regions = page_elem.findall('.//page:TextRegion', ns)
            total_regions_in_document += len(text_regions)
            
            # Process each text region
            for tr in text_regions:
                custom_attr = tr.get('custom', '')
                
                # Extract region type
                region_type = None
                if 'type:' in custom_attr:
                    match = re.search(r'type:([^;}\s]+)', custom_attr)
                    if match:
                        region_type = match.group(1).strip()
                
                if region_type not in region_types:
                    continue
                    
                # Extract region information (just coordinates)
                coords_elem = tr.find('.//page:Coords', ns)
                if coords_elem is None:
                    continue
                    
                region_info = extract_region_info(tr, coords_elem, page_width, page_height, page_area, ns)
                if region_info is None:
                    continue
                
                # Store region info
                regions_by_type[region_type].append(region_info)
            
            page_info.append(regions_by_type)
            
        except Exception as e:
            print(f"Warning: Error processing page {idx} for context: {e}")
            page_info.append({})
    
    # Calculate document average regions per page
    doc_avg_regions = total_regions_in_document / total_rows if total_rows > 0 else 0
    
    # Second pass: extract full features with page context
    if verbose:
        print("Second pass: Extracting document structure features...")
        print(f"Document average regions per page: {doc_avg_regions:.2f}")
    
    features_list = []
    
    for idx, xml_str in enumerate(df[xml_column]):
        if verbose and idx % 100 == 0:
            print(f"Second pass: Processing row {idx+1}/{total_rows} ({idx/total_rows*100:.1f}%)")
        
        # Get previous and next page info for context
        prev_page = page_info[idx-1] if idx > 0 else None
        next_page = page_info[idx+1] if idx < len(page_info) - 1 else None
        
        # Page position info
        page_position = {
            'is_first_page': 1 if idx == 0 else 0,
            'is_last_page': 1 if idx == len(page_info) - 1 else 0
        }
        
        # Extract features with page context
        features = parse_pagexml_to_features(
            xml_str, 
            prev_page_info=prev_page,
            next_page_info=next_page,
            page_position=page_position,
            doc_avg_regions=doc_avg_regions
        )
        
        features_list.append(features)
    
    # Convert features to DataFrame
    features_df = pd.json_normalize(features_list)
    
    # Combine original DataFrame (without XML column) with features
    result_df = df.drop(columns=[xml_column]).reset_index(drop=True)
    result_df = pd.concat([result_df, features_df], axis=1)
    
    # Save to output CSV
    result_df.to_csv(output_csv_path, index=False)
    
    if verbose:
        print(f"Finished processing. Output saved to: {output_csv_path}")
        print(f"Generated {len(features_df.columns)} feature columns")
        
        # Print summary of extracted features
        if not features_df.empty:
            print("\nFeature summary:")
            print("================")
            
            print("\nGeneral emptiness indicators:")
            print(f"Has any regions (average): {features_df['has_any_regions'].mean():.2f}")
            print(f"Total region count (average): {features_df['total_region_count'].mean():.2f}")
            print(f"Content density score (average): {features_df['content_density_score'].mean():.2f}")
            
            print("\nDocument start indicators:")
            print(f"Has header region (average): {features_df['has_header_region'].mean():.2f}")
            print(f"Header size ratio (average): {features_df['header_size_ratio'].mean():.2f}")
            print(f"Content increase from prev (average): {features_df['content_increase_from_prev'].mean():.2f}")
            
            print("\nMiddle page indicators:")
            print(f"Paragraph to region ratio (average): {features_df['paragraph_to_region_ratio'].mean():.2f}")
            print(f"Has no signature (average): {features_df['has_no_signature'].mean():.2f}")
            print(f"Surrounded by content (average): {features_df['surrounded_by_content'].mean():.2f}")
            
            print("\nEnd page indicators:")
            print(f"Has signature (average): {features_df['has_signature'].mean():.2f}")
            print(f"Signature at bottom (average): {features_df['signature_at_bottom'].mean():.2f}")
            print(f"Next page emptiness (average): {features_df['next_page_emptiness'].mean():.2f}")
            
            print("\nTransition indicators:")
            print(f"Is structural boundary (average): {features_df['is_structural_boundary'].mean():.2f}")
            print(f"Content discontinuity score (average): {features_df['content_discontinuity_score'].mean():.2f}")
            
            print("\nTop correlations with 'output' (if present):")
            if 'output' in result_df.columns:
                try:
                    correlations = features_df.corrwith(result_df['output'].astype(float)).abs().sort_values(ascending=False)
                    print(correlations.head(10))  # Show top 10 correlations
                except:
                    print("Could not calculate correlations (output may not be numeric)")
        else:
            print("Warning: No features were extracted. Check for parsing issues.")

if __name__ == "__main__":
    # Example usage - update these paths to match your environment
    input_csv = "/Users/gavinl/Desktop/TANAP Segmentation/Data/Testing/test_set_xml.csv"
    output_csv = "/Users/gavinl/Desktop/TANAP Segmentation/Data/Testing/test_set_features.csv"
    xml_col = "input"  # Name of the column containing XML strings
    
    process_dataset(input_csv, xml_col, output_csv)