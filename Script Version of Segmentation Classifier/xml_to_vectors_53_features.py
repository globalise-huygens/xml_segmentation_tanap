import pandas as pd
import xml.etree.ElementTree as ET
import re
import math
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict

def create_document_structure_features() -> Dict[str, Any]:
    """Initialize feature dictionary with original and new features for a page."""
    features = {
        # Original features (document structure indicators)
        'has_any_regions': 0,
        'total_region_count': 0,
        'content_density_score': 0,
        'has_header_region': 0,
        'header_size_ratio': 0,
        'header_top_position': 0,
        'prev_page_has_signature': 0,
        'prev_page_emptiness': 0,
        'content_increase_from_prev': 0,
        'first_content_vertical_position': 0,
        'paragraph_to_region_ratio': 0,
        'has_no_signature': 0,
        'has_small_header_only': 0,
        'surrounded_by_content': 0,
        'content_continuity_score': 0,
        'consistent_layout_with_neighbors': 0,
        'has_signature': 0,
        'signature_at_bottom': 0,
        'next_page_emptiness': 0,
        'content_decrease_to_next': 0,
        'bottom_region_is_final': 0,
        'is_structural_boundary': 0,
        'content_discontinuity_score': 0,
        'regions_vs_document_avg': 0,
        'is_outlier_in_sequence': 0,
        # New features – text volume and density
        'total_text_length': 0,
        'total_word_count': 0,
        'total_line_count': 0,
        'avg_region_text_length': 0,
        'avg_region_word_count': 0,
        'avg_region_line_count': 0,
        'page_char_density': 0,            # chars per full page area
        'avg_char_density_per_region': 0,  # avg char density across regions
        'max_char_density': 0,
        # New features – signature and region counts
        'signature_count': 0,
        'signature_area_ratio': 0,
        'next_page_has_signature': 0,
        'paragraph_count': 0,
        'page_number_count': 0,
        'catch_word_count': 0,
        # New features – 3-page sequence context
        'prev3_region_count_avg': 0,
        'next3_region_count_avg': 0,
        'prev3_content_score_avg': 0,
        'next3_content_score_avg': 0,
        'prev3_content_pages_count': 0,
        'next3_content_pages_count': 0,
        'region_count_change_prev3': 0,
        'region_count_change_next3': 0,
        'content_score_change_prev3': 0,
        'content_score_change_next3': 0,
        # New features – position in 7-page window and doc
        'position_in_7page_window': 0,
        'is_first_page': 0,
        'is_last_page': 0
    }
    return features

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_region_info(tr_elem, coords_elem, page_width, page_height, page_area, ns) -> Optional[Dict[str, Any]]:
    """Extract region information including coordinates, size, and text content for a TextRegion element."""
    points_str = coords_elem.get('points', '')
    if not points_str:
        return None
    try:
        # Parse polygon coordinates of region
        points = []
        for pair_str in points_str.split():
            if ',' not in pair_str:
                continue
            x_str, y_str = pair_str.split(',', 1)
            try:
                x_val = float(x_str); y_val = float(y_str)
                points.append((x_val, y_val))
            except ValueError:
                continue
        if not points:
            return None

        # Calculate bounding box and region area
        all_x = [p[0] for p in points]; all_y = [p[1] for p in points]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        width = max_x - min_x; height = max_y - min_y
        area = width * height

        # Calculate center point
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Normalize coordinates and area if page dimensions are known
        if page_width > 0 and page_height > 0:
            x_norm = min_x / page_width
            y_norm = min_y / page_height
            w_norm = width / page_width
            h_norm = height / page_height
            area_norm = area / page_area if page_area else 0
            center_x_norm = center_x / page_width
            center_y_norm = center_y / page_height
        else:
            x_norm = min_x; y_norm = min_y
            w_norm = width; h_norm = height
            area_norm = area
            center_x_norm = center_x; center_y_norm = center_y

        # Extract text content from all lines and words in this region
        text_lines = []
        word_count = 0
        for text_line in tr_elem.findall('.//page:TextLine', ns):
            line_text = ""
            for word_elem in text_line.findall('.//page:Word', ns):
                text_equiv = word_elem.find('.//page:TextEquiv', ns)
                if text_equiv is not None:
                    unicode_elem = text_equiv.find('.//page:Unicode', ns)
                    if unicode_elem is not None and unicode_elem.text:
                        # Append word text
                        if line_text:
                            line_text += " "
                        line_text += unicode_elem.text
                        word_count += 1  # count each word encountered
            if line_text:
                text_lines.append(line_text)
        text_content = "\n".join(text_lines) if text_lines else ""
        text_length = len(text_content)
        line_count = len(text_lines)

        # Compute character density for this region (chars per region area)
        char_density = (text_length / area) if area > 0 else 0

        # Compile region info dictionary
        region_info = {
            'x': min_x, 'y': min_y,
            'width': width, 'height': height,
            'area': area,
            'center_x': center_x, 'center_y': center_y,
            'x_norm': x_norm, 'y_norm': y_norm,
            'w_norm': w_norm, 'h_norm': h_norm,
            'area_norm': area_norm,
            'center_x_norm': center_x_norm, 'center_y_norm': center_y_norm,
            'aspect_ratio': (width / height) if height else 0,
            'text_content': text_content,
            'text_length': text_length,
            'word_count': word_count,
            'line_count': line_count,
            'char_density': char_density,
            'page_width': page_width,
            'page_height': page_height
        }
        return region_info
    except Exception as e:
        print(f"Warning: Error extracting region info: {e}")
        return None

def calculate_layout_similarity(page1_regions: Dict[str, List[Dict]], page2_regions: Dict[str, List[Dict]]) -> float:
    """Calculate a similarity score between the layouts of two pages based on region positions."""
    if not page1_regions or not page2_regions:
        return 0.0
    grid_size = 5
    grid1 = np.zeros((grid_size, grid_size))
    grid2 = np.zeros((grid_size, grid_size))
    # Helper to mark grid cells covered by regions
    def populate_grid(grid, regions):
        for region in regions:
            x_start = min(int(region['x_norm'] * grid_size), grid_size - 1)
            y_start = min(int(region['y_norm'] * grid_size), grid_size - 1)
            x_end = min(int((region['x_norm'] + region['w_norm']) * grid_size), grid_size - 1)
            y_end = min(int((region['y_norm'] + region['h_norm']) * grid_size), grid_size - 1)
            for i in range(y_start, y_end + 1):
                for j in range(x_start, x_end + 1):
                    grid[i, j] = 1
    # Flatten all regions from type dicts
    all_regions1 = [r for regions in page1_regions.values() for r in regions]
    all_regions2 = [r for regions in page2_regions.values() for r in regions]
    populate_grid(grid1, all_regions1)
    populate_grid(grid2, all_regions2)
    # Compute Intersection over Union of occupied grid cells
    intersection = np.sum(np.logical_and(grid1, grid2))
    union = np.sum(np.logical_or(grid1, grid2))
    return (intersection / union) if union > 0 else 0.0

def calculate_page_content_score(regions_by_type: Dict[str, List[Dict]]) -> float:
    """Calculate a composite score representing the amount of content on a page (area + text)."""
    total_area_norm = 0.0
    total_text_length = 0
    for region_list in regions_by_type.values():
        for region in region_list:
            total_area_norm += region.get('area_norm', 0)
            total_text_length += region.get('text_length', 0)
    # Combine normalized area and text length (scaled down) for a comprehensive content measure
    return total_area_norm + (total_text_length / 1000.0)

def parse_pagexml_to_features(xml_string: str,
                              prev_page_info: Optional[Dict[str, List[Dict]]] = None,
                              next_page_info: Optional[Dict[str, List[Dict]]] = None,
                              page_position: Optional[Dict[str, int]] = None,
                              doc_avg_regions: Optional[float] = None) -> Dict[str, Any]:
    """
    Parse a PAGE XML string and extract document-structure features for the page.
    """
    features = create_document_structure_features()
    # Handle empty or invalid XML as an outside page
    if not isinstance(xml_string, str) or not xml_string.strip():
        features['is_outlier_in_sequence'] = 1
        return features
    # Parse the XML content
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        print("Warning: Failed to parse XML.")
        features['is_outlier_in_sequence'] = 1
        return features

    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    page_elem = root.find('.//page:Page', ns)
    if page_elem is None:
        features['is_outlier_in_sequence'] = 1
        return features

    # Page dimensions
    try:
        page_width = float(page_elem.get('imageWidth', '0'))
        page_height = float(page_elem.get('imageHeight', '0'))
        page_area = page_width * page_height if page_width and page_height else 0
    except ValueError:
        page_width, page_height, page_area = 0, 0, 0

    # Initialize storage for region info by type
    region_types = ["paragraph", "header", "page-number", "marginalia", "signature-mark", "catch-word"]
    regions_by_type = {rt: [] for rt in region_types}

    text_regions = page_elem.findall('.//page:TextRegion', ns)
    total_region_count = len(text_regions)
    features['total_region_count'] = total_region_count
    features['has_any_regions'] = 1 if total_region_count > 0 else 0

    # Track first content position and cumulative area
    min_y_position = float('inf')
    total_region_area_norm = 0.0
    # Track cumulative text, word, line counts and char density extents
    total_text_length = 0
    total_word_count = 0
    total_line_count = 0
    sum_char_density = 0.0
    max_char_density = 0.0

    # Process each text region
    for tr in text_regions:
        region_id = tr.get('id', '')
        custom_attr = tr.get('custom', '')
        # Identify region type from custom attributes
        region_type = None
        if 'type:' in custom_attr:
            match = re.search(r'type:([^;}\s]+)', custom_attr)
            if match:
                region_type = match.group(1).strip()
        if region_type not in region_types:
            continue  # skip regions of types not in our list

        coords_elem = tr.find('.//page:Coords', ns)
        if coords_elem is None:
            continue
        region_info = extract_region_info(tr, coords_elem, page_width, page_height, page_area, ns)
        if region_info is None:
            continue
        # Label region with ID and type
        region_info['id'] = region_id
        region_info['type'] = region_type

        # Track topmost content position (normalize to [0,1], 0 = top of page)
        if region_info.get('y_norm', 1.0) < min_y_position:
            min_y_position = region_info['y_norm']
        # Accumulate total normalized content area
        total_region_area_norm += region_info.get('area_norm', 0)

        # Collect region info into our structure
        regions_by_type[region_type].append(region_info)
        # Accumulate text, word, line totals
        total_text_length += region_info.get('text_length', 0)
        total_word_count += region_info.get('word_count', 0)
        total_line_count += region_info.get('line_count', 0)
        # Accumulate char density stats
        cd = region_info.get('char_density', 0.0)
        sum_char_density += cd
        if cd > max_char_density:
            max_char_density = cd

    # After processing regions, fill in features
    features['content_density_score'] = total_region_area_norm  # fraction of page area filled
    features['first_content_vertical_position'] = (min_y_position if min_y_position != float('inf') else 1.0)
    # Header-related
    header_regions = regions_by_type.get('header', [])
    features['has_header_region'] = 1 if header_regions else 0
    if header_regions:
        header_areas = [r['area_norm'] for r in header_regions]
        largest_header_area = max(header_areas)
        features['header_size_ratio'] = largest_header_area / total_region_area_norm if total_region_area_norm > 0 else 0
        # Topmost header position
        features['header_top_position'] = min(r['y_norm'] for r in header_regions)
        # If all headers are very small relative to page (e.g., running headers)
        avg_header_area = sum(header_areas) / len(header_areas)
        features['has_small_header_only'] = 1 if avg_header_area < 0.05 else 0
    # Paragraph-related
    paragraph_regions = regions_by_type.get('paragraph', [])
    features['paragraph_to_region_ratio'] = (len(paragraph_regions) / total_region_count) if total_region_count > 0 else 0
    # Signature-related
    signature_regions = regions_by_type.get('signature-mark', [])
    features['has_signature'] = 1 if signature_regions else 0
    features['has_no_signature'] = 1 if not signature_regions else 0
    if signature_regions:
        # Check if any signature region extends into bottom 20% of page
        bottom_positions = [(r['y_norm'] + r['h_norm']) for r in signature_regions]
        if bottom_positions and max(bottom_positions) > 0.8:
            features['signature_at_bottom'] = 1
        # Compute signature area ratio relative to total content area
        total_signature_area = sum(r['area_norm'] for r in signature_regions)
        features['signature_area_ratio'] = total_signature_area / total_region_area_norm if total_region_area_norm > 0 else 0
    # Page context (previous page features)
    if prev_page_info is not None:
        prev_signatures = prev_page_info.get('signature-mark', [])
        features['prev_page_has_signature'] = 1 if prev_signatures else 0
        # Previous page emptiness: 1 means completely empty
        prev_total_area = 0.0
        for regions in prev_page_info.values():
            prev_total_area += sum(r.get('area_norm', 0) for r in regions)
        features['prev_page_emptiness'] = 1 - prev_total_area
        # Content increase from previous page (relative change in content score)
        prev_content_score = calculate_page_content_score(prev_page_info)
        curr_content_score = calculate_page_content_score(regions_by_type)
        if prev_content_score > 0:
            change_ratio = (curr_content_score - prev_content_score) / prev_content_score
            features['content_increase_from_prev'] = max(0, change_ratio)
        else:
            features['content_increase_from_prev'] = 1 if curr_content_score > 0 else 0
        # Layout continuity with previous page
        features['content_continuity_score'] = calculate_layout_similarity(regions_by_type, prev_page_info)
    if next_page_info is not None:
        next_total_area = 0.0
        for regions in next_page_info.values():
            next_total_area += sum(r.get('area_norm', 0) for r in regions)
        features['next_page_emptiness'] = 1 - next_total_area
        next_content_score = calculate_page_content_score(next_page_info)
        curr_content_score = calculate_page_content_score(regions_by_type)
        if curr_content_score > 0:
            change_ratio = (next_content_score - curr_content_score) / curr_content_score
            features['content_decrease_to_next'] = max(0, -change_ratio)
        else:
            features['content_decrease_to_next'] = 0
        # New: check for signature on the next page
        next_signatures = next_page_info.get('signature-mark', [])
        features['next_page_has_signature'] = 1 if next_signatures else 0
    # Surrounded by content (both neighbors have content)
    if prev_page_info is not None and next_page_info is not None:
        prev_has_content = any(len(reg_list) > 0 for reg_list in prev_page_info.values())
        next_has_content = any(len(reg_list) > 0 for reg_list in next_page_info.values())
        features['surrounded_by_content'] = 1 if (prev_has_content and next_has_content) else 0
        # Layout consistency with neighbors (average similarity)
        prev_layout_sim = calculate_layout_similarity(regions_by_type, prev_page_info)
        next_layout_sim = calculate_layout_similarity(regions_by_type, next_page_info)
        features['consistent_layout_with_neighbors'] = (prev_layout_sim + next_layout_sim) / 2
    # Structural boundary indicators
    if prev_page_info is not None:
        prev_layout_sim = calculate_layout_similarity(regions_by_type, prev_page_info)
        features['is_structural_boundary'] = 1 if prev_layout_sim < 0.3 else 0
        features['content_discontinuity_score'] = 1 - prev_layout_sim
    # Bottom region type check for final section indicators
    all_regions = [r for regions in regions_by_type.values() for r in regions]
    if all_regions:
        # Bottommost region = region with largest bottom y_norm coordinate
        bottommost_region = max(all_regions, key=lambda r: (r['y_norm'] + r['h_norm']))
        bottom_type = bottommost_region.get('type', '')
        features['bottom_region_is_final'] = 1 if bottom_type in ['signature-mark', 'catch-word'] else 0
    # Relative metrics w.r.t. document average
    if doc_avg_regions is not None and doc_avg_regions > 0:
        features['regions_vs_document_avg'] = total_region_count / doc_avg_regions
        ratio = total_region_count / doc_avg_regions
        if ratio < 0.3 or ratio > 2.0:
            features['is_outlier_in_sequence'] = 1

    # New: Fill in text volume and density features for the page
    features['total_text_length'] = total_text_length
    features['total_word_count'] = total_word_count
    features['total_line_count'] = total_line_count
    if total_region_count > 0:
        features['avg_region_text_length'] = total_text_length / total_region_count
        features['avg_region_word_count'] = total_word_count / total_region_count
        features['avg_region_line_count'] = total_line_count / total_region_count
        features['avg_char_density_per_region'] = sum_char_density / total_region_count
    # Character density for entire page (per full page area)
    features['page_char_density'] = (total_text_length / page_area) if page_area > 0 else 0
    features['max_char_density'] = max_char_density
    # New: Region type counts
    features['paragraph_count'] = len(paragraph_regions)
    features['page_number_count'] = len(regions_by_type.get('page-number', []))
    features['catch_word_count'] = len(regions_by_type.get('catch-word', []))
    features['signature_count'] = len(signature_regions)
    # New: Incorporate first/last page flags if provided
    if page_position:
        features['is_first_page'] = page_position.get('is_first_page', 0)
        features['is_last_page'] = page_position.get('is_last_page', 0)

    return features

def process_dataset(input_csv_path: str, xml_column: str, output_csv_path: str, verbose: bool = True):
    """
    Process a dataset containing PAGE XML data and extract extended features for each page.
    """
    if verbose:
        print(f"Reading input CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    if xml_column not in df.columns:
        raise ValueError(f"XML column '{xml_column}' not found in input CSV.")

    total_rows = len(df)
    if verbose:
        print(f"Processing {total_rows} pages...")

    # First pass: parse basic region info for all pages to use as context and compute document-wide stats
    page_info_list = []
    total_regions_all_pages = 0
    if verbose:
        print("First pass: Gathering region info for context...")
    for idx, xml_str in enumerate(df[xml_column]):
        if verbose and idx % 100 == 0:
            print(f"  Processed {idx}/{total_rows} pages for context")
        try:
            root = ET.fromstring(xml_str)
            ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
            page_elem = root.find('.//page:Page', ns)
            if page_elem is None:
                page_info_list.append({})
                continue
            # Get page regions by type (like in parse_pagexml but simpler, no neighbor context)
            page_width = float(page_elem.get('imageWidth', '0') or 0)
            page_height = float(page_elem.get('imageHeight', '0') or 0)
            page_area = page_width * page_height if page_width and page_height else 0
            region_types = ["paragraph", "header", "page-number", "marginalia", "signature-mark", "catch-word"]
            regions_by_type = {rt: [] for rt in region_types}
            text_regions = page_elem.findall('.//page:TextRegion', ns)
            total_regions_all_pages += len(text_regions)
            for tr in text_regions:
                custom_attr = tr.get('custom', '')
                region_type = None
                if 'type:' in custom_attr:
                    m = re.search(r'type:([^;}\s]+)', custom_attr)
                    if m:
                        region_type = m.group(1).strip()
                if region_type not in region_types:
                    continue
                coords_elem = tr.find('.//page:Coords', ns)
                if coords_elem is None:
                    continue
                region_info = extract_region_info(tr, coords_elem, page_width, page_height, page_area, ns)
                if region_info is None:
                    continue
                regions_by_type[region_type].append(region_info)
            page_info_list.append(regions_by_type)
        except Exception as e:
            print(f"Warning: Error processing page {idx} in context pass: {e}")
            page_info_list.append({})
    doc_avg_regions = total_regions_all_pages / total_rows if total_rows > 0 else 0

    # Second pass: extract full features for each page, using context from neighbors
    if verbose:
        print("Second pass: Extracting features for each page...")
        print(f"Average regions per page across document: {doc_avg_regions:.2f}")
    features_list = []
    for idx, xml_str in enumerate(df[xml_column]):
        if verbose and idx % 100 == 0:
            print(f"  Processed {idx}/{total_rows} pages for feature extraction")
        # Gather context info for neighbors (up to 3 pages on each side)
        prev_pages_context = page_info_list[max(0, idx-3) : idx]
        next_pages_context = page_info_list[idx+1 : idx+4]
        prev_page = page_info_list[idx-1] if idx > 0 else None
        next_page = page_info_list[idx+1] if idx < len(page_info_list) - 1 else None
        page_position = {
            'is_first_page': 1 if idx == 0 else 0,
            'is_last_page': 1 if idx == len(page_info_list) - 1 else 0
        }
        # Extract core features from the current page XML
        feat = parse_pagexml_to_features(xml_str,
                                         prev_page_info=prev_page,
                                         next_page_info=next_page,
                                         page_position=page_position,
                                         doc_avg_regions=doc_avg_regions)
        # Compute 3-page sequence features outside parse_pagexml
        # Compute average region count and content score for prev 3 and next 3 pages
        region_counts_prev = []
        content_scores_prev = []
        content_pages_prev = 0
        for pinfo in prev_pages_context:
            if pinfo and any(len(reg_list) > 0 for reg_list in pinfo.values()):
                # page has content
                region_counts_prev.append(sum(len(reg_list) for reg_list in pinfo.values()))
                content_scores_prev.append(calculate_page_content_score(pinfo))
                if any(len(reg_list) > 0 for reg_list in pinfo.values()):
                    content_pages_prev += 1
            else:
                # treat empty/missing page as having 0 regions and 0 content
                region_counts_prev.append(0)
                content_scores_prev.append(0.0)
        region_counts_next = []
        content_scores_next = []
        content_pages_next = 0
        for ninfo in next_pages_context:
            if ninfo and any(len(reg_list) > 0 for reg_list in ninfo.values()):
                region_counts_next.append(sum(len(reg_list) for reg_list in ninfo.values()))
                content_scores_next.append(calculate_page_content_score(ninfo))
                if any(len(reg_list) > 0 for reg_list in ninfo.values()):
                    content_pages_next += 1
            else:
                region_counts_next.append(0)
                content_scores_next.append(0.0)
        # Compute averages (use available pages count, or 0 if none available)
        prev_count_avg = (sum(region_counts_prev) / len(region_counts_prev)) if region_counts_prev else 0
        next_count_avg = (sum(region_counts_next) / len(region_counts_next)) if region_counts_next else 0
        prev_score_avg = (sum(content_scores_prev) / len(content_scores_prev)) if content_scores_prev else 0.0
        next_score_avg = (sum(content_scores_next) / len(content_scores_next)) if content_scores_next else 0.0
        # Differences between current page and context averages
        curr_count = feat['total_region_count']
        curr_score = calculate_page_content_score(page_info_list[idx] if idx < len(page_info_list) else {})
        feat['prev3_region_count_avg'] = prev_count_avg
        feat['next3_region_count_avg'] = next_count_avg
        feat['prev3_content_score_avg'] = prev_score_avg
        feat['next3_content_score_avg'] = next_score_avg
        feat['prev3_content_pages_count'] = content_pages_prev
        feat['next3_content_pages_count'] = content_pages_next
        feat['region_count_change_prev3'] = curr_count - prev_count_avg
        feat['region_count_change_next3'] = curr_count - next_count_avg
        feat['content_score_change_prev3'] = curr_score - prev_score_avg
        feat['content_score_change_next3'] = curr_score - next_score_avg
        # Determine position index (1–7) of current page in a 7-page window context
        total_pages = len(page_info_list)
        if idx < 3:
            position_idx = idx + 1  # near beginning
        elif idx > total_pages - 4:
            # near end
            offset_from_end = (total_pages - 1) - idx
            position_idx = 7 - offset_from_end
        else:
            position_idx = 4  # center of a full window
        feat['position_in_7page_window'] = position_idx

        features_list.append(feat)

    # Create DataFrame of features
    features_df = pd.DataFrame(features_list)
    # Combine with original columns (excluding the XML column)
    result_df = pd.concat([df.drop(columns=[xml_column]).reset_index(drop=True), features_df], axis=1)
    result_df.to_csv(output_csv_path, index=False)
    if verbose:
        print(f"Feature extraction complete. Output saved to {output_csv_path}")
        print(f"Total feature columns: {len(features_df.columns)}")