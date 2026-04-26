import re

def is_nearby(bbox1, bbox2, threshold=50):
    """Check if two bounding boxes are spatially close."""
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    distance = ((x1_center - x2_center)**2 + (y1_center - y2_center)**2)**0.5
    return distance < threshold

def detect_structure(lines, images, doc_id):
    """
    Module 5: Enhanced Structural Continuity & Image Context Linking

    Reconstructs logical document structure with:
    - Font-size based heading detection (most reliable for digital PDFs)
    - Pattern-based heading detection (fallback for scanned/OCR pages)
    - Element-type hints from Module 3 classification
    - Multi-column aware reading order
    - Paragraph grouping preservation from Module 4
    - Cross-page continuity detection
    - Image context linking with LLM descriptions
    """
    sections = []
    current_section = None
    section_counter = 0

    # ===== Compute font statistics for heading detection =====
    font_sizes = [line.get('font_size') for line in lines if line.get('font_size')]
    if font_sizes:
        sorted_sizes = sorted(font_sizes)
        median_font = sorted_sizes[len(sorted_sizes) // 2]
        max_font = sorted_sizes[-1]
        use_font_detection = True
    else:
        median_font = 10
        max_font = 10
        use_font_detection = False

    # ===== Column-aware reading order sort =====
    page_widths = {}
    for line in lines:
        p = line['page']
        right_edge = line['bbox'][2]
        if p not in page_widths or right_edge > page_widths[p]:
            page_widths[p] = right_edge

    def reading_order_sort_key(line):
        page = line['page']
        width = page_widths.get(page, 600)
        center_x = width / 2
        x_mid = (line['bbox'][0] + line['bbox'][2]) / 2

        if x_mid < (center_x - 50):
            col = 0
        elif x_mid > (center_x + 50):
            col = 1
        else:
            col = 0  # Spanning or centered elements treated as primary column

        return (page, col, round(line['bbox'][1] / 10) * 10)

    sorted_lines = sorted(lines, key=reading_order_sort_key)

    # ===== A. Text Structure Reconstruction =====
    for line in sorted_lines:
        text = line.get('text', '')
        l_type = line.get('type', line.get('element_type', 'text'))

        # Skip table cells from section header detection but still assign them
        if l_type == 'table_cell':
            if current_section is None:
                section_counter += 1
                section_id = f"sec_{doc_id}_{section_counter}"
                current_section = {
                    "section_id": section_id,
                    "title": f"Section {section_counter}",
                    "lines": [],
                    "images": [],
                    "page_start": line['page'],
                    "page_end": line['page']
                }
                sections.append(current_section)
            current_section["lines"].append(line['line_id'])
            current_section["page_end"] = line['page']
            line["section_id"] = current_section["section_id"]
            line["label"] = current_section["title"]
            continue

        # ===== Heading Detection: Multi-method =====
        is_header = False

        # Method 1: Font-size based detection (most reliable for digital PDFs)
        if use_font_detection and line.get('font_size'):
            fs = line['font_size']
            if fs >= max_font * 0.9 and fs > median_font * 1.3 and len(text.strip()) < 120:
                is_header = True
            elif fs > median_font * 1.15 and line.get('is_bold', False) and len(text.strip()) < 100:
                is_header = True

        # Method 2: Element type from Module 3 / Module 4 classification
        if l_type in ('heading', 'subheading'):
            is_header = True

        # Method 3: Pattern-based detection (fallback)
        if not is_header:
            patterns = [
                r'^(MODULE|SECTION|CHAPTER|PART)\s+\d+',
                r'^\d+(\.\d+)*\s+[A-Z]',
                r'^[A-Z]{2,}(\s+[A-Z]{2,})*$',
                r'^Project:\s+.*',
                r'^Objective:?\s*.*'
            ]
            if any(re.match(p, text.strip(), re.IGNORECASE) for p in patterns):
                if len(text.strip()) < 80:
                    is_header = True

        if is_header or current_section is None:
            section_counter += 1
            section_id = f"sec_{doc_id}_{section_counter}"
            title = text.strip() if is_header else f"Section {section_counter}"
            current_section = {
                "section_id": section_id,
                "title": title,
                "lines": [],
                "images": [],
                "page_start": line['page'],
                "page_end": line['page']
            }
            sections.append(current_section)

        # Section Assignment
        current_section["lines"].append(line['line_id'])
        current_section["page_end"] = line['page']
        line["section_id"] = current_section["section_id"]

        if l_type == 'table_data':
            line["label"] = f"Table in {current_section['title']}"
        else:
            line["label"] = current_section["title"]

    # ===== B. Cross-Page Continuity Detection =====
    for i in range(len(sorted_lines) - 1):
        current = sorted_lines[i]
        next_line = sorted_lines[i + 1]

        if next_line['page'] == current['page'] + 1:
            if not re.match(r'^(MODULE|SECTION|\d+\.|[A-Z\s]{4,}:)', next_line.get('text', '')):
                if current.get('section_id') == next_line.get('section_id'):
                    current['continues_to'] = next_line['line_id']
                    next_line['continues_from'] = current['line_id']

    # ===== C. Image Context Linking =====
    for img in images:
        # Extract page number from image_id
        try:
            img_page_num_str = img['image_id'].split('_p')[1].split('_')[0]
            img_page = int(img_page_num_str)
        except (IndexError, ValueError):
            continue

        # 1. Section Assignment
        relevant_section = None
        for sec in sections:
            if sec['page_start'] <= img_page <= sec['page_end']:
                relevant_section = sec

        if relevant_section:
            relevant_section["images"].append(img['image_id'])
            img["section_id"] = relevant_section["section_id"]
        else:
            img["section_id"] = "unassigned"

        # 2. Caption Detection
        caption_line = None
        for line in sorted_lines:
            if line['page'] == img_page:
                if re.match(r'^(Figure|Fig\.|Image|Table|Diagram|Chart)\s+\d+',
                            line.get('text', ''), re.IGNORECASE):
                    if is_nearby(line['bbox'], img['bbox'], threshold=100):
                        caption_line = line['line_id']
                        break

        if caption_line:
            img['caption_line'] = caption_line

        # 3. Spatial Association
        associated_lines = []
        for line in sorted_lines:
            if line['page'] == img_page:
                if is_nearby(line['bbox'], img['bbox'], threshold=80):
                    associated_lines.append(line['line_id'])

        img['associated_lines'] = associated_lines

    return sections, sorted_lines, images

