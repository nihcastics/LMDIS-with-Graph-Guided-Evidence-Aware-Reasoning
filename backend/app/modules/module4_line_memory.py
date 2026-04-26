import uuid
import json
import re


def normalize_lines(extraction_results, doc_id):
    """
    Module 4: Enhanced Atomic Line Normalization with Paragraph Detection

    Converts raw text blocks into:
    - Sequential, addressable line units with full metadata (font, bold, italic)
    - Paragraph groupings with start/end boundaries and categories
    - Cell-level table data with row/column indexing
    - Element type classifications (heading, body, list_item, caption, etc.)
    - Proper reading order preservation

    Returns: (normalized_lines, document_elements)
    - normalized_lines: list of line dicts (backward compatible with downstream modules)
    - document_elements: list of paragraph/table element dicts (rich structure)
    """
    normalized_lines = []
    document_elements = []
    global_sequence = 0

    for page_num in sorted(extraction_results.keys()):
        data = extraction_results[page_num]
        text_blocks = data.get("text_blocks", [])
        tables = data.get("tables", [])
        page_dims = data.get("page_dimensions", {"width": 612, "height": 792})

        # ===== 1. Process Tables (cell-level) =====
        table_bboxes = []
        for table in tables:
            table_id = table.get("table_id", f"table_p{page_num}_{uuid.uuid4().hex[:6]}")
            table_bbox = table.get("bbox", (0, 0, 0, 0))
            table_bboxes.append(table_bbox)

            table_element = {
                "element_type": "table",
                "element_id": table_id,
                "doc_id": doc_id,
                "page": page_num,
                "bbox": table_bbox,
                "source": table.get("source", "unknown"),
                "confidence": table.get("confidence", 0.9),
                "row_count": table.get("row_count", 0),
                "col_count": table.get("col_count", 0),
                "rows": [],
                "cells": []
            }

            for row in table.get("rows", []):
                row_cells = []
                for cell in row.get("cells", []):
                    cell_id = f"{doc_id}_p{page_num}_{table_id}_r{cell['row']}_c{cell['col']}"
                    cell_data = {
                        "cell_id": cell_id,
                        "row": cell["row"],
                        "col": cell["col"],
                        "text": cell["text"],
                        "is_header": cell.get("is_header", False)
                    }
                    row_cells.append(cell_data)
                    table_element["cells"].append(cell_data)

                    # Create a normalized line for each non-empty cell
                    if cell["text"].strip():
                        global_sequence += 1
                        normalized_lines.append({
                            "line_id": cell_id,
                            "doc_id": doc_id,
                            "page": page_num,
                            "text": cell["text"],
                            "bbox": table_bbox,
                            "source": table.get("source", "unknown"),
                            "confidence": table.get("confidence", 0.9),
                            "type": "table_cell",
                            "table_id": table_id,
                            "cell_row": cell["row"],
                            "cell_col": cell["col"],
                            "is_header": cell.get("is_header", False),
                            "sequence": global_sequence
                        })

                table_element["rows"].append(row_cells)

            document_elements.append(table_element)

        # ===== 2. Filter text blocks inside tables =====
        non_table_blocks = []
        for block in text_blocks:
            if block.get("in_table"):
                continue
            is_in_table = False
            for t_bbox in table_bboxes:
                if _is_inside_bbox(block["bbox"], t_bbox, tolerance=10):
                    is_in_table = True
                    break
            if not is_in_table:
                non_table_blocks.append(block)

        # ===== 3. Process legacy table blocks (from OCR) =====
        regular_blocks = [b for b in non_table_blocks if b.get("type") != "table"]
        legacy_table_blocks = [b for b in non_table_blocks if b.get("type") == "table"]

        for t_block in legacy_table_blocks:
            global_sequence += 1
            normalized_lines.append({
                "line_id": f"{doc_id}_p{page_num}_table_{uuid.uuid4().hex[:8]}",
                "doc_id": doc_id,
                "page": page_num,
                "text": t_block["text"],
                "bbox": t_block["bbox"],
                "source": t_block["source"],
                "confidence": t_block.get("confidence", 1.0),
                "type": "table_data",
                "sequence": global_sequence
            })

        if not regular_blocks:
            continue

        # ===== 4. Visual line segmentation =====
        visual_lines = _group_into_visual_lines(regular_blocks, page_dims)

        # ===== 5. Paragraph detection & line normalization =====
        paragraphs = _detect_paragraphs(visual_lines, page_dims)

        for para_idx, paragraph in enumerate(paragraphs):
            para_id = f"{doc_id}_p{page_num}_para_{para_idx}"
            para_element = {
                "element_type": "paragraph",
                "element_id": para_id,
                "doc_id": doc_id,
                "page": page_num,
                "line_ids": [],
                "paragraph_index": para_idx,
                "category": paragraph["category"]
            }

            for line_data in paragraph["lines"]:
                global_sequence += 1
                line_id = f"{doc_id}_p{page_num}_{uuid.uuid4().hex[:8]}"
                element_type = line_data.get("element_type", "body")

                normalized_line = {
                    "line_id": line_id,
                    "doc_id": doc_id,
                    "page": page_num,
                    "text": line_data["text"],
                    "bbox": line_data["bbox"],
                    "source": line_data.get("source", "unknown"),
                    "confidence": line_data.get("confidence", 1.0),
                    "font_size": line_data.get("font_size"),
                    "is_bold": line_data.get("is_bold", False),
                    "is_italic": line_data.get("is_italic", False),
                    "element_type": element_type,
                    "type": element_type,
                    "paragraph_id": para_id,
                    "paragraph_index": para_idx,
                    "line_in_paragraph": len(para_element["line_ids"]),
                    "sequence": global_sequence
                }

                normalized_lines.append(normalized_line)
                para_element["line_ids"].append(line_id)

            # Compute paragraph bounding box
            if para_element["line_ids"]:
                line_id_set = set(para_element["line_ids"])
                para_lines = [l for l in normalized_lines if l["line_id"] in line_id_set]
                if para_lines:
                    para_element["bbox"] = (
                        min(l["bbox"][0] for l in para_lines),
                        min(l["bbox"][1] for l in para_lines),
                        max(l["bbox"][2] for l in para_lines),
                        max(l["bbox"][3] for l in para_lines)
                    )

            document_elements.append(para_element)

    return normalized_lines, document_elements


def _is_inside_bbox(inner, outer, tolerance=5):
    """Check if inner bbox is inside outer bbox with tolerance."""
    return (inner[0] >= outer[0] - tolerance and
            inner[1] >= outer[1] - tolerance and
            inner[2] <= outer[2] + tolerance and
            inner[3] <= outer[3] + tolerance)


def _group_into_visual_lines(blocks, page_dims):
    """Group text blocks into visual lines based on Y-coordinate alignment."""
    if not blocks:
        return []

    sorted_blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    page_width = page_dims.get("width", 612)
    column_gap_threshold = max(page_width * 0.06, 40)
    y_tolerance = 5

    visual_lines = []
    current_line_blocks = [sorted_blocks[0]]
    current_y_center = (sorted_blocks[0]["bbox"][1] + sorted_blocks[0]["bbox"][3]) / 2

    for block in sorted_blocks[1:]:
        y_center = (block["bbox"][1] + block["bbox"][3]) / 2

        if abs(y_center - current_y_center) <= y_tolerance:
            last_block = current_line_blocks[-1]
            gap = block["bbox"][0] - last_block["bbox"][2]
            if gap > column_gap_threshold:
                # Column boundary detected
                merged = _merge_line_blocks(current_line_blocks)
                if merged:
                    visual_lines.append(merged)
                current_line_blocks = [block]
                current_y_center = y_center
            else:
                current_line_blocks.append(block)
                current_y_center = sum(
                    (b["bbox"][1] + b["bbox"][3]) / 2 for b in current_line_blocks
                ) / len(current_line_blocks)
        else:
            # Vertical jump -> new line
            merged = _merge_line_blocks(current_line_blocks)
            if merged:
                visual_lines.append(merged)
            current_line_blocks = [block]
            current_y_center = y_center

    if current_line_blocks:
        merged = _merge_line_blocks(current_line_blocks)
        if merged:
            visual_lines.append(merged)

    return visual_lines


def _merge_line_blocks(blocks):
    """Merge multiple horizontally aligned text blocks into a single atomic line."""
    if not blocks:
        return None

    blocks.sort(key=lambda b: b["bbox"][0])
    text = " ".join(b["text"].strip() for b in blocks if b["text"].strip())
    if not text:
        return None

    bbox = (
        min(b["bbox"][0] for b in blocks),
        min(b["bbox"][1] for b in blocks),
        max(b["bbox"][2] for b in blocks),
        max(b["bbox"][3] for b in blocks)
    )
    avg_conf = sum(b.get("confidence", 1.0) for b in blocks) / len(blocks)

    font_sizes = [b.get("font_size") for b in blocks if b.get("font_size")]
    font_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else None
    is_bold = any(b.get("is_bold", False) for b in blocks)
    is_italic = any(b.get("is_italic", False) for b in blocks)
    element_types = [b.get("element_type", "body") for b in blocks]
    element_type = max(set(element_types), key=element_types.count) if element_types else "body"

    return {
        "text": text,
        "bbox": bbox,
        "confidence": avg_conf,
        "font_size": font_size,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "element_type": element_type,
        "source": blocks[0].get("source", "unknown")
    }


def _detect_paragraphs(visual_lines, page_dims):
    """
    Detect paragraph boundaries based on:
    - Vertical spacing between lines (large gap = new paragraph)
    - Element type changes (heading -> body)
    - Indentation changes
    - List item boundaries
    """
    if not visual_lines:
        return []

    paragraphs = []
    current_para_lines = [visual_lines[0]]

    for i in range(1, len(visual_lines)):
        prev_line = visual_lines[i - 1]
        curr_line = visual_lines[i]

        prev_bottom = prev_line["bbox"][3]
        curr_top = curr_line["bbox"][1]
        vertical_gap = curr_top - prev_bottom

        prev_height = prev_line["bbox"][3] - prev_line["bbox"][1]
        curr_height = curr_line["bbox"][3] - curr_line["bbox"][1]
        avg_line_height = (prev_height + curr_height) / 2 if (prev_height + curr_height) > 0 else 12

        new_paragraph = False
        curr_elem = curr_line.get("element_type", "body")
        prev_elem = prev_line.get("element_type", "body")

        # Rule 1: Element type boundary (headings, captions always start new paragraph)
        if curr_elem in ("heading", "subheading", "caption"):
            new_paragraph = True
        elif prev_elem in ("heading", "subheading", "caption") and curr_elem != prev_elem:
            new_paragraph = True

        # Rule 2: Large vertical gap (> 1.8x line height)
        if avg_line_height > 0 and vertical_gap > avg_line_height * 1.8:
            new_paragraph = True

        # Rule 3: Indentation change (first-line indent)
        prev_x0 = prev_line["bbox"][0]
        curr_x0 = curr_line["bbox"][0]
        if abs(curr_x0 - prev_x0) > 15 and curr_elem == "body":
            new_paragraph = True

        # Rule 4: List item boundary
        if curr_elem == "list_item" and prev_elem != "list_item":
            new_paragraph = True

        if new_paragraph:
            paragraphs.append({
                "lines": current_para_lines,
                "category": _determine_paragraph_category(current_para_lines)
            })
            current_para_lines = [curr_line]
        else:
            current_para_lines.append(curr_line)

    if current_para_lines:
        paragraphs.append({
            "lines": current_para_lines,
            "category": _determine_paragraph_category(current_para_lines)
        })

    return paragraphs


def _determine_paragraph_category(lines):
    """Determine the category of a paragraph from its constituent line types."""
    if not lines:
        return "body"
    types = [l.get("element_type", "body") for l in lines]
    if "heading" in types:
        return "heading"
    if "subheading" in types:
        return "subheading"
    if "caption" in types:
        return "caption"
    if all(t == "list_item" for t in types):
        return "list"
    if any(t == "list_item" for t in types):
        return "list"
    if "footer" in types or "page_header" in types:
        return "metadata"
    if "footnote" in types:
        return "footnote"
    return "body"


def save_lines(lines, output_path="app/storage/lines.json"):
    with open(output_path, "w") as f:
        json.dump(lines, f, indent=2)
