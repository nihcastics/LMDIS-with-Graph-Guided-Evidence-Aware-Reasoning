def normalize_components(lines, images, sections, document_elements=None):
    """
    Module 6: Enhanced Multimodal Component Normalization

    Consolidates all document elements into a canonical schema:
    - text_components: individual lines with full metadata (font, element type, paragraph)
    - image_components: images with LLM descriptions, captions, and spatial links
    - table_components: tables with cell-level structure (rows, columns, cells)
    - paragraph_components: paragraph groupings with line references and categories
    - sections: logical document sections
    """
    normalized_data = {
        "text_components": [],
        "image_components": [],
        "table_components": [],
        "paragraph_components": [],
        "sections": sections
    }

    # Create section line order mapping
    section_line_orders = {}
    for section in sections:
        section_line_orders[section["section_id"]] = section["lines"]

    # ===== 1. Text Component Normalization =====
    for line in lines:
        section_id = line.get("section_id")

        # Calculate order_in_section
        order_in_section = 0
        if section_id and section_id in section_line_orders:
            try:
                order_in_section = section_line_orders[section_id].index(line["line_id"])
            except ValueError:
                order_in_section = 0

        component = {
            "type": "text_component",
            "line_id": line["line_id"],
            "doc_id": line["doc_id"],
            "section_id": section_id,
            "order_in_section": order_in_section,
            "text": line["text"],
            "page": line["page"],
            "bbox": line["bbox"],
            "confidence": line["confidence"],
            "source": line.get("source", "unknown"),
            "font_size": line.get("font_size"),
            "is_bold": line.get("is_bold", False),
            "is_italic": line.get("is_italic", False),
            "element_type": line.get("element_type", "body"),
            "paragraph_id": line.get("paragraph_id"),
            "sequence": line.get("sequence", 0)
        }

        # Table cell metadata
        if line.get("type") == "table_cell":
            component["subtype"] = "table_cell"
            component["table_id"] = line.get("table_id")
            component["cell_row"] = line.get("cell_row")
            component["cell_col"] = line.get("cell_col")
            component["is_header"] = line.get("is_header", False)

        # Continuation links
        if "continues_to" in line:
            component["continues_to"] = line["continues_to"]
        if "continues_from" in line:
            component["continues_from"] = line["continues_from"]

        normalized_data["text_components"].append(component)

    # ===== 2. Image Component Normalization =====
    for img in images:
        try:
            img_page_str = img['image_id'].split('_p')[1].split('_')[0]
            img_page = int(img_page_str)
        except Exception:
            img_page = 0

        component = {
            "type": "image_component",
            "image_id": img["image_id"],
            "page": img_page,
            "section_id": img.get("section_id"),
            "bbox": img["bbox"],
            "format": img.get("format", "unknown"),
            "path": img.get("path", ""),
            "linked_lines": img.get("associated_lines", []),
            "llm_description": img.get("llm_description", ""),
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "role": "figure" if img.get("llm_description") else "undetermined"
        }

        if "caption_line" in img:
            component["caption_line"] = img["caption_line"]

        normalized_data["image_components"].append(component)

    # ===== 3. Table Component Normalization (Cell-Level) =====
    if document_elements:
        for elem in document_elements:
            if elem.get("element_type") == "table":
                table_comp = {
                    "type": "table_component",
                    "table_id": elem["element_id"],
                    "page": elem["page"],
                    "bbox": elem.get("bbox", (0, 0, 0, 0)),
                    "source": elem.get("source", "unknown"),
                    "confidence": elem.get("confidence", 0.9),
                    "row_count": elem.get("row_count", 0),
                    "col_count": elem.get("col_count", 0),
                    "rows": elem.get("rows", []),
                    "cells": elem.get("cells", []),
                    "header_rows": [],
                    "data_rows": []
                }
                if table_comp["rows"]:
                    table_comp["header_rows"] = [table_comp["rows"][0]]
                    table_comp["data_rows"] = table_comp["rows"][1:] if len(table_comp["rows"]) > 1 else []
                normalized_data["table_components"].append(table_comp)
    else:
        # Fallback: basic table detection from line patterns
        table_candidates = detect_tables(lines)
        for table in table_candidates:
            normalized_data["table_components"].append(table)

    # ===== 4. Paragraph Component Normalization =====
    if document_elements:
        for elem in document_elements:
            if elem.get("element_type") == "paragraph":
                para_comp = {
                    "type": "paragraph_component",
                    "paragraph_id": elem["element_id"],
                    "page": elem["page"],
                    "doc_id": elem.get("doc_id", ""),
                    "category": elem.get("category", "body"),
                    "line_ids": elem.get("line_ids", []),
                    "line_count": len(elem.get("line_ids", [])),
                    "paragraph_index": elem.get("paragraph_index", 0),
                    "bbox": elem.get("bbox", (0, 0, 0, 0))
                }
                normalized_data["paragraph_components"].append(para_comp)

    return normalized_data


def detect_tables(lines):
    """
    Fallback table detection using row/column heuristics.
    Only used when document_elements are not available.
    """
    tables = []

    # Group lines by page
    pages = {}
    for line in lines:
        page = line["page"]
        if page not in pages:
            pages[page] = []
        pages[page].append(line)

    for page_num, page_lines in pages.items():
        sorted_lines = sorted(page_lines, key=lambda x: x["bbox"][1])

        rows = []
        current_row = []
        prev_y = None

        for line in sorted_lines:
            y = line["bbox"][1]
            if prev_y is None or abs(y - prev_y) < 10:
                current_row.append(line["line_id"])
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [line["line_id"]]
            prev_y = y
        if current_row:
            rows.append(current_row)

        multi_col_rows = [r for r in rows if len(r) >= 2]

        if len(multi_col_rows) >= 2:
            table = {
                "type": "table_component",
                "table_id": f"table_p{page_num}",
                "page": page_num,
                "rows": multi_col_rows,
                "header_rows": [multi_col_rows[0]] if multi_col_rows else [],
                "data_rows": multi_col_rows[1:] if len(multi_col_rows) > 1 else []
            }
            tables.append(table)

    return tables

