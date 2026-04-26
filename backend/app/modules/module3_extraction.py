import pymupdf as fitz
import os
import re
import base64

try:
    from paddleocr import PaddleOCR
    paddle_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    from paddleocr import PPStructure
    structure_engine = PPStructure(show_log=False, image_orientation=True)
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from openai import OpenAI
    from backend.app.config import OPENROUTER_API_KEY, OPENROUTER_FALLBACK_MODELS
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# Get absolute path to the backend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_IMAGE_DIR = os.path.join(BASE_DIR, "app", "storage", "extracted_images")


def _describe_image_with_llm(image_path):
    """Send an image to LLM for contextual understanding and description."""
    if not LLM_AVAILABLE:
        return ""
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/png")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        vision_models = ["google/gemini-2.0-flash:free"] + OPENROUTER_FALLBACK_MODELS
        for model_name in vision_models:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": (
                            "Describe this image in detail. What does it show? "
                            "If it contains a chart, table, diagram, or figure, describe the data, "
                            "labels, and relationships shown. Be specific and factual. "
                            "Keep the description under 200 words."
                        )},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{img_b64}"
                        }}
                    ]}],
                    max_tokens=400
                )
                desc = response.choices[0].message.content.strip()
                desc = re.sub(r'<think>.*?</think>', '', desc, flags=re.DOTALL).strip()
                if desc:
                    return desc
            except Exception:
                continue
        return ""
    except Exception:
        return ""


def _extract_tables_pdfplumber(doc_path, page_num):
    """Extract tables with cell-level precision using pdfplumber."""
    if not PDFPLUMBER_AVAILABLE:
        return []
    tables_data = []
    try:
        with pdfplumber.open(doc_path) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            tables = page.find_tables()
            for t_idx, table in enumerate(tables):
                bbox = table.bbox
                extracted = table.extract()
                if not extracted:
                    continue
                table_data = {
                    "table_id": f"table_p{page_num}_{t_idx}",
                    "bbox": tuple(bbox) if bbox else (0, 0, 0, 0),
                    "rows": [],
                    "row_count": len(extracted),
                    "col_count": max(len(row) for row in extracted) if extracted else 0,
                    "source": "pdfplumber",
                    "confidence": 0.95
                }
                for r_idx, row in enumerate(extracted):
                    row_data = {"row_index": r_idx, "is_header": r_idx == 0, "cells": []}
                    for c_idx, cell in enumerate(row):
                        cell_text = (cell or "").strip()
                        row_data["cells"].append({
                            "row": r_idx, "col": c_idx,
                            "text": cell_text, "is_header": r_idx == 0
                        })
                    table_data["rows"].append(row_data)
                tables_data.append(table_data)
    except Exception as e:
        print(f"pdfplumber table extraction error on page {page_num}: {e}")
    return tables_data


def _classify_text_element(text, font_size, is_bold, avg_font_size, max_font_size, bbox, page_height):
    """Classify a text element based on visual and textual properties."""
    text_stripped = text.strip()
    if not text_stripped:
        return "empty"
    y_center = (bbox[1] + bbox[3]) / 2
    if page_height > 0:
        if y_center > page_height * 0.95:
            return "footer"
        if y_center < page_height * 0.04 and len(text_stripped) < 50:
            return "page_header"
    if font_size and avg_font_size and max_font_size:
        if font_size >= max_font_size * 0.9 and font_size > avg_font_size * 1.3 and len(text_stripped) < 120:
            return "heading"
        if font_size > avg_font_size * 1.15 and is_bold and len(text_stripped) < 100:
            return "subheading"
    # Bullet/list detection
    bullet_chars = '\u2022\u2023\u25e6\u2043\u2219'
    list_patterns = [
        r'^\s*[' + bullet_chars + r'\-\*]\s+',
        r'^\s*\d+[\.\)]\s+',
        r'^\s*[a-zA-Z][\.\)]\s+',
        r'^\s*[ivxlcdm]+[\.\)]\s+',
    ]
    for pattern in list_patterns:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return "list_item"
    if re.match(r'^(Figure|Fig\.|Table|Image|Diagram|Chart|Graph|Exhibit)\s+\d+',
                text_stripped, re.IGNORECASE):
        return "caption"
    if font_size and avg_font_size and font_size < avg_font_size * 0.8:
        return "footnote"
    return "body"


def _parse_html_table(html):
    """Parse an HTML table string into a list of rows, each containing cell texts."""
    rows = []
    row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
    for row_html in row_matches:
        cells = []
        cell_matches = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
        for cell_html in cell_matches:
            cell_text = re.sub(r'<[^>]+>', '', cell_html).strip()
            cells.append(cell_text)
        if cells:
            rows.append(cells)
    return rows


def _bbox_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes."""
    x0, y0 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    x1, y1 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area1 = max((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]), 1)
    area2 = max((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]), 1)
    return intersection / min(area1, area2)

def extract_content(doc_path, page_types, output_image_dir=None):
    """
    Module 3: High-Precision Lossless Content Extraction (Text + Tables + Images)

    Extracts document content with:
    - Line-level text extraction with font metadata (size, bold, italic)
    - Element type classification (heading, body, list_item, caption, footer, etc.)
    - Cell-level table extraction via pdfplumber (digital) or PaddleOCR structure (scanned)
    - Image extraction with bounding boxes and LLM-based contextual descriptions
    - Sequential reading order preservation
    - OCR deduplication for scanned pages
    """
    if output_image_dir is None:
        output_image_dir = DEFAULT_IMAGE_DIR
    os.makedirs(output_image_dir, exist_ok=True)
    doc = fitz.open(doc_path)
    extraction_results = {}

    for page_num_0, page in enumerate(doc):
        page_num = page_num_0 + 1
        p_type = page_types.get(page_num, "scanned")
        page_height = page.rect.height
        page_width = page.rect.width

        page_data = {
            "text_blocks": [],
            "images": [],
            "tables": [],
            "page_num": page_num,
            "page_dimensions": {"width": page_width, "height": page_height}
        }

        # ============================================================
        # A. TEXT EXTRACTION - Sequential line-by-line with metadata
        # ============================================================
        if p_type == "digital":
            # --- Digital Pages: Full metadata line-level extraction ---
            blocks = page.get_text("dict", sort=True)["blocks"]

            # First pass: collect font size statistics for classification
            all_font_sizes = []
            for b in blocks:
                if b['type'] == 0:
                    for line in b['lines']:
                        for span in line['spans']:
                            if span['text'].strip():
                                all_font_sizes.append(span.get('size', 10))
            avg_font = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 10
            max_font = max(all_font_sizes) if all_font_sizes else 10

            # Second pass: extract each LINE (not span) as atomic unit
            line_seq = 0
            for b in blocks:
                if b['type'] == 0:
                    for line in b['lines']:
                        line_parts = []
                        line_sizes = []
                        line_fonts = []
                        is_bold = False
                        is_italic = False
                        for span in line['spans']:
                            txt = span['text']
                            if txt.strip():
                                line_parts.append(txt)
                                line_sizes.append(span.get('size', 10))
                                fname = span.get('font', '')
                                line_fonts.append(fname)
                                if 'bold' in fname.lower() or 'black' in fname.lower():
                                    is_bold = True
                                if 'italic' in fname.lower() or 'oblique' in fname.lower():
                                    is_italic = True
                        full_text = "".join(line_parts).strip()
                        if not full_text:
                            continue
                        dom_size = max(set(line_sizes), key=line_sizes.count) if line_sizes else 10
                        elem_type = _classify_text_element(
                            full_text, dom_size, is_bold,
                            avg_font, max_font, line['bbox'], page_height
                        )
                        line_seq += 1
                        page_data["text_blocks"].append({
                            "text": full_text,
                            "bbox": tuple(line['bbox']),
                            "font_size": dom_size,
                            "font_name": line_fonts[0] if line_fonts else "",
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "element_type": elem_type,
                            "sequence": line_seq,
                            "source": "digital",
                            "confidence": 1.0
                        })

            # Cell-level table extraction via pdfplumber
            tables = _extract_tables_pdfplumber(doc_path, page_num)
            if tables:
                page_data["tables"] = tables
                # Mark text blocks that fall inside table bounding boxes
                for table in tables:
                    t_bbox = table["bbox"]
                    for block in page_data["text_blocks"]:
                        bx0, by0, bx1, by1 = block["bbox"]
                        if (bx0 >= t_bbox[0] - 5 and by0 >= t_bbox[1] - 5 and
                                bx1 <= t_bbox[2] + 5 and by1 <= t_bbox[3] + 5):
                            block["in_table"] = True
                            block["table_id"] = table["table_id"]

        elif p_type in ["mixed", "scanned"]:
            # --- Scanned/Mixed Pages: OCR with high-precision structure detection ---
            pix = page.get_pixmap(dpi=300)  # High DPI for OCR precision
            img_path = os.path.join(output_image_dir, f"{os.path.basename(doc_path)}_p{page_num}.png")
            pix.save(img_path)

            if PADDLE_AVAILABLE:
                # 1. Structure Analysis (Tables/Figures/Text regions)
                try:
                    import cv2
                    img = cv2.imread(img_path)
                    structure_res = structure_engine(img)

                    for region in structure_res:
                        r_type = region.get('type', '')
                        bbox = region.get('bbox', [0, 0, 0, 0])

                        if r_type == 'table':
                            # Parse table with cell-level precision from HTML
                            if 'res' in region and isinstance(region['res'], dict) and 'html' in region['res']:
                                html = region['res']['html']
                                parsed_rows = _parse_html_table(html)
                                table_data = {
                                    "table_id": f"table_p{page_num}_{int(bbox[0])}",
                                    "bbox": tuple(bbox),
                                    "rows": [],
                                    "row_count": len(parsed_rows),
                                    "col_count": max(len(r) for r in parsed_rows) if parsed_rows else 0,
                                    "source": "paddle_structure",
                                    "confidence": 0.95
                                }
                                for r_idx, row_cells in enumerate(parsed_rows):
                                    row_data = {"row_index": r_idx, "is_header": r_idx == 0, "cells": []}
                                    for c_idx, cell_text in enumerate(row_cells):
                                        row_data["cells"].append({
                                            "row": r_idx, "col": c_idx,
                                            "text": cell_text.strip(), "is_header": r_idx == 0
                                        })
                                    table_data["rows"].append(row_data)
                                if table_data["rows"]:
                                    page_data["tables"].append(table_data)
                            else:
                                # Fallback: store as raw table text block
                                table_text = ""
                                if 'res' in region:
                                    if isinstance(region['res'], dict):
                                        table_text = region['res'].get('text', '')
                                    elif isinstance(region['res'], str):
                                        table_text = region['res']
                                page_data["text_blocks"].append({
                                    "text": f"[TABLE] {table_text}",
                                    "bbox": tuple(bbox),
                                    "source": "paddle_structure_table",
                                    "type": "table",
                                    "confidence": 0.95
                                })
                        elif r_type in ['figure', 'chart']:
                            page_data["images"].append({
                                "image_id": f"fig_{os.path.basename(doc_path)}_p{page_num}_{int(bbox[0])}",
                                "bbox": tuple(bbox),
                                "type": r_type,
                                "source": "paddle_structure"
                            })
                except Exception as e:
                    print(f"Structure analysis error on page {page_num}: {e}")

                # 2. General OCR with deduplication
                try:
                    results = paddle_engine.ocr(img_path, cls=True)
                    if results and results[0]:
                        existing_bboxes = [(b["bbox"], b["text"]) for b in page_data["text_blocks"]]
                        line_seq = len(page_data["text_blocks"])
                        for res in results[0]:
                            bbox_points = res[0]
                            text, conf = res[1]
                            x0 = min(p[0] for p in bbox_points)
                            y0 = min(p[1] for p in bbox_points)
                            x1 = max(p[0] for p in bbox_points)
                            y1 = max(p[1] for p in bbox_points)
                            new_bbox = (x0, y0, x1, y1)

                            # Deduplication: skip if overlaps existing block
                            is_duplicate = False
                            for ex_bbox, ex_text in existing_bboxes:
                                overlap = _bbox_overlap(new_bbox, ex_bbox)
                                if overlap > 0.5 or (text.strip() == ex_text.strip() and overlap > 0.2):
                                    is_duplicate = True
                                    break
                            if not is_duplicate and text.strip():
                                line_seq += 1
                                page_data["text_blocks"].append({
                                    "text": text.strip(),
                                    "bbox": new_bbox,
                                    "source": "paddle_ocr",
                                    "confidence": float(conf),
                                    "sequence": line_seq,
                                    "element_type": "body"
                                })
                except Exception as e:
                    print(f"PaddleOCR error on page {page_num}: {e}")
                    p_type = "fallback_tesseract"

            if not PADDLE_AVAILABLE or p_type == "fallback_tesseract":
                if OCR_AVAILABLE:
                    try:
                        img = Image.open(img_path)
                        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

                        # Group word-level OCR results into line-level for precision
                        lines_dict = {}
                        n_boxes = len(ocr_data['text'])
                        for i in range(n_boxes):
                            text = ocr_data['text'][i].strip()
                            if text:
                                key = (ocr_data['block_num'][i], ocr_data['line_num'][i])
                                if key not in lines_dict:
                                    lines_dict[key] = {
                                        "texts": [], "x0": float('inf'), "y0": float('inf'),
                                        "x1": 0, "y1": 0, "confidences": []
                                    }
                                lines_dict[key]["texts"].append(text)
                                x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                                              ocr_data['width'][i], ocr_data['height'][i])
                                lines_dict[key]["x0"] = min(lines_dict[key]["x0"], x)
                                lines_dict[key]["y0"] = min(lines_dict[key]["y0"], y)
                                lines_dict[key]["x1"] = max(lines_dict[key]["x1"], x + w)
                                lines_dict[key]["y1"] = max(lines_dict[key]["y1"], y + h)
                                conf_val = float(ocr_data['conf'][i])
                                if conf_val > 0:
                                    lines_dict[key]["confidences"].append(conf_val / 100.0)

                        line_seq = 0
                        for key in sorted(lines_dict.keys()):
                            ld = lines_dict[key]
                            full_text = " ".join(ld["texts"])
                            avg_conf = sum(ld["confidences"]) / len(ld["confidences"]) if ld["confidences"] else 0.5
                            line_seq += 1
                            page_data["text_blocks"].append({
                                "text": full_text,
                                "bbox": (ld["x0"], ld["y0"], ld["x1"], ld["y1"]),
                                "source": "tesseract_ocr",
                                "confidence": avg_conf,
                                "sequence": line_seq,
                                "element_type": "body"
                            })
                    except Exception as e:
                        print(f"Tesseract OCR error on page {page_num}: {e}")

        # --- Final Fallback: native extraction if nothing found ---
        if not page_data["text_blocks"]:
            fallback_dict = page.get_text("dict", sort=True)
            if "blocks" in fallback_dict:
                line_seq = 0
                for b in fallback_dict["blocks"]:
                    if b['type'] == 0:
                        for line in b['lines']:
                            full_text = "".join(s['text'] for s in line['spans']).strip()
                            if full_text:
                                line_seq += 1
                                page_data["text_blocks"].append({
                                    "text": full_text,
                                    "bbox": tuple(line['bbox']),
                                    "font_size": line['spans'][0].get('size', 10) if line['spans'] else 10,
                                    "source": "digital_fallback",
                                    "confidence": 0.5,
                                    "sequence": line_seq,
                                    "element_type": "body"
                                })

        # ============================================================
        # B. IMAGE EXTRACTION with bounding boxes + LLM description
        # ============================================================
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_id = f"img_{os.path.basename(doc_path)}_p{page_num}_{img_index}"

                img_filename = f"{image_id}.{image_ext}"
                img_save_path = os.path.join(output_image_dir, img_filename)
                with open(img_save_path, "wb") as f_img:
                    f_img.write(image_bytes)

                rects = page.get_image_rects(xref)
                bbox = tuple(rects[0]) if rects else (0, 0, 0, 0)

                # Filter tiny images (icons, decorations, line separators)
                if bbox != (0, 0, 0, 0):
                    img_w = bbox[2] - bbox[0]
                    img_h = bbox[3] - bbox[1]
                    if img_w < 30 or img_h < 30:
                        continue

                # LLM-based image understanding
                llm_description = _describe_image_with_llm(img_save_path)

                page_data["images"].append({
                    "image_id": image_id,
                    "bbox": bbox,
                    "format": image_ext,
                    "path": img_save_path,
                    "llm_description": llm_description,
                    "width": bbox[2] - bbox[0] if bbox != (0, 0, 0, 0) else 0,
                    "height": bbox[3] - bbox[1] if bbox != (0, 0, 0, 0) else 0
                })
            except Exception as e:
                print(f"Error extracting image {img_index} on page {page_num}: {e}")

        # ============================================================
        # C. READING ORDER SORT & SEQUENCE ASSIGNMENT
        # ============================================================
        page_data["text_blocks"].sort(
            key=lambda b: (round(b["bbox"][1] / 5) * 5, b["bbox"][0])
        )
        for idx, block in enumerate(page_data["text_blocks"]):
            block["sequence"] = idx + 1

        extraction_results[page_num] = page_data

    doc.close()
    return extraction_results
