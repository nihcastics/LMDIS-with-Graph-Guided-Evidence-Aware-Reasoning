import pymupdf as fitz

def analyze_page_types(doc_path):
    """
    Module 2: Page-Level Content Type Analysis
    Determines if pages are 'digital', 'scanned', or 'mixed'.
    """
    doc = fitz.open(doc_path)
    page_types = {}

    for page_num_0, page in enumerate(doc):
        page_num = page_num_0 + 1
        
        # 1. Text Object Detection
        text = page.get_text()
        has_text = len(text.strip()) > 5  # Threshold for usable text (reduced from 50 to 5)
        
        # 2. Raster Dominance Analysis
        images = page.get_images()
        has_images = len(images) > 0
        
        # Calculate image coverage (simplified)
        image_area = 0.0
        for img in images:
            try:
                # img[0] is the xref
                rects = page.get_image_rects(img[0])
                for r in rects:
                    image_area += abs(r)
            except:
                pass
        
        page_area = abs(page.rect)
        coverage_ratio = image_area / page_area if page_area > 0 else 0

        # 4. Page Classification Logic
        if has_text and not has_images:
            p_type = "digital"
        elif not has_text and has_images:
            p_type = "scanned"
        elif has_text and has_images:
            # Check overlap or coverage
            if coverage_ratio > 0.8:
                p_type = "scanned" 
            else:
                p_type = "mixed"
        else:
            p_type = "scanned" # Fallback
            
        page_types[page_num] = p_type

    return page_types
