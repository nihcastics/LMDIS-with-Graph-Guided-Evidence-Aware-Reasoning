def annotate_reliability(G, threshold=0.8):
    """
    Module 8: OCR Confidence & Reliability Annotation
    Injects uncertainty awareness into the document memory.
    Handles text_components, table cells, paragraphs, sections, and tables.
    """
    low_confidence_nodes = []

    # 1. Attach ocr_confidence to text component nodes (including table cells)
    for node, data in G.nodes(data=True):
        if data.get("type") == "text_component":
            conf = data.get("confidence", 1.0)
            G.nodes[node]["ocr_confidence"] = conf

            if conf < threshold:
                G.nodes[node]["reliability_flag"] = "low"
                low_confidence_nodes.append(node)
            else:
                G.nodes[node]["reliability_flag"] = "high"

    # 2. Propagate confidence to sections, tables, and paragraphs
    for node, data in G.nodes(data=True):
        if data.get("type") in ["section", "table", "paragraph"]:
            member_confs = []
            for neighbor in G.predecessors(node):
                n_data = G.nodes[neighbor]
                if n_data.get("type") == "text_component":
                    member_confs.append(n_data.get("confidence", 1.0))
            # Also check successors for paragraph -> line edges
            for neighbor in G.successors(node):
                n_data = G.nodes[neighbor]
                if n_data.get("type") == "text_component":
                    member_confs.append(n_data.get("confidence", 1.0))

            if member_confs:
                avg_conf = sum(member_confs) / len(member_confs)
                G.nodes[node]["ocr_confidence"] = avg_conf
                G.nodes[node]["reliability_flag"] = "low" if avg_conf < threshold else "high"
            else:
                G.nodes[node]["ocr_confidence"] = 1.0
                G.nodes[node]["reliability_flag"] = "high"

    return G, low_confidence_nodes
