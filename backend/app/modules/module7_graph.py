"""
Module 7: Deep Semantic Document Memory Graph
==================================================
Builds a richly interconnected NetworkX DiGraph where every piece of
information is linked to its semantic neighbours, its structural
parents, its claims, and cross-references within the document.

Node types:
  document, section, paragraph, text_component, image_component, table

Edge types (structural):
  BELONGS_TO, NEXT_LINE, NEXT_SECTION, PARAGRAPH_CONTAINS,
  CAPTIONED_BY, DESCRIBED_BY, CONTINUES_ON_PAGE, TABLE_CELL_OF

Edge types (semantic — added here):
  SECTION_SUMMARY_OF        section  → document
  PARAGRAPH_SUMMARY_OF      paragraph → section
  TOPIC_BRIDGE              section  ↔ section    (shared key-concept)
  DEFINITION_USED_IN        text_component → text_component
  TABLE_CONTEXT             table → section
  IMAGE_CONTEXT             image → section
"""

import re
import networkx as nx


def build_graph(normalized_data, doc_meta=None):
    G = nx.DiGraph()

    doc_id = doc_meta["doc_id"] if doc_meta else "unknown_doc"
    G.add_node(doc_id, type="document",
               label=doc_meta.get("filename", "Document") if doc_meta else "Document",
               detected_title=doc_meta.get("detected_title", "") if doc_meta else "")

    # ═══ 1. Section Nodes ═══
    previous_section_node = None
    section_ids = []
    for sec in normalized_data["sections"]:
        sid = sec["section_id"]
        section_ids.append(sid)
        G.add_node(sid, type="section",
                   label=sec["title"], page=sec["page_start"])
        G.add_edge(doc_id, sid, relationship="BELONGS_TO")
        G.add_edge(sid, doc_id, relationship="SECTION_SUMMARY_OF")

        if previous_section_node:
            G.add_edge(previous_section_node, sid, relationship="NEXT_SECTION")
        previous_section_node = sid

    # ═══ 2. Paragraph Nodes ═══
    for para in normalized_data.get("paragraph_components", []):
        pid = para["paragraph_id"]
        G.add_node(pid, type="paragraph",
                   category=para.get("category", "body"),
                   page=para.get("page", 0),
                   line_count=para.get("line_count", 0),
                   bbox=para.get("bbox", (0, 0, 0, 0)))

    # ═══ 3. Text Component Nodes ═══
    previous_line_node = None
    previous_line_section = None
    all_lines = normalized_data["text_components"]

    for line in all_lines:
        lid = line["line_id"]
        csec = line.get("section_id")
        G.add_node(lid, **line)

        if csec and csec in G:
            G.add_edge(lid, csec, relationship="BELONGS_TO")

        if previous_line_node and previous_line_section == csec:
            G.add_edge(previous_line_node, lid, relationship="NEXT_LINE")

        if "continues_to" in line:
            G.add_edge(lid, line["continues_to"], relationship="CONTINUES_ON_PAGE")

        previous_line_node = lid
        previous_line_section = csec

    # ═══ 4. Paragraph → Line edges + Paragraph → Section ═══
    for para in normalized_data.get("paragraph_components", []):
        pid = para["paragraph_id"]
        if pid not in G:
            continue
        for lid in para.get("line_ids", []):
            if lid in G:
                G.add_edge(pid, lid, relationship="PARAGRAPH_CONTAINS")
                sec_id = G.nodes[lid].get("section_id")
                if sec_id and sec_id in G and not G.has_edge(pid, sec_id):
                    G.add_edge(pid, sec_id, relationship="BELONGS_TO")
                    G.add_edge(pid, sec_id, relationship="PARAGRAPH_SUMMARY_OF")

    # ═══ 5. Image Nodes ═══
    for img in normalized_data["image_components"]:
        G.add_node(img["image_id"], **img)
        sec_id = img.get("section_id")
        if sec_id and sec_id != "unassigned" and sec_id in G:
            G.add_edge(img["image_id"], sec_id, relationship="BELONGS_TO")
            G.add_edge(img["image_id"], sec_id, relationship="IMAGE_CONTEXT")
        if "caption_line" in img and img["caption_line"] in G:
            G.add_edge(img["image_id"], img["caption_line"], relationship="CAPTIONED_BY")
        for ll in img.get("linked_lines", []):
            if ll in G:
                G.add_edge(img["image_id"], ll, relationship="DESCRIBED_BY")

    # ═══ 6. Table Nodes ═══
    for table in normalized_data.get("table_components", []):
        tid = table["table_id"]
        G.add_node(tid, type="table", table_id=tid,
                   page=table.get("page", 0),
                   row_count=table.get("row_count", 0),
                   col_count=table.get("col_count", 0),
                   bbox=table.get("bbox", (0, 0, 0, 0)),
                   confidence=table.get("confidence", 0.9))
        G.add_edge(doc_id, tid, relationship="BELONGS_TO")
        # Link table to nearest section via cells
        linked_section = None
        for cell in table.get("cells", []):
            cid = cell.get("cell_id")
            if cid and cid in G:
                G.add_edge(cid, tid, relationship="TABLE_CELL_OF")
                if not linked_section:
                    ls = G.nodes[cid].get("section_id")
                    if ls and ls in G:
                        linked_section = ls
        if linked_section:
            G.add_edge(tid, linked_section, relationship="TABLE_CONTEXT")
        for row in table.get("rows", []):
            if isinstance(row, list):
                for lid in row:
                    if lid in G:
                        G.add_edge(lid, tid, relationship="BELONGS_TO")

    # ═══ 7. Semantic TOPIC_BRIDGE edges between sections ═══
    _add_topic_bridges(G, section_ids)

    # ═══ 8. DEFINITION_USED_IN edges ═══
    _add_definition_links(G, all_lines)

    return G


# ─── Helpers ────────────────────────────────────────────────────────────────

_DEFINITION_PATTERNS = re.compile(
    r'(?:defined?\s+as|refers?\s+to|known\s+as|is\s+(?:a|an|the)\b)',
    re.IGNORECASE,
)


def _add_topic_bridges(G, section_ids):
    """Connect sections that share key concepts via TOPIC_BRIDGE edges."""
    sec_words = {}
    for sid in section_ids:
        if sid not in G:
            continue
        words = set()
        for pred in G.predecessors(sid):
            nd = G.nodes.get(pred, {})
            if nd.get("type") == "text_component":
                text = nd.get("text", "").lower()
                # Extract significant words (>= 5 chars, skipping stopwords)
                words.update(w for w in text.split() if len(w) >= 5)
        sec_words[sid] = words

    ids = list(sec_words.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            common = sec_words[ids[i]] & sec_words[ids[j]]
            if len(common) >= 3:
                overlap_ratio = len(common) / min(len(sec_words[ids[i]]) or 1,
                                                   len(sec_words[ids[j]]) or 1)
                if overlap_ratio > 0.05:
                    G.add_edge(ids[i], ids[j], relationship="TOPIC_BRIDGE",
                               shared_concepts=list(common)[:10],
                               overlap=round(overlap_ratio, 4))
                    G.add_edge(ids[j], ids[i], relationship="TOPIC_BRIDGE",
                               shared_concepts=list(common)[:10],
                               overlap=round(overlap_ratio, 4))


def _add_definition_links(G, all_lines):
    """Find definition lines and link them to lines that later use that term."""
    definitions = {}  # term_lower -> line_id
    for line in all_lines:
        text = line.get("text", "")
        if _DEFINITION_PATTERNS.search(text):
            # Extract the term before the definition phrase
            m = re.match(r'^["\']?([\w\s]{2,30}?)["\']?\s+(?:is\s+(?:defined|a|an|the)|refers?\s+to|known\s+as)',
                         text, re.IGNORECASE)
            if m:
                term = m.group(1).strip().lower()
                if len(term) > 2:
                    definitions[term] = line["line_id"]

    if not definitions:
        return

    for line in all_lines:
        lid = line["line_id"]
        text_lower = line.get("text", "").lower()
        for term, def_lid in definitions.items():
            if def_lid != lid and term in text_lower:
                if not G.has_edge(def_lid, lid):
                    G.add_edge(def_lid, lid,
                               relationship="DEFINITION_USED_IN",
                               defined_term=term)

