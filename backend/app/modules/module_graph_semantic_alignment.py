"""
Module GSA: LLM-Based Graph Semantic Alignment Layer
======================================================
Runs during ingestion after the evidence graph is built.

Compares document content and arranges the graph using LLM reasoning
to frame information in a logical, semantically coherent structure.

Pipeline:
  1. Collect all text nodes grouped by section
  2. Send document summary + node samples to LLM for theme identification
  3. Create semantic_theme nodes representing logical concepts
  4. Link text nodes to their themes via THEME_MEMBER edges
  5. Create SEMANTIC_ALIGNMENT edges between related nodes across sections
  6. Store theme metadata for retrieval-time expansion

This layer enables the retrieval pipeline to find contextually related
content even when queries and document text use different vocabulary.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from backend.app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_STRONG_MODEL,
    OPENROUTER_STRONG_FALLBACK_MODELS,
    OPENROUTER_MODEL,
    OPENROUTER_FALLBACK_MODELS,
    GOOGLE_API_KEY,
    GOOGLE_STRONG_MODEL,
)

_llm_client = None
_google_client = None


def _get_client():
    global _llm_client
    if _llm_client is None and OPENROUTER_API_KEY:
        _llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _llm_client


def _get_google_client():
    global _google_client
    if _google_client is None and GOOGLE_API_KEY:
        _google_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GOOGLE_API_KEY,
        )
    return _google_client


def _call_llm(prompt, system_msg="You are a precise document analysis assistant.", max_tokens=1200):
    """Call Google Gemini 2.5 Pro directly, with OpenRouter as fallback."""
    if not GOOGLE_API_KEY and not OPENROUTER_API_KEY:
        return None

    # Try Google first
    if GOOGLE_API_KEY:
        try:
            gclient = _get_google_client()
            if gclient:
                resp = gclient.chat.completions.create(
                    model=GOOGLE_STRONG_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content.strip()
                content = re.sub(r'<(think|thought|reasoning)>.*?</\1>', '', content, flags=re.DOTALL | re.I)
                content = re.sub(r'<(think|thought|reasoning)>.*', '', content, flags=re.DOTALL | re.I)
                if content.strip():
                    return content.strip()
        except Exception as e:
            print(f"[GSA] Google error, falling back to OpenRouter: {e}")

    models = [OPENROUTER_STRONG_MODEL] + OPENROUTER_STRONG_FALLBACK_MODELS + [OPENROUTER_MODEL] + OPENROUTER_FALLBACK_MODELS

    for model in models:
        try:
            client = _get_client()
            if client is None:
                return None
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Document Intelligence System",
                },
            )
            content = resp.choices[0].message.content.strip()
            content = re.sub(r'<(think|thought|reasoning)>.*?</\1>', '', content, flags=re.DOTALL | re.I)
            content = re.sub(r'<(think|thought|reasoning)>.*', '', content, flags=re.DOTALL | re.I)
            return content.strip()
        except Exception as e:
            if any(c in str(e) for c in ["429", "404", "400", "401", "402", "503", "502"]):
                continue
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main entry-point
# ═══════════════════════════════════════════════════════════════════════════

def align_graph_semantics(G, doc_meta=None):
    """
    LLM-based graph semantic alignment.

    1. Collects document content grouped by section
    2. Identifies semantic themes via LLM
    3. Creates theme nodes and membership edges
    4. Creates cross-section alignment edges

    Returns the mutated graph G.
    """
    print("[GSA] Starting LLM-based graph semantic alignment...")

    # ── 1. Collect section content ──
    sections = _collect_sections(G)
    if not sections:
        print("[GSA] No sections found, skipping alignment.")
        return G

    # ── 2. Build document overview for theme identification ──
    doc_overview = _build_document_overview(G, sections, doc_meta)

    # ── 3. Identify semantic themes via LLM ──
    themes = _identify_themes(doc_overview, sections)
    if not themes:
        print("[GSA] LLM theme identification failed, using heuristic fallback.")
        themes = _heuristic_theme_identification(sections)

    if not themes:
        print("[GSA] No themes identified, skipping alignment.")
        return G

    print(f"[GSA] Identified {len(themes)} semantic themes.")

    # ── 4. Create theme nodes ──
    for i, theme in enumerate(themes):
        theme_id = f"semantic_theme_{i}"
        G.add_node(theme_id,
                   type="semantic_theme",
                   label=theme["label"],
                   description=theme.get("description", ""),
                   keywords=theme.get("keywords", []),
                   related_concepts=theme.get("related_concepts", []))

        # Link theme to its member sections
        for sec_id in theme.get("section_ids", []):
            if sec_id in G:
                G.add_edge(theme_id, sec_id, relationship="THEME_SECTION")

    # ── 5. Assign text nodes to themes ──
    _assign_nodes_to_themes(G, themes)

    # ── 6. Create cross-section semantic alignment edges ──
    _create_alignment_edges(G, themes)

    theme_labels = [t["label"] for t in themes]
    print(f"[GSA] Semantic alignment complete. Themes: {theme_labels}")
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _collect_sections(G):
    """Collect text content organized by section."""
    sections = {}
    for node, data in G.nodes(data=True):
        if data.get("type") != "section":
            continue
        sec_id = node
        label = data.get("label", "Untitled")
        lines = []
        for pred in G.predecessors(sec_id):
            nd = G.nodes.get(pred, {})
            if nd.get("type") == "text_component":
                text = nd.get("text", "").strip()
                if text and len(text) > 10:
                    lines.append({"node_id": pred, "text": text, "page": nd.get("page", 0)})
        if lines:
            sections[sec_id] = {
                "label": label,
                "page": data.get("page", 0),
                "lines": lines,
            }
    return sections


def _build_document_overview(G, sections, doc_meta):
    """Build a concise document overview for LLM consumption."""
    title = (doc_meta or {}).get("detected_title", "Unknown Document")
    parts = [f"Document: {title}\n"]

    for sec_id, sec_data in list(sections.items())[:15]:
        label = sec_data["label"]
        # Take first few lines from each section (max 200 chars total per section)
        sample_text = " ".join(l["text"] for l in sec_data["lines"][:5])
        if len(sample_text) > 300:
            sample_text = sample_text[:300] + "..."
        parts.append(f"[{label}] {sample_text}")

    return "\n\n".join(parts)


def _identify_themes(doc_overview, sections):
    """Use LLM to identify semantic themes across the document."""
    section_list = "\n".join(
        f"- {sid}: \"{sdata['label']}\" ({len(sdata['lines'])} lines)"
        for sid, sdata in list(sections.items())[:20]
    )

    prompt = (
        "Analyze this document and identify 4-8 semantic themes that organize "
        "the information logically. Each theme should group related concepts "
        "that a user might ask about.\n\n"
        f"{doc_overview}\n\n"
        f"Sections:\n{section_list}\n\n"
        "For each theme, provide:\n"
        "1. A concise label (2-5 words)\n"
        "2. A description (1 sentence)\n"
        "3. Keywords that belong to this theme (5-10 words)\n"
        "4. Related concepts a user might ask about (3-5 phrases)\n"
        "5. Which section IDs belong to this theme\n\n"
        "IMPORTANT: Always include a 'Document Metadata' theme covering author, "
        "date, title, publisher, affiliation, and other document properties.\n\n"
        "Return ONLY a JSON array:\n"
        '[{"label": "...", "description": "...", "keywords": [...], '
        '"related_concepts": [...], "section_ids": [...]}]'
    )

    result = _call_llm(prompt, max_tokens=1500)
    if not result:
        return None

    try:
        m = re.search(r'\[.*\]', result, re.DOTALL)
        if m:
            themes = json.loads(m.group())
            if isinstance(themes, list) and len(themes) >= 1:
                # Validate and clean themes
                valid_section_ids = set(sections.keys())
                for theme in themes:
                    theme["section_ids"] = [
                        s for s in theme.get("section_ids", [])
                        if s in valid_section_ids
                    ]
                    theme["keywords"] = theme.get("keywords", [])[:15]
                    theme["related_concepts"] = theme.get("related_concepts", [])[:8]
                return themes
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _heuristic_theme_identification(sections):
    """Fallback theme identification using keyword patterns when LLM fails."""
    themes = []

    # Always create a metadata theme
    metadata_keywords = [
        "author", "title", "date", "published", "journal", "affiliation",
        "university", "email", "doi", "abstract", "keywords", "institution",
        "department", "corresponding", "editor", "volume", "issue",
    ]
    themes.append({
        "label": "Document Metadata",
        "description": "Author, publication, and document identity information",
        "keywords": metadata_keywords,
        "related_concepts": [
            "who wrote this", "author name", "publication date",
            "where published", "document title",
        ],
        "section_ids": [],
    })

    # Scan sections for common themes
    _theme_patterns = {
        "Introduction & Background": {
            "triggers": ["introduction", "background", "overview", "related work", "literature"],
            "keywords": ["introduction", "background", "overview", "context", "motivation", "prior work"],
            "concepts": ["what is this about", "paper overview", "research context"],
        },
        "Methodology & Approach": {
            "triggers": ["method", "approach", "technique", "procedure", "algorithm", "design", "implementation"],
            "keywords": ["method", "approach", "technique", "algorithm", "design", "procedure", "framework"],
            "concepts": ["how was this done", "research method", "approach used"],
        },
        "Results & Findings": {
            "triggers": ["result", "finding", "experiment", "evaluation", "performance", "analysis"],
            "keywords": ["result", "finding", "performance", "accuracy", "experiment", "evaluation", "outcome"],
            "concepts": ["what were the results", "key findings", "performance achieved"],
        },
        "Conclusion & Discussion": {
            "triggers": ["conclusion", "discussion", "summary", "future work", "limitation"],
            "keywords": ["conclusion", "discussion", "summary", "future", "limitation", "implication"],
            "concepts": ["main conclusion", "key takeaways", "future directions"],
        },
    }

    for theme_label, pattern_data in _theme_patterns.items():
        matched_sections = []
        for sec_id, sec_data in sections.items():
            sec_label_lower = sec_data["label"].lower()
            sec_text_sample = " ".join(l["text"] for l in sec_data["lines"][:3]).lower()
            combined = sec_label_lower + " " + sec_text_sample
            if any(trigger in combined for trigger in pattern_data["triggers"]):
                matched_sections.append(sec_id)
        if matched_sections:
            themes.append({
                "label": theme_label,
                "description": f"Content related to {theme_label.lower()}",
                "keywords": pattern_data["keywords"],
                "related_concepts": pattern_data["concepts"],
                "section_ids": matched_sections,
            })

    # Assign metadata sections
    for sec_id, sec_data in sections.items():
        sec_label_lower = sec_data["label"].lower()
        sec_text = " ".join(l["text"] for l in sec_data["lines"][:5]).lower()
        if any(kw in sec_label_lower or kw in sec_text for kw in ["author", "abstract", "keyword", "affiliation"]):
            themes[0]["section_ids"].append(sec_id)

    return themes


def _assign_nodes_to_themes(G, themes):
    """Assign text_component nodes to their best-matching theme via keyword overlap."""
    theme_keyword_sets = []
    for i, theme in enumerate(themes):
        kw_set = set()
        for kw in theme.get("keywords", []):
            kw_set.update(kw.lower().split())
        for concept in theme.get("related_concepts", []):
            kw_set.update(concept.lower().split())
        theme_keyword_sets.append((i, kw_set))

    for node, data in G.nodes(data=True):
        if data.get("type") != "text_component":
            continue
        text = data.get("text", "").strip()
        if not text or len(text) < 10:
            continue
        text_lower = text.lower()
        text_words = set(re.findall(r'\w+', text_lower))

        best_theme_idx = -1
        best_overlap = 0
        for idx, kw_set in theme_keyword_sets:
            overlap = len(text_words & kw_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_theme_idx = idx

        # Also check section membership
        sec_id = data.get("section_id")
        for idx, theme in enumerate(themes):
            if sec_id in theme.get("section_ids", []):
                # Section membership gives a boost
                if idx == best_theme_idx or best_overlap < 2:
                    best_theme_idx = idx
                    best_overlap = max(best_overlap, 2)

        if best_theme_idx >= 0 and best_overlap >= 1:
            theme_id = f"semantic_theme_{best_theme_idx}"
            if theme_id in G:
                G.add_edge(theme_id, node, relationship="THEME_MEMBER")


def _create_alignment_edges(G, themes):
    """Create SEMANTIC_ALIGNMENT edges between nodes in different sections
    that belong to the same theme.  This enables cross-section retrieval
    when a query matches content spread across multiple sections."""
    for i, theme in enumerate(themes):
        theme_id = f"semantic_theme_{i}"
        if theme_id not in G:
            continue

        # Collect member nodes grouped by section
        section_members = {}
        for member in G.successors(theme_id):
            edge = G.edges.get((theme_id, member), {})
            if edge.get("relationship") != "THEME_MEMBER":
                continue
            sec_id = G.nodes.get(member, {}).get("section_id", "unknown")
            section_members.setdefault(sec_id, []).append(member)

        # Create alignment edges between representative nodes of different sections
        sec_ids = list(section_members.keys())
        for a_idx in range(len(sec_ids)):
            for b_idx in range(a_idx + 1, len(sec_ids)):
                nodes_a = section_members[sec_ids[a_idx]][:3]  # Top 3 per section
                nodes_b = section_members[sec_ids[b_idx]][:3]
                for na in nodes_a:
                    for nb in nodes_b:
                        if not G.has_edge(na, nb):
                            G.add_edge(na, nb,
                                       relationship="SEMANTIC_ALIGNMENT",
                                       theme=theme["label"])
