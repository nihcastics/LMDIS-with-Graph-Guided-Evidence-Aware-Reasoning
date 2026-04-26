"""
Module 9b: Transformer Semantic Enrichment Pipeline
=====================================================
Deep semantic analysis of every element in the document memory graph.

Pipeline steps:
  1. Information Type Classification  — LLM batch classification with heuristic fallback
  2. Semantic Density Scoring         — How information-rich each element is
  3. Key Entity / Concept Extraction  — Per-section key concepts via LLM
  4. Hierarchical Propagation         — Roll up to paragraphs & sections
  5. Contextual Section Summaries     — Dense LLM summaries per section
  6. Table & Image Type Classification— Heuristic content-type labels
  7. Cross-Element Semantic Linking   — SEMANTICALLY_RELATED edges via embeddings

All enrichment data is stored as node attributes on the graph (G).
"""

import os
import re
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from backend.app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS

# Cached OpenAI client — reuse across all LLM calls
_openai_client = None

def _get_client():
    global _openai_client
    if _openai_client is None and OPENROUTER_API_KEY:
        _openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _openai_client

# ─── Information-type taxonomy ───────────────────────────────────────────────
INFORMATION_TYPES = [
    "definition",            # Defines a term or concept
    "fact",                  # States a verifiable factual claim
    "statistic",             # Contains numerical data or statistics
    "methodology",           # Describes a method, procedure, or algorithm
    "result",                # Presents experimental results or findings
    "conclusion",            # Draws conclusions from evidence
    "hypothesis",            # States a hypothesis, proposition, or research question
    "reference",             # Cites or references other work
    "background",            # Provides background / related work context
    "comparison",            # Compares two or more items
    "enumeration",           # Lists discrete items
    "example",               # Provides a concrete example
    "equation",              # Contains mathematical / formulaic content
    "metadata",              # Author info, dates, affiliations, headers, footers
    "table_data",            # Tabular information
    "figure_description",    # Describes a figure or image
]

_INFO_SET = set(INFORMATION_TYPES)

# ─── LLM helper ─────────────────────────────────────────────────────────────

def _call_llm(prompt, system_msg="You are a precise document analysis assistant.", max_tokens=2000):
    """Call OpenRouter with automatic model fallback."""
    if not OPENROUTER_API_KEY:
        return None

    models = [OPENROUTER_MODEL] + OPENROUTER_FALLBACK_MODELS

    for model_name in models:
        try:
            client = _get_client()
            if client is None:
                return None
            resp = client.chat.completions.create(
                model=model_name,
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
            # Strip any CoT / thinking wrappers
            content = re.sub(r'<(think|thought|reasoning)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<(think|thought|reasoning)>.*', '', content, flags=re.DOTALL | re.IGNORECASE)
            return content.strip()
        except Exception as e:
            if any(c in str(e) for c in ["429", "404", "400", "503", "502"]):
                continue
            return None
    return None


# ─── Ordered-node helper ────────────────────────────────────────────────────

def _build_ordered_nodes(G):
    """Return text_component nodes in reading order (page → NEXT_LINE chain)."""
    page_lines = {}
    for node, data in G.nodes(data=True):
        if data.get("type") == "text_component":
            page_lines.setdefault(data.get("page", 1), []).append(node)

    ordered = []
    for page in sorted(page_lines.keys()):
        nodes_on_page = set(page_lines[page])
        succ_map = {}
        pred_set = set()
        for node in nodes_on_page:
            for follower in G.successors(node):
                if G.edges[node, follower].get("relationship") == "NEXT_LINE" and follower in nodes_on_page:
                    succ_map[node] = follower
                    pred_set.add(follower)

        heads = [n for n in nodes_on_page if n not in pred_set]
        visited = set()
        for head in sorted(heads, key=lambda n: G.nodes[n].get("bbox", [0, 0, 0, 0])[1]):
            curr = head
            while curr and curr not in visited:
                visited.add(curr)
                ordered.append(curr)
                curr = succ_map.get(curr)
        for node in nodes_on_page:
            if node not in visited:
                ordered.append(node)
    return ordered


# ─── Step 1: Information-type classification ─────────────────────────────────

def _classify_information_types(G, ordered_nodes):
    """Classify every text_component via fast heuristic rules (zero LLM calls)."""

    _keyword_rules = [
        ("definition",  ["define", "defined as", "refers to", "is a", "known as", "is the"]),
        ("statistic",   ["%", "percent", "ratio", "average", "mean", "p-value", "p <", "p=", "standard deviation", "n ="]),
        ("methodology", ["method", "approach", "technique", "procedure", "algorithm", "we propose", "we use", "we employ"]),
        ("result",      ["result", "found that", "showed", "demonstrated", "observed", "achieved", "accuracy", "performance"]),
        ("conclusion",  ["conclude", "conclusion", "therefore", "thus", "hence", "in summary", "overall", "findings suggest"]),
        ("hypothesis",  ["hypothes", "assume", "proposit", "conjecture", "research question"]),
        ("reference",   ["et al", "doi:", "isbn", "journal", "[1]", "[2]", "cited", "references"]),
        ("equation",    ["equation", "formula", "∑", "∫", "≥", "≤"]),
        ("enumeration", ["(i)", "(ii)", "(a)", "(b)", "firstly", "secondly"]),
        ("comparison",  ["compare", "contrast", "versus", "while", "whereas", "outperform", "better than", "worse than"]),
        ("metadata",    ["abstract", "keywords", "affiliation", "corresponding author", "received", "accepted"]),
        ("background",  ["background", "related work", "previous", "prior work", "existing", "literature review"]),
        ("example",     ["example", "for instance", "such as", "e.g.", "illustrat", "consider the"]),
    ]

    for node in ordered_nodes:
        data = G.nodes[node]
        if data.get("type") != "text_component":
            continue
        text = data.get("text", "").strip()
        if not text:
            continue
        text_lower = text.lower()
        matched = False
        for itype, keywords in _keyword_rules:
            if any(kw in text_lower for kw in keywords):
                data["information_type"] = itype
                matched = True
                break
        if not matched:
            # Structural heuristics for metadata not covered by keyword rules
            page = data.get("page", 99)

            # Email addresses are always document metadata
            if re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text):
                data["information_type"] = "metadata"

            # Affiliation / institution lines on early pages
            elif page <= 2 and len(text.split()) <= 25 and re.search(
                r'\b(?:University|Institute|School\s+of|Department\s+of|'
                r'College|Laboratory|Research\s+(?:Center|Centre|Institute)|'
                r'VIT|IIT|MIT|UCLA|IEEE|ACM)\b',
                text,
            ):
                data["information_type"] = "metadata"

            # Short proper-name lines on early pages (author name candidates)
            # Matches: "Sachin S", "John Smith", "J. Smith", "A. B. Kumar"
            elif page <= 2 and re.match(
                r'^[A-Z][a-z]*\.?(?:\s+[A-Z][a-z]*\.?){1,4}\s*$', text.strip()
            ):
                data["information_type"] = "metadata"

            else:
                data["information_type"] = "fact"

    return G


# ─── Step 2: Semantic density scoring ────────────────────────────────────────

_TYPE_WEIGHTS = {
    "definition": 0.15, "statistic": 0.15, "result": 0.13,
    "methodology": 0.10, "conclusion": 0.13, "hypothesis": 0.10,
    "comparison": 0.08,  "equation": 0.10,   "fact": 0.05,
    "background": 0.03,  "reference": 0.02,   "metadata": 0.00,
    "enumeration": 0.05, "example": 0.05,     "table_data": 0.10,
    "figure_description": 0.08,
}


def _compute_semantic_density(G, ordered_nodes):
    """Score each text_component on a 0-1 information-density scale."""
    for node in ordered_nodes:
        data = G.nodes[node]
        if data.get("type") != "text_component":
            continue
        text = data.get("text", "").strip()
        if not text:
            data["semantic_density"] = 0.0
            continue

        words = text.split()
        wc = len(words)
        score = 0.50

        # Capitalized words (approximate named entities)
        caps = sum(1 for w in words[1:] if w and w[0].isupper())
        score += min(0.12, caps * 0.025)

        # Numbers / statistics
        nums = sum(1 for w in words if any(c.isdigit() for c in w))
        score += min(0.12, nums * 0.04)

        # Technical terms (long words ≥ 9 chars)
        tech = sum(1 for w in words if len(w) >= 9)
        score += min(0.08, tech * 0.02)

        # Information-type weight
        score += _TYPE_WEIGHTS.get(data.get("information_type", "fact"), 0.0)

        # Penalise very short lines
        if wc < 3:
            score *= 0.3

        data["semantic_density"] = round(min(1.0, max(0.0, score)), 4)

    return G


# ─── Step 3+5 Combined: Key concepts + section summaries in one LLM call ────

def _extract_concepts_and_summaries(G):
    """Extract key concepts AND generate summaries per section in parallel.

    Combines the old steps 3 and 5 into a single LLM call per section,
    halving the total per-section LLM calls. Sections are processed in
    parallel using ThreadPoolExecutor.
    """
    sections = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "section"]

    # Collect section data upfront
    section_tasks = []
    for s_id, s_data in sections:
        title = s_data.get("label", "Untitled")
        lines = []
        for pred in G.predecessors(s_id):
            if G.nodes[pred].get("type") == "text_component":
                lines.append(G.nodes[pred].get("text", ""))
        if len(lines) < 2:
            continue

        text_block = " ".join(lines)
        if len(text_block) > 5000:
            text_block = text_block[:5000]
        section_tasks.append((s_id, s_data, title, text_block))

    # Limit to top 5 sections by content length for speed
    section_tasks.sort(key=lambda t: len(t[3]), reverse=True)
    section_tasks = section_tasks[:5]

    def _process_section(task):
        s_id, _, title, text_block = task
        prompt = (
            f"Analyze this document section and provide:\n"
            f"1. Key concepts (max 10)\n"
            f"2. Information flow (1 sentence)\n"
            f"3. A precise 2-3 sentence summary capturing ALL key claims, numbers, and entities\n\n"
            f"Section: {title}\nContent:\n{text_block}\n\n"
            'Respond with ONLY a JSON object:\n'
            '{"concepts": ["concept1", ...], '
            '"information_flow": "brief description", '
            '"summary": "2-3 sentence summary"}\n'
        )
        result = _call_llm(prompt, max_tokens=600)
        return s_id, result

    # Process sections in parallel (up to 4 concurrent LLM calls)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_process_section, task) for task in section_tasks]
        for future in as_completed(futures):
            s_id, result = future.result()
            s_data = G.nodes[s_id]
            if result:
                try:
                    m = re.search(r'\{.*\}', result, re.DOTALL)
                    if m:
                        parsed = json.loads(m.group())
                        s_data["key_concepts"] = parsed.get("concepts", [])[:10]
                        s_data["information_flow"] = parsed.get("information_flow", "")
                        summary = parsed.get("summary", "")
                        if summary:
                            s_data["semantic_summary"] = summary
                except (json.JSONDecodeError, KeyError):
                    pass
    return G


# ─── Step 4: Propagate to paragraphs & sections ─────────────────────────────

def _propagate_enrichment(G):
    """Roll up information types and density scores to paragraphs and sections."""

    # ── Paragraph aggregation ──
    for node, data in G.nodes(data=True):
        if data.get("type") != "paragraph":
            continue
        types, densities = [], []
        for neighbor in G.successors(node):
            edge = G.edges.get((node, neighbor), {})
            if edge.get("relationship") == "PARAGRAPH_CONTAINS":
                nd = G.nodes[neighbor]
                if "information_type" in nd:
                    types.append(nd["information_type"])
                if "semantic_density" in nd:
                    densities.append(nd["semantic_density"])
        if types:
            tc = Counter(types)
            data["dominant_info_type"] = tc.most_common(1)[0][0]
            data["info_type_distribution"] = dict(tc)
        if densities:
            data["avg_semantic_density"] = round(sum(densities) / len(densities), 4)
            data["max_semantic_density"] = round(max(densities), 4)

    # ── Section aggregation ──
    for node, data in G.nodes(data=True):
        if data.get("type") != "section":
            continue
        types, densities = [], []
        for pred in G.predecessors(node):
            nd = G.nodes[pred]
            if nd.get("type") == "text_component":
                if "information_type" in nd:
                    types.append(nd["information_type"])
                if "semantic_density" in nd:
                    densities.append(nd["semantic_density"])
        if types:
            tc = Counter(types)
            data["dominant_info_type"] = tc.most_common(1)[0][0]
            data["info_type_distribution"] = dict(tc)
        if densities:
            data["avg_semantic_density"] = round(sum(densities) / len(densities), 4)
            data["max_semantic_density"] = round(max(densities), 4)

    return G


# ─── Step 6: Table & image content-type classification ──────────────────────

def _classify_tables_and_images(G):
    """Assign content-type labels to tables and images."""

    # Tables
    for node, data in G.nodes(data=True):
        if data.get("type") != "table":
            continue
        cell_texts = []
        for neighbor in G.successors(node):
            if G.edges[node, neighbor].get("relationship") == "TABLE_CELL_OF":
                cell_texts.append(G.nodes[neighbor].get("text", ""))
        combined = " ".join(cell_texts).lower()

        if any(w in combined for w in ["accuracy", "precision", "recall", "f1", "auc", "score"]):
            data["table_type"] = "performance_metrics"
        elif any(w in combined for w in ["mean", "std", "median", "average", "variance"]):
            data["table_type"] = "statistical_summary"
        elif any(w in combined for w in ["author", "year", "title", "journal", "reference"]):
            data["table_type"] = "literature_review"
        elif any(w in combined for w in ["parameter", "setting", "value", "config", "hyperparameter"]):
            data["table_type"] = "configuration"
        else:
            data["table_type"] = "data_table"
        data["information_type"] = "table_data"

    # Images
    for node, data in G.nodes(data=True):
        if data.get("type") != "image_component":
            continue
        desc = data.get("llm_description", "").lower()

        if any(w in desc for w in ["graph", "chart", "plot", "bar", "pie", "histogram", "line graph"]):
            data["image_type"] = "chart"
        elif any(w in desc for w in ["diagram", "flowchart", "architecture", "pipeline", "workflow"]):
            data["image_type"] = "diagram"
        elif any(w in desc for w in ["equation", "formula", "mathematical"]):
            data["image_type"] = "equation"
        elif any(w in desc for w in ["screenshot", "interface", "ui", "gui"]):
            data["image_type"] = "screenshot"
        elif any(w in desc for w in ["photo", "photograph"]):
            data["image_type"] = "photograph"
        else:
            data["image_type"] = "figure"
        data["information_type"] = "figure_description"

    return G


# ─── Step 7: Cross-element semantic linking ──────────────────────────────────

def _detect_semantic_links(G, embedding_manager):
    """Add SEMANTICALLY_RELATED edges between high-density nodes in different sections."""
    if embedding_manager is None or embedding_manager.counter == 0:
        return G

    high_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "text_component" and d.get("semantic_density", 0) >= 0.65
    ]

    # Tight scope for speed: top 15 nodes × k=3
    for node in high_nodes[:15]:
        text = G.nodes[node].get("text", "")
        section_id = G.nodes[node].get("section_id")

        results = embedding_manager.search(text, k=3)
        for other_node, distance in results:
            if other_node == node:
                continue
            other_section = G.nodes.get(other_node, {}).get("section_id")
            # Only cross-section links with meaningful similarity
            if other_section and other_section != section_id and distance < 0.35:
                if not G.has_edge(node, other_node):
                    G.add_edge(
                        node, other_node,
                        relationship="SEMANTICALLY_RELATED",
                        similarity=round(1.0 - distance, 4),
                    )
    return G


# ─── Public entry-point ─────────────────────────────────────────────────────

def enrich_semantics(G, embedding_manager=None):
    """
    Module 9b: Full Semantic Enrichment Pipeline.

    Enriches every node in the document memory graph with:
      • information_type          (text, table, image nodes)
      • semantic_density          (text nodes: 0-1 scale)
      • dominant_info_type        (paragraph & section nodes)
      • info_type_distribution    (paragraph & section nodes)
      • avg / max semantic_density(paragraph & section nodes)
      • key_concepts              (section nodes)
      • information_flow          (section nodes)
      • semantic_summary          (section nodes)
      • table_type / image_type   (table / image nodes)
      • SEMANTICALLY_RELATED edges(cross-section high-density nodes)
    """
    print("[M9b] Starting semantic enrichment pipeline...")

    ordered = _build_ordered_nodes(G)

    print(f"[M9b] Step 1/6: Classifying information types ({len(ordered)} lines)...")
    G = _classify_information_types(G, ordered)

    print("[M9b] Step 2/6: Computing semantic density scores...")
    G = _compute_semantic_density(G, ordered)

    print("[M9b] Step 3/6: Extracting concepts + summaries per section (parallel)...")
    G = _extract_concepts_and_summaries(G)

    print("[M9b] Step 4/6: Propagating enrichment to paragraphs & sections...")
    G = _propagate_enrichment(G)

    print("[M9b] Step 5/6: Classifying tables & images...")
    G = _classify_tables_and_images(G)

    print("[M9b] Step 6/6: Detecting cross-element semantic links...")
    G = _detect_semantic_links(G, embedding_manager)

    print("[M9b] Semantic enrichment complete.")
    return G
