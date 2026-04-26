"""
Module 13-15: Deep Graph-Guided Evidence Retrieval
=====================================================
Multi-strategy retrieval pipeline:

1. Bi-encoder semantic search (bge-large-en-v1.5) with query expansion
2. Entity-targeted search using extracted key entities
3. Deep graph traversal through 10+ edge types
4. Section-concept and evidence-summary anchor expansion
5. Semantic enrichment boosting (info-type, density, answer-type)
6. Cross-encoder re-ranking (ms-marco-MiniLM-L-12-v2)
7. LLM-powered semantic re-ranking for deep contextual understanding
8. Metadata-targeted scanning for document properties (author, date, etc.)
9. Fuzzy keyword matching with prefix/stem overlap
10. Context-aware evidence assembly for the generation layer
"""

import json
import math
import re

from openai import OpenAI
from backend.app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_STRONG_MODEL,
    OPENROUTER_STRONG_FALLBACK_MODELS,
    GOOGLE_API_KEY,
    GOOGLE_STRONG_MODEL,
)


def _text_match_score(query_text, node_text):
    """Simple text overlap score between query and node text (0 to 1)."""
    if not query_text or not node_text:
        return 0.0
    q_lower = query_text.lower()
    n_lower = node_text.lower()
    q_tokens = set(w for w in re.split(r'\W+', q_lower) if len(w) > 2)
    n_tokens = set(w for w in re.split(r'\W+', n_lower) if len(w) > 2)
    if not q_tokens:
        return 0.0
    # Token overlap ratio
    overlap = q_tokens & n_tokens
    token_ratio = len(overlap) / len(q_tokens)
    # Exact substring bonus for multi-word query phrases
    exact_bonus = 0.0
    q_words = q_lower.split()
    for i in range(len(q_words)):
        for j in range(i + 2, min(i + 5, len(q_words) + 1)):
            phrase = ' '.join(q_words[i:j])
            if len(phrase) > 5 and phrase in n_lower:
                exact_bonus = max(exact_bonus, 0.3 * (j - i) / len(q_words))
    # Rare / long term bonus
    rare_bonus = sum(0.1 for w in overlap if len(w) >= 6)
    return min(1.0, token_ratio + exact_bonus + min(rare_bonus, 0.3))


def _deduplicate_results(results):
    """Remove results with heavily overlapping text to ensure diverse context."""
    if not results:
        return results
    deduped = [results[0]]
    for item in results[1:]:
        item_words = set(item["text"].lower().split())
        is_dup = False
        for kept in deduped:
            kept_words = set(kept["text"].lower().split())
            if not item_words or not kept_words:
                continue
            overlap = len(item_words & kept_words) / min(len(item_words), len(kept_words))
            if overlap > 0.6:
                is_dup = True
                break
        if not is_dup:
            deduped.append(item)
    return deduped


# ── Module-level singleton: loaded once, reused on every query ──────────────
_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            import logging
            logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
                max_length=512,
            )
            print("[M13] Cross-encoder loaded: ms-marco-MiniLM-L-12-v2")
        except Exception as e:
            print(f"[M13] Cross-encoder unavailable ({e})")
    return _CROSS_ENCODER


def _cross_encoder_rerank(query, items, top_n=30):
    reranker = _get_cross_encoder()
    if reranker is None:
        return items[:top_n]
    try:
        pairs = [(query, c["text"]) for c in items]
        scores = reranker.predict(pairs, show_progress_bar=False)
        for i, s in enumerate(scores):
            items[i]["cross_encoder_score"] = float(s)
        items.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return items[:top_n]
    except Exception:
        return items[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# LLM-powered semantic re-ranking (deep contextual understanding)
# ═══════════════════════════════════════════════════════════════════════════

_llm_client = None
_google_client = None


def _get_llm_client():
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


def _llm_semantic_rerank(query, items, top_n=8):
    """
    Send the top retrieval candidates to a strong LLM for semantic relevance
    scoring.  This catches cases where the bi-encoder and keyword matching
    fail due to vocabulary mismatch (e.g. 'author' vs 'written by John').

    The LLM assigns a 0-10 relevance score to each candidate considering
    contextual meaning, not just lexical overlap.
    """
    if (not OPENROUTER_API_KEY and not GOOGLE_API_KEY) or not items:
        return items[:top_n]

    # Prepare a numbered list of candidate texts (max 20 to fit in context)
    candidates_block = []
    for i, item in enumerate(items[:20]):
        text_snip = item["text"][:300].strip()
        candidates_block.append(f"[{i}] {text_snip}")
    candidates_text = "\n".join(candidates_block)

    prompt = (
        "You are a document retrieval relevance judge. Score each candidate passage's "
        "relevance to the query on a scale of 0-10.\n\n"
        "IMPORTANT: Consider SEMANTIC meaning, not just keyword matches. For example:\n"
        "- 'Who is the author?' is relevant to text containing names near the document title\n"
        "- 'What year was this published?' is relevant to text with dates\n"
        "- Metadata, headers, and attribution information are relevant to identity questions\n\n"
        f"Query: \"{query}\"\n\n"
        f"Candidates:\n{candidates_text}\n\n"
        "Return ONLY a JSON array of objects with 'id' (int) and 'score' (0-10).\n"
        "Example: [{\"id\": 0, \"score\": 9}, {\"id\": 1, \"score\": 2}]\n"
        "Score ALL candidates. Return ONLY the JSON array."
    )

    system_msg = "You are a precise relevance scoring assistant. Return only JSON."

    def _apply_scores(content, model_label):
        content = re.sub(r'<(think|thought|reasoning)>.*?</\1>', '', content, flags=re.DOTALL | re.I)
        content = re.sub(r'<(think|thought|reasoning)>.*', '', content, flags=re.DOTALL | re.I)
        content = content.strip()
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if m:
            scores = json.loads(m.group())
            score_map = {}
            for entry in scores:
                idx = entry.get("id", -1)
                sc = entry.get("score", 0)
                if 0 <= idx < len(items):
                    score_map[idx] = float(sc) / 10.0
            for i in range(min(len(items), 20)):
                if i in score_map:
                    llm_sc = score_map[i]
                    orig = items[i]["alignment_score"]
                    items[i]["llm_relevance"] = llm_sc
                    items[i]["alignment_score"] = min(1.0, 0.45 * llm_sc + 0.35 * orig + 0.20 * items[i].get("cross_encoder_score_norm", orig))
            items.sort(key=lambda x: x["alignment_score"], reverse=True)
            print(f"[M13] LLM semantic rerank applied ({model_label})")
            return True
        return False

    # Try Google Gemini 2.5 Pro directly first
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
                    max_tokens=800,
                )
                if _apply_scores(resp.choices[0].message.content.strip(), GOOGLE_STRONG_MODEL):
                    return items[:top_n]
        except Exception as e:
            print(f"[M13] Google rerank error, falling back: {e}")

    models = [OPENROUTER_STRONG_MODEL] + OPENROUTER_STRONG_FALLBACK_MODELS
    for model in models:
        try:
            client = _get_llm_client()
            if client is None:
                return items[:top_n]
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=800,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Document Intelligence System",
                },
            )
            if _apply_scores(resp.choices[0].message.content.strip(), model):
                return items[:top_n]
        except Exception as e:
            err = str(e)
            if any(c in err for c in ("429", "404", "400", "401", "402", "503", "502")):
                continue
            print(f"[M13] LLM rerank error: {e}")
            break
    return items[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# Metadata-targeted scan (author, title, date, etc.)
# ═══════════════════════════════════════════════════════════════════════════

_METADATA_TRIGGERS = {
    "author":      ["author", "authors", "written by", "by", "writer", "contributor", "creator", "created by", "developed by", "built by", "made by", "student", "students", "team", "developers", "project team"],
    "title":       ["title", "heading", "name of the document", "called"],
    "date":        ["date", "published", "year", "when", "issued", "submitted", "received", "accepted"],
    "publisher":   ["publisher", "journal", "publication", "venue", "conference", "press"],
    "affiliation": ["affiliation", "university", "institution", "department", "organization", "school"],
    "email":       ["email", "e-mail", "contact", "mail"],
    "doi":         ["doi", "identifier", "isbn", "issn"],
    "abstract":    ["abstract", "summary", "overview"],
    "keyword":     ["keyword", "keywords", "tags", "terms", "index terms"],
}


def _detect_metadata_intent(query_text):
    """Detect if the query is asking about document metadata properties."""
    q_lower = query_text.lower()
    triggered = set()
    for category, triggers in _METADATA_TRIGGERS.items():
        for trigger in triggers:
            if trigger in q_lower:
                triggered.add(category)
    return triggered


def _metadata_targeted_scan(G, candidates, query_text, structured_query):
    """
    When the query asks about document metadata (author, title, date, etc.),
    scan ALL text nodes for metadata patterns and boost matches heavily.

    This ensures that 'Who is the author?' finds 'John Smith' even when
    bi-encoder fails due to vocabulary mismatch.
    """
    meta_intents = _detect_metadata_intent(query_text)
    if not meta_intents:
        return

    # Patterns for metadata extraction
    _meta_patterns = {
        "author": [
            re.compile(r'(?:author|authors|written\s+by|by|created\s+by|developed\s+by|built\s+by|made\s+by|team\s+members?)\s*[:\-–—]\s*(.+)', re.I),
            re.compile(r'^([A-Z][a-z]*\.?(?:\s+[A-Z][a-z]*\.?){1,4})\s*$'),  # standalone proper names incl. initials (e.g. "Sachin S", "J. Smith")
            re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]*\.?)', re.I),  # "John A. Smith", "John S"
            re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),  # email address → co-located with author name
            re.compile(r'(?:University|Institute|Department|School|College|Laboratory|Research)\s+(?:of\s+)?[\w\s,]+', re.I),  # affiliation lines
        ],
        "title": [
            re.compile(r'(?:title)\s*[:\-–—]\s*(.+)', re.I),
        ],
        "date": [
            re.compile(r'(?:date|published|year|received|accepted|submitted)\s*[:\-–—]\s*(.+)', re.I),
            re.compile(r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', re.I),
            re.compile(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b'),
        ],
        "publisher": [
            re.compile(r'(?:journal|publisher|published\s+(?:in|by)|venue|conference)\s*[:\-–—]\s*(.+)', re.I),
        ],
        "affiliation": [
            re.compile(r'(?:affiliation|department|university|institution|school)\s*[:\-–—]\s*(.+)', re.I),
            re.compile(r'(?:University|Institute|Department|School|College|Laboratory)\s+(?:of\s+)?[\w\s]+', re.I),
        ],
        "email": [
            re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),
        ],
        "doi": [
            re.compile(r'(?:doi|DOI)\s*[:\-–—]?\s*(10\.\S+)', re.I),
            re.compile(r'10\.\d{4,}/\S+'),
        ],
        "abstract": [
            re.compile(r'(?:abstract)\s*[:\-–—]\s*(.+)', re.I),
        ],
        "keyword": [
            re.compile(r'(?:keywords?|index\s+terms?|tags)\s*[:\-–—]\s*(.+)', re.I),
        ],
    }

    for node, data in G.nodes(data=True):
        if data.get("type") != "text_component":
            continue
        text = data.get("text", "").strip()
        if not text or len(text) < 3:
            continue

        # Check the node's information_type attribute
        node_info_type = data.get("information_type", "")
        is_metadata_type = node_info_type == "metadata"

        # Check if node is on early pages (where metadata typically lives)
        page = data.get("page", 99)
        is_early_page = page <= 2

        for intent in meta_intents:
            patterns = _meta_patterns.get(intent, [])
            matched = False
            for pat in patterns:
                if pat.search(text):
                    matched = True
                    break

            if matched:
                # Strong boost for pattern match — raised for early pages so
                # metadata nodes compete with abstract / body paragraphs
                score = 0.90 if is_early_page else 0.65
                if is_metadata_type:
                    score = min(1.0, score + 0.05)
                _update(candidates, node, score)
            elif is_early_page and intent == "author":
                # For author queries also pull in any early-page node that looks
                # like it could be metadata (fact or metadata type, short text)
                word_count = len(text.split())
                if node_info_type in ("metadata", "fact") and word_count <= 15:
                    _update(candidates, node, 0.45)
            elif is_metadata_type and is_early_page:
                # Weaker boost for metadata-type nodes on early pages
                _update(candidates, node, 0.45)


# ═══════════════════════════════════════════════════════════════════════════
# Main retrieval entry-point
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_evidence(G, embedding_manager, structured_query):
    """
    Returns a list of evidence dicts ready for the generation layer.

    Improvements over baseline:
    - Searches all claims + expanded terms for broader recall
    - Entity-targeted search for precision on specific concepts
    - Answer-type-aware boosting (e.g., boost statistics for quantitative)
    - Section-concept matching leveraging semantic enrichment
    - Metadata-targeted scanning for document property queries
    - LLM-powered semantic re-ranking for deep contextual understanding
    - Fuzzy keyword matching with prefix/stem overlap
    - Fixed cross-encoder score combination
    """
    candidates = {}  # node_id -> peak_score
    intent = structured_query.get("intent", "factual_lookup")
    query_text = structured_query["original_text"]
    key_entities = structured_query.get("key_entities", [])
    expected_answer_type = structured_query.get("expected_answer_type", "description")
    expanded_terms = structured_query.get("expanded_terms", [])

    # ── 1. Bi-encoder semantic search over claims ─────────────────────────
    search_k = _search_k_for_intent(intent)

    for sub_claim in structured_query.get("claims", [query_text]):
        results = embedding_manager.search(sub_claim, k=search_k)
        for node_id, distance in results:
            score = 1.0 / (1.0 + distance)
            score = _apply_node_boosts(G, node_id, score, intent, expected_answer_type)
            _update(candidates, node_id, score)
            if node_id in G:
                _traverse_edges(G, node_id, score, candidates)

    # ── 2. Entity-targeted search (no distance filter for lossless recall) ─
    for entity in key_entities[:5]:
        results = embedding_manager.search(entity, k=8)
        for node_id, distance in results:
            score = (1.0 / (1.0 + distance)) * 0.8
            score = _apply_node_boosts(G, node_id, score, intent, expected_answer_type)
            _update(candidates, node_id, score)

    # ── 3. Expanded-term search for broader recall ────────────────────────
    for term in expanded_terms[:5]:
        results = embedding_manager.search(term, k=8)
        for node_id, distance in results:
            score = (1.0 / (1.0 + distance)) * 0.65
            _update(candidates, node_id, score)

    # ── 3b. Keyword fallback: scan graph for direct query-term matches ────
    _keyword_recall(G, candidates, query_text, key_entities)

    # ── 3c. Metadata-targeted scan: author, date, title, etc. ────────────
    _metadata_targeted_scan(G, candidates, query_text, structured_query)

    # ── 3d. Semantic theme scan: query against graph semantic themes ──────
    _expand_via_semantic_themes(G, candidates, query_text, key_entities)

    # ── 4. Section-concept matching ───────────────────────────────────────
    _expand_via_section_concepts(G, candidates, key_entities, query_text)

    # ── 5. Evidence-summary anchor expansion ──────────────────────────────
    _expand_via_evidence_summaries(G, candidates, query_text, key_entities)

    # ── 6. Claim back-link expansion ─────────────────────────────────────
    _expand_via_claims(G, candidates)

    # ── 7. Consolidation & scoring ───────────────────────────────────────
    pre_rerank = _consolidate(G, candidates, structured_query, intent)

    # ── 8. Cross-encoder re-ranking with fixed score combination ─────────
    pre_rerank.sort(key=lambda x: x["alignment_score"], reverse=True)
    top_pool = pre_rerank[:40]
    if top_pool:
        reranked = _cross_encoder_rerank(query_text, top_pool, top_n=25)
        for item in reranked:
            ce = item.get("cross_encoder_score", 0.0)
            orig = item["alignment_score"]
            ce_norm = 1.0 / (1.0 + math.exp(-ce))
            item["cross_encoder_score_norm"] = ce_norm
            item["alignment_score"] = min(1.0, 0.55 * ce_norm + 0.45 * orig)
        reranked.sort(key=lambda x: x["alignment_score"], reverse=True)

        # ── 9. LLM semantic re-ranking for deep contextual understanding ──
        final = _llm_semantic_rerank(query_text, reranked, top_n=8)
        return final

    return pre_rerank[:8]


def _search_k_for_intent(intent):
    """Search pool sized by intent — generous for maximum recall."""
    return {
        "summarization": 50,
        "comparative": 45,
        "enumerative": 45,
        "causal": 40,
        "definition": 35,
        "procedural": 35,
    }.get(intent, 35)


def _apply_node_boosts(G, node_id, score, intent, expected_answer_type):
    """Apply intent-aware and answer-type-aware boosts to a candidate score."""
    if node_id not in G:
        return score

    nd = G.nodes[node_id]

    if intent == "summarization":
        if nd.get("page", 0) <= 2:
            score += 0.25
        if nd.get("type") == "section":
            score += 0.2

    score += nd.get("semantic_density", 0.5) * 0.04

    info_type = nd.get("information_type", "")
    _intent_boosts = {
        "factual_lookup": {"definition": 0.12, "fact": 0.10, "statistic": 0.10, "result": 0.08},
        "comparative":    {"comparison": 0.15, "result": 0.10, "statistic": 0.10, "table_data": 0.08},
        "summarization":  {"conclusion": 0.12, "result": 0.10, "methodology": 0.08, "background": 0.06},
        "quantitative":   {"statistic": 0.18, "result": 0.12, "table_data": 0.15, "fact": 0.08},
        "definition":     {"definition": 0.18, "fact": 0.10, "background": 0.08},
        "causal":         {"result": 0.12, "conclusion": 0.12, "methodology": 0.10, "fact": 0.08},
        "procedural":     {"methodology": 0.18, "enumeration": 0.10, "fact": 0.08},
        "enumerative":    {"enumeration": 0.15, "fact": 0.10, "table_data": 0.10},
    }
    score += _intent_boosts.get(intent, {}).get(info_type, 0.0)

    # Boost nodes whose info type aligns with expected answer type
    _answer_type_boosts = {
        "number":      {"statistic": 0.10, "table_data": 0.10, "result": 0.06},
        "percentage":  {"statistic": 0.12, "result": 0.08},
        "date":        {"fact": 0.08, "metadata": 0.06},
        "definition":  {"definition": 0.12, "background": 0.06},
        "list":        {"enumeration": 0.12, "table_data": 0.08},
    }
    score += _answer_type_boosts.get(expected_answer_type, {}).get(info_type, 0.0)

    return score


# ═══════════════════════════════════════════════════════════════════════════
# Graph traversal helpers
# ═══════════════════════════════════════════════════════════════════════════

_EDGE_WEIGHTS = {
    "NEXT_LINE":            0.95,
    "CONTINUES_ON_PAGE":    0.93,
    "SEMANTICALLY_RELATED": 0.85,
    "SEMANTIC_ALIGNMENT":   0.82,
    "TOPIC_BRIDGE":         0.80,
    "DEFINITION_USED_IN":   0.78,
    "EVIDENCE_CHAIN":       0.75,
    "SECTION_SUMMARY_OF":   0.70,
    "PARAGRAPH_SUMMARY_OF": 0.70,
    "TABLE_CONTEXT":        0.72,
    "IMAGE_CONTEXT":        0.68,
    "THEME_MEMBER":         0.65,
}


def _update(candidates, nid, score):
    if nid not in candidates or score > candidates[nid]:
        candidates[nid] = score


def _traverse_edges(G, node_id, base_score, candidates):
    """Walk 1-hop outgoing edges with type-specific decay."""
    for neighbor in G.successors(node_id):
        rel = G.edges[node_id, neighbor].get("relationship", "")
        weight = _EDGE_WEIGHTS.get(rel)
        if weight is None:
            continue
        # For SEMANTICALLY_RELATED, use stored similarity
        if rel == "SEMANTICALLY_RELATED":
            weight *= G.edges[node_id, neighbor].get("similarity", 0.5)
        n_score = base_score * weight
        _update(candidates, neighbor, n_score)


def _expand_via_evidence_summaries(G, candidates, query_text, key_entities=None):
    """
    If a section's evidence_summary node mentions concepts from the query
    or key entities, pull its aggregated claims' source lines into the pool.
    """
    query_words = set(query_text.lower().split())
    entity_words = set()
    for e in (key_entities or []):
        entity_words.update(e.lower().split())
    search_words = query_words | entity_words

    for n, d in G.nodes(data=True):
        if d.get("type") != "evidence_summary":
            continue
        digest = d.get("facts_digest", "").lower()
        digest_words = set(digest.split())

        # Keyword overlap
        word_overlap = len(digest_words & search_words)
        # Entity overlap (stronger signal)
        entity_overlap = len(digest_words & entity_words) if entity_words else 0

        if word_overlap < 2 and entity_overlap < 1:
            continue

        anchor_score = min(0.70, 0.12 * word_overlap + 0.15 * entity_overlap)
        for agg_target in G.successors(n):
            if G.edges[n, agg_target].get("relationship") != "AGGREGATES":
                continue
            claim_data = G.nodes.get(agg_target, {})
            for src in claim_data.get("source_lines", []):
                _update(candidates, src, anchor_score)


def _expand_via_section_concepts(G, candidates, key_entities, query_text):
    """
    Match query key entities against section-level key_concepts from
    semantic enrichment (module 9b). When a section's concepts overlap
    with the query, pull in the section's high-density lines.
    """
    if not key_entities:
        return

    entity_lower = {e.lower() for e in key_entities}
    query_words = set(query_text.lower().split())

    for n, d in G.nodes(data=True):
        if d.get("type") != "section":
            continue
        concepts = d.get("key_concepts", [])
        if not concepts:
            continue

        concept_lower = {c.lower() for c in concepts}
        # Check entity-concept overlap
        overlap = 0
        for entity in entity_lower:
            for concept in concept_lower:
                if entity in concept or concept in entity:
                    overlap += 1
        # Also check query word overlap with concepts
        for qw in query_words:
            if len(qw) > 3 and any(qw in c for c in concept_lower):
                overlap += 0.5

        if overlap < 1:
            continue

        section_score = min(0.55, 0.15 * overlap)
        # Pull in all text components belonging to this section (no density filter)
        for pred in G.predecessors(n):
            nd = G.nodes.get(pred, {})
            if nd.get("type") == "text_component":
                _update(candidates, pred, section_score)


def _expand_via_claims(G, candidates):
    """
    For any claim node already scoring, follow SUPPORTS / DEPENDS_ON /
    LINKED_TO edges to bring in related claims' source lines.
    """
    claim_hits = {
        nid: sc for nid, sc in list(candidates.items())
        if nid in G and G.nodes[nid].get("type") == "claim"
    }
    for cid, sc in claim_hits.items():
        for neighbor in G.successors(cid):
            rel = G.edges[cid, neighbor].get("relationship", "")
            if rel in ("SUPPORTS", "DEPENDS_ON", "LINKED_TO"):
                nd = G.nodes.get(neighbor, {})
                for src in nd.get("source_lines", []):
                    _update(candidates, src, sc * 0.65)


def _keyword_recall(G, candidates, query_text, key_entities):
    """Brute-force keyword scan of ALL text nodes as a recall safety net.

    Catches content the bi-encoder misses due to vocabulary mismatch.
    Enhanced with fuzzy prefix/stem matching for morphological variants
    (e.g. 'author' matches 'authors', 'authored', 'authoring').
    """
    _STOP = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'for',
        'and', 'or', 'what', 'who', 'which', 'where', 'when', 'this',
        'that', 'does', 'do', 'did', 'how', 'by', 'to', 'with', 'at',
        'on', 'it', 'be', 'not', 'from', 'as', 'but', 'has', 'have',
        'had', 'can', 'will', 'would', 'could', 'should', 'its', 'so',
    }
    q_words = {w.lower() for w in re.findall(r'\w+', query_text) if len(w) > 2}
    q_terms = q_words - _STOP
    entity_terms = set()
    for e in (key_entities or []):
        for w in re.findall(r'\w+', e.lower()):
            if len(w) > 2:
                entity_terms.add(w)
    search_terms = q_terms | entity_terms

    if not search_terms:
        return

    # Build prefix stems for fuzzy matching (min 4 chars)
    search_stems = set()
    for term in search_terms:
        if len(term) >= 4:
            search_stems.add(term[:4])
        if len(term) >= 5:
            search_stems.add(term[:5])

    for node, data in G.nodes(data=True):
        if data.get("type") != "text_component":
            continue
        text = data.get("text", "")
        if not text or len(text) < 15:
            continue
        text_lower = text.lower()
        text_words = set(re.findall(r'\w+', text_lower))

        # Exact matches
        exact_hits = search_terms & text_words
        entity_hits = entity_terms & text_words

        # Fuzzy prefix/stem matches (morphological variants)
        fuzzy_hits = 0
        if search_stems:
            text_stems = {w[:4] for w in text_words if len(w) >= 4}
            text_stems5 = {w[:5] for w in text_words if len(w) >= 5}
            fuzzy_hits = len(search_stems & (text_stems | text_stems5))

        # Combined match strength
        total_signal = len(exact_hits) + len(entity_hits) * 1.5 + fuzzy_hits * 0.4

        # Require meaningful overlap: ≥2 query terms or ≥1 entity match or ≥3 fuzzy
        if total_signal < 1.5 and not entity_hits:
            continue

        # Score proportional to term coverage
        score = 0.15 + 0.06 * len(exact_hits) + 0.10 * len(entity_hits) + 0.03 * fuzzy_hits
        score = min(score, 0.55)
        _update(candidates, node, score)


# ═══════════════════════════════════════════════════════════════════════════
# Semantic theme expansion (graph semantic alignment layer)
# ═══════════════════════════════════════════════════════════════════════════

def _expand_via_semantic_themes(G, candidates, query_text, key_entities):
    """
    Leverage semantic_theme nodes (created by the graph semantic alignment
    module) to find conceptually related content.  When the query matches
    a theme, pull ALL nodes belonging to that theme into the candidate pool.
    """
    query_lower = query_text.lower()
    q_words = set(re.findall(r'\w+', query_lower))
    entity_words = set()
    for e in (key_entities or []):
        entity_words.update(e.lower().split())
    search_words = q_words | entity_words

    for node, data in G.nodes(data=True):
        if data.get("type") != "semantic_theme":
            continue
        label = data.get("label", "").lower()
        description = data.get("description", "").lower()
        theme_words = set(re.findall(r'\w+', label + " " + description))

        # Check overlap between query and theme
        overlap = len(search_words & theme_words)
        entity_overlap = len(entity_words & theme_words) if entity_words else 0

        if overlap < 2 and entity_overlap < 1:
            continue

        theme_score = min(0.65, 0.10 * overlap + 0.15 * entity_overlap)

        # Pull in all member nodes of this theme
        for member in G.successors(node):
            edge = G.edges.get((node, member), {})
            if edge.get("relationship") == "THEME_MEMBER":
                _update(candidates, member, theme_score)
        # Also pull in nodes linked via SEMANTIC_ALIGNMENT
        for member in G.successors(node):
            edge = G.edges.get((node, member), {})
            if edge.get("relationship") == "SEMANTIC_ALIGNMENT":
                _update(candidates, member, theme_score * 0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Consolidation
# ═══════════════════════════════════════════════════════════════════════════

def _consolidate(G, candidates, structured_query, intent):
    """Score, expand context, and build the output list.

    Scoring: final_score = 0.7 * semantic_score + 0.3 * text_match_score
    - Paragraph-level nodes (longer text) strongly preferred
    - No filtering on density, information type, or node type
    - Deduplicated to avoid overlapping context windows
    """
    query_text = structured_query["original_text"]

    section_max = {}
    for nid, sc in candidates.items():
        if nid not in G:
            continue
        sid = G.nodes[nid].get("section_id")
        if sid:
            section_max[sid] = max(section_max.get(sid, 0.0), sc)

    # Identify top 3 sections to avoid over-filtering
    sorted_sections = sorted(section_max.items(), key=lambda x: x[1], reverse=True)
    top_sections = {s for s, _ in sorted_sections[:3]} if sorted_sections else set()

    results = []

    for node_id, semantic_score in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        if node_id not in G:
            continue
        nd = G.nodes[node_id]
        if nd.get("type") != "text_component":
            continue

        text = nd.get("text", "")

        # ── Structural / low-information noise penalty ───────────────
        stripped = text.strip()
        word_count = len(stripped.split())
        is_structural = False

        # Table-of-contents patterns: repeated dots, leader lines
        if re.search(r'\.{4,}', stripped) or re.search(r'-{4,}', stripped):
            is_structural = True

        # Page-number-only or number-dot patterns ("1.2.3", "12")
        if re.match(r'^[\d\.\s\-]+$', stripped):
            is_structural = True

        # Index-like lines: "Chapter 3 ..... 45", "Section 2.1 ... 12"
        if re.search(r'\.\s*\d+\s*$', stripped) and '...' in stripped:
            is_structural = True

        # All-caps short headings with no informational content
        if word_count <= 3 and stripped.isupper() and not any(c.isdigit() for c in stripped):
            is_structural = True

        # Combined scoring: semantic + text match
        text_match = _text_match_score(query_text, text)
        score = 0.7 * semantic_score + 0.3 * text_match

        # Apply structural penalty (mild: context expansion will recover content)
        if is_structural:
            score -= 0.12

        # Boost nodes with strong text match even if semantic score was weak
        if text_match > 0.5 and semantic_score < 0.4:
            score += 0.15

        # Paragraph preference: prioritize longer, complete text
        text_len = len(text)
        if text_len > 120:
            score += 0.06
        elif text_len > 60:
            score += 0.03

        # Bonus for meaningful sentence structure
        if word_count >= 8 and text_len > 60:
            score += 0.04

        sid = nd.get("section_id")

        # Softer section-dominance: only penalize if NOT a broad intent
        if intent not in ("summarization", "comparative", "enumerative", "causal") \
                and top_sections and sid not in top_sections:
            score *= 0.85

        ocr = nd.get("ocr_confidence", 1.0)
        score *= ocr ** 1.5

        # Context expansion (4 forward, 2 backward)
        parts = [text]
        cur = node_id
        for _ in range(4):
            fwd = [f for f in G.successors(cur)
                   if G.edges[cur, f].get("relationship") in ("NEXT_LINE", "CONTINUES_ON_PAGE")]
            if fwd:
                cur = fwd[0]
                parts.append(G.nodes[cur].get("text", ""))
            else:
                break
        cur = node_id
        for _ in range(2):
            bk = [p for p in G.predecessors(cur)
                  if G.edges[p, cur].get("relationship") in ("NEXT_LINE", "CONTINUES_ON_PAGE")]
            if bk:
                cur = bk[0]
                parts.insert(0, G.nodes[cur].get("text", ""))
            else:
                break

        section_label = "Unknown Section"
        if sid and sid in G:
            section_label = G.nodes[sid].get("label", section_label)

        results.append({
            "line_id": node_id,
            "text": " ".join(parts),
            "section_id": sid,
            "alignment_score": min(score, 1.0),
            "ocr_confidence": ocr,
            "metadata": {
                "page": nd.get("page"),
                "section": section_label,
                "information_type": nd.get("information_type", ""),
            },
        })

    # Deduplicate overlapping contexts before returning
    results = _deduplicate_results(results)

    return results


