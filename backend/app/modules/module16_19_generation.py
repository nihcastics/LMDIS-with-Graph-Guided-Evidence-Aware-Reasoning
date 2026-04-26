"""
Module 16-19: Evidence-Bound Answer Generation
================================================
Path A of the dual-path pipeline.

M16 — Controlled generation from evidence text via OpenRouter LLM
M17 — Adversarial validation (supported / unsupported check)
M18 — Token-level evidence attribution
M19 — Evidence coverage verification

Returns {"answer_text": str, "contributing_lines": list} consumed by
the Answer Selection Layer.
"""

from openai import OpenAI
import os
import re
from backend.app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_STRONG_MODEL,
    OPENROUTER_STRONG_FALLBACK_MODELS,
    GOOGLE_API_KEY,
    GOOGLE_STRONG_MODEL,
)

# Cached OpenAI client
_openai_client = None
_google_client = None
_VERBOSE_LOGS = os.getenv("LMDIS_VERBOSE_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(message):
    if _VERBOSE_LOGS:
        print(message)

def _get_client():
    global _openai_client
    if _openai_client is None and OPENROUTER_API_KEY:
        _openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _openai_client


def _get_google_client():
    global _google_client
    if _google_client is None and GOOGLE_API_KEY:
        _google_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GOOGLE_API_KEY,
        )
    return _google_client


# ═══════════════════════════════════════════════════════════════════════════
# Extractive answer helpers (for short factual queries)
# ═══════════════════════════════════════════════════════════════════════════

_FACTUAL_PATTERNS = re.compile(
    r'^(who|what|which|when|where|whom|whose)\b',
    re.I,
)
_FACTUAL_KEYWORDS = {
    'author', 'name', 'title', 'date', 'year', 'founder',
    'creator', 'publisher', 'editor', 'affiliation', 'institution',
    'location', 'city', 'country', 'number', 'version',
}


def _is_factual_query(query_text, intent, expected_type):
    """Return True if the query is a short factual question expecting a specific value."""
    q = query_text.strip()
    if len(q.split()) > 12:
        return False
    if _FACTUAL_PATTERNS.search(q):
        return True
    q_lower = q.lower()
    if any(kw in q_lower for kw in _FACTUAL_KEYWORDS):
        return True
    if expected_type in ('name', 'date', 'number', 'percentage', 'entity'):
        return True
    if intent in ('factual_lookup', 'definition'):
        if len(q.split()) <= 8:
            return True
    return False


# Concept synonym map: query phrases → canonical label concepts they should match.
# Each key is a frozenset of trigger words; value is a set of label words that
# should be treated as matching.  Kept small & deterministic — no model needed.
_CONCEPT_SYNONYMS = {
    frozenset({'author'}):      {'author', 'authors', 'written', 'writer', 'writers', 'by'},
    frozenset({'wrote'}):       {'author', 'authors', 'written', 'writer', 'by'},
    frozenset({'written'}):     {'author', 'authors', 'written', 'writer', 'by'},
    frozenset({'writer'}):      {'author', 'authors', 'writer', 'writers', 'by'},
    frozenset({'title'}):       {'title', 'name', 'heading', 'subject'},
    frozenset({'name'}):        {'name', 'title', 'called', 'named', 'label'},
    frozenset({'date'}):        {'date', 'published', 'year', 'issued', 'received', 'accepted', 'submitted'},
    frozenset({'year'}):        {'date', 'year', 'published', 'issued'},
    frozenset({'published'}):   {'date', 'published', 'publication', 'year', 'issued'},
    frozenset({'publisher'}):   {'publisher', 'published', 'publication', 'press', 'journal'},
    frozenset({'journal'}):     {'journal', 'publisher', 'publication', 'published'},
    frozenset({'editor'}):      {'editor', 'editors', 'edited'},
    frozenset({'affiliation'}): {'affiliation', 'affiliations', 'institution', 'university', 'department', 'school', 'college', 'organization'},
    frozenset({'institution'}): {'institution', 'university', 'affiliation', 'department', 'school', 'college', 'organization'},
    frozenset({'university'}):  {'university', 'institution', 'affiliation', 'school', 'college'},
    frozenset({'location'}):    {'location', 'city', 'country', 'address', 'place'},
    frozenset({'city'}):        {'city', 'location', 'place', 'address'},
    frozenset({'country'}):     {'country', 'location', 'nation'},
    frozenset({'email'}):       {'email', 'mail', 'contact', 'e-mail'},
    frozenset({'doi'}):         {'doi', 'identifier', 'id'},
    frozenset({'abstract'}):    {'abstract', 'summary', 'overview'},
    frozenset({'keyword'}):     {'keyword', 'keywords', 'tags', 'terms'},
    frozenset({'version'}):     {'version', 'release', 'revision'},
    frozenset({'founder'}):     {'founder', 'founded', 'creator', 'established'},
    frozenset({'creator'}):     {'creator', 'created', 'founder', 'author'},
}


def _expand_hint_words(hint_words):
    """Expand raw hint words via the synonym map to get all possible label concepts."""
    expanded = set(hint_words)
    for trigger, synonyms in _CONCEPT_SYNONYMS.items():
        if trigger & hint_words:
            expanded |= synonyms
    return expanded


def _soft_label_match(label_words, expanded_hints):
    """Score how well a label matches expanded hint concepts.

    Returns a float ≥ 0.  Accounts for:
      - exact word overlap
      - prefix/substring overlap (≥4 chars) for morphological variants
    """
    score = 0.0
    for lw in label_words:
        if lw in expanded_hints:
            score += 1.0
            continue
        # Prefix / substring match (e.g. "authors" ↔ "author")
        for hw in expanded_hints:
            if len(lw) >= 4 and len(hw) >= 4:
                if lw.startswith(hw[:4]) or hw.startswith(lw[:4]):
                    score += 0.7
                    break
    return score


def _try_extractive_answer(query_text, aligned_evidence):
    """
    Attempt to extract a short, exact answer span from the evidence.

    Strategies (tried in order for each line):
    1. Soft label matching — "Label: Value" patterns where the label
       matches query concepts via synonym expansion + prefix overlap.
    2. High-value span matching — short lines rich in proper nouns,
       numbers, or capitalized phrases that overlap query topic words.
    3. Short-line topic overlap — fallback for lines containing query
       topic words.
    """
    q_lower = query_text.lower()
    q_words = set(re.findall(r'\w+', q_lower))
    _STOP = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'for',
        'and', 'or', 'what', 'who', 'which', 'where', 'when', 'this',
        'that', 'does', 'do', 'did', 'how', 'by', 'to', 'with', 'at',
    }
    hint_words = {w for w in q_words if w not in _STOP and len(w) > 2}
    expanded_hints = _expand_hint_words(hint_words)

    best_span = None
    best_score = 0.0
    best_line_id = None

    for ev in aligned_evidence:
        ev_text = ev.get('text', '')
        ev_score = ev.get('alignment_score', 0)
        line_id = ev.get('line_id')

        for line in ev_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Strategy 1: Soft label matching on "Label: Value" patterns
            m = re.match(r'^([A-Za-z][A-Za-z\s]{0,30})\s*[:\-\u2013\u2014]\s*(.+)', line)
            if m:
                label_part = m.group(1).strip().lower()
                value_part = m.group(2).strip()
                label_words = set(label_part.split())
                label_match = _soft_label_match(label_words, expanded_hints)
                if label_match >= 0.7 and len(value_part) > 1:
                    score = ev_score + 0.3 + 0.12 * label_match
                    if score > best_score:
                        best_score = score
                        best_span = value_part
                        best_line_id = line_id
                    continue

            # Strategy 2: High-value span — short line with proper nouns / numbers
            line_lower = line.lower()
            line_words_set = set(re.findall(r'\w+', line_lower))
            proper_nouns = set(re.findall(r'\b[A-Z][a-z]{2,}\b', line))
            numbers = set(re.findall(r'\b\d[\d.,]*\b', line))
            high_value_count = len(proper_nouns) + len(numbers)

            if high_value_count >= 1 and len(line.split()) <= 20:
                overlap = expanded_hints & line_words_set
                if overlap:
                    score = ev_score + 0.06 * len(overlap) + 0.04 * high_value_count
                    if score > best_score:
                        best_score = score
                        best_span = line
                        best_line_id = line_id
                    continue

            # Strategy 3: Short line topic overlap
            overlap = hint_words & line_words_set
            if overlap and len(line.split()) <= 25:
                score = ev_score + 0.05 * len(overlap)
                if score > best_score:
                    best_score = score
                    best_span = line
                    best_line_id = line_id

    if best_span and best_score > 0.3:
        best_span = best_span.strip().rstrip('.').strip()
        return {
            'answer_text': best_span,
            'contributing_lines': [best_line_id] if best_line_id else [],
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer(structured_query, aligned_evidence, doc_meta=None):
    """
    Evidence-graph generation path (Path A).
    Returns {"answer_text": str, "contributing_lines": list[str]} or None.
    """
    if not aligned_evidence:
        return None

    query_text = structured_query["original_text"]
    intent = structured_query.get("intent", "factual_lookup")
    expected_type = structured_query.get("expected_answer_type", "description")

    # ── Fast extractive path for short factual queries ──
    if _is_factual_query(query_text, intent, expected_type):
        extractive = _try_extractive_answer(query_text, aligned_evidence)
        if extractive:
            return extractive

    # ── Build evidence text for the LLM prompt, ranked by score ──
    seen_texts = set()
    evidence_parts = []
    for ev in sorted(aligned_evidence, key=lambda x: x.get("alignment_score", 0), reverse=True):
        meta = ev.get("metadata", {})
        text = ev["text"].strip()
        # Deduplicate near-identical evidence
        text_key = text[:100].lower()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        info_type = meta.get("information_type", "")
        type_tag = f" ({info_type})" if info_type else ""
        evidence_parts.append(
            f"[Page {meta.get('page', '?')}, {meta.get('section', 'Unknown')}{type_tag}] "
            f"(relevance: {ev.get('alignment_score', 0):.2f}) {text}"
        )
        if len(evidence_parts) >= 20:
            break
    evidence_text = "\n".join(evidence_parts)

    # ── Try LLM generation with enriched context ──
    ai_answer = _generate_with_openrouter(
        query_text, evidence_text, doc_meta, intent, expected_type,
        evidence_count=len(evidence_parts),
    )

    if ai_answer:
        citations = _select_citations(ai_answer, aligned_evidence)
        return {
            "answer_text": ai_answer,
            "contributing_lines": citations,
        }

    # ── Deterministic fallback ──
    return _deterministic_fallback(structured_query, aligned_evidence, doc_meta)


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive token budget
# ═══════════════════════════════════════════════════════════════════════════

def _compute_max_tokens(query_text: str, intent: str, expected_type: str, evidence_count: int = 0) -> int:
    """Return an adaptive max_tokens budget based on query complexity."""
    _INTENT_BUDGET = {
        "summarization":   3200,
        "comparative":     2400,
        "procedural":      2400,
        "enumerative":     2000,
        "causal":          1800,
        "quantitative":    1600,
        "definition":      1400,
        "factual_lookup":  1024,
    }
    base = _INTENT_BUDGET.get(intent, 1400)

    # Longer queries tend to expect more elaborate answers
    words = len(query_text.split())
    if words > 30:
        base = int(base * 1.4)
    elif words > 15:
        base = int(base * 1.2)

    # More evidence chunks → richer answer possible
    if evidence_count >= 15:
        base = int(base * 1.25)
    elif evidence_count >= 8:
        base = int(base * 1.1)

    # Structured / long-form expected types get extra headroom
    if expected_type in ("list", "explanation", "summary"):
        base = max(base, 1800)

    return min(base, 4096)


# ═══════════════════════════════════════════════════════════════════════════
# LLM call
# ═══════════════════════════════════════════════════════════════════════════

def _generate_with_openrouter(query, evidence_text, doc_meta, intent="factual_lookup", expected_type="description", evidence_count=0):
    if not OPENROUTER_API_KEY:
        return None

    title = (doc_meta or {}).get("detected_title", "Unknown")
    filename = (doc_meta or {}).get("filename", "Unknown")

    # Intent-specific instructions for the LLM
    _intent_instructions = {
        "factual_lookup": "Provide a precise, direct answer to the factual question.",
        "comparative": "Clearly compare the items, highlighting similarities and differences.",
        "summarization": "Provide a comprehensive summary covering all key points from the evidence.",
        "quantitative": "Focus on providing the exact numbers, statistics, or quantities asked for.",
        "definition": "Provide a clear, precise definition based on the evidence.",
        "causal": "Explain the cause-and-effect relationship clearly.",
        "procedural": "Describe the process or method step by step.",
        "enumerative": "List all relevant items found in the evidence.",
    }
    intent_instruction = _intent_instructions.get(intent, _intent_instructions["factual_lookup"])

    system = (
        "You are an expert Document Intelligence Assistant. Provide "
        "accurate, comprehensive answers based on the evidence provided. "
        f"{intent_instruction} "
        "Structure with logical sections and bold key terms. "
        "Pay careful attention to which evidence passages have the highest relevance scores. "
        "Use ALL available evidence to construct the most complete answer possible. "
        "Include specific details: numbers, names, dates, and exact values from the evidence. "
        "Do NOT show reasoning, chain-of-thought, or <think> tags."
    )
    prompt = (
        f"### CONTEXT:\nDocument: {filename}\n"
        f"{f'Title: {title}' if title != 'Unknown' else ''}\n"
        f"Query intent: {intent} | Expected answer format: {expected_type}\n\n"
        f"### EVIDENCE (sorted by relevance):\n{evidence_text}\n\n"
        f"### QUESTION:\n{query}\n\n### ANSWER:"
    )

    max_tokens = _compute_max_tokens(query, intent, expected_type, evidence_count)
    _debug_log(f"[M16] Adaptive max_tokens={max_tokens} (intent={intent}, ev={evidence_count}, words={len(query.split())})")

    # Try Google Gemini 2.5 Pro directly first (if API key available)
    if GOOGLE_API_KEY:
        try:
            gclient = _get_google_client()
            if gclient:
                resp = gclient.chat.completions.create(
                    model=GOOGLE_STRONG_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.15,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content.strip()
                content = _clean_cot(content)
                if content:
                    _debug_log(f"[M16] Answer via Google {GOOGLE_STRONG_MODEL}")
                    return content
        except Exception as e:
            _debug_log(f"[M16] Google error, falling back to OpenRouter: {e}")

    models = [OPENROUTER_STRONG_MODEL] + OPENROUTER_STRONG_FALLBACK_MODELS + [OPENROUTER_MODEL] + OPENROUTER_FALLBACK_MODELS
    for model in models:
        try:
            client = _get_client()
            if client is None:
                return None
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.15,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Document Intelligence System",
                },
            )
            content = resp.choices[0].message.content.strip()
            content = _clean_cot(content)
            return content or None
        except Exception as e:
            err = str(e)
            if any(c in err for c in ("429", "404", "400", "401", "402", "503", "502")):
                continue
            _debug_log(f"[M16] Critical error with {model}: {e}")
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_COT_RE = [
    (re.compile(r"<(think|thought|reasoning)>.*?</\1>", re.DOTALL | re.I), ""),
    (re.compile(r"<(think|thought|reasoning)>.*", re.DOTALL | re.I), ""),
    (re.compile(r"^###?\s*(Reasoning|Thought|Internal thought):?\s*", re.I | re.M), ""),
]


def _clean_cot(text):
    for pat, repl in _COT_RE:
        text = pat.sub(repl, text)
    return text.strip()


def _select_citations(answer_text, aligned_evidence):
    """
    Select evidence items that genuinely contributed to the answer.

    Uses a multi-signal approach:
      1. Unigram + bigram overlap (excludes stop words)
      2. Named-entity / number overlap (high-value tokens)
      3. Alignment score from the retrieval pipeline as a prior
    Returns line_ids sorted by combined attribution score.
    """
    stop = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "with", "which", "that", "this", "for",
        "it", "be", "has", "have", "had", "not", "as", "at", "by", "from",
        "can", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "its", "their", "they", "we", "he", "she", "been",
        "being", "about", "into", "through", "during", "before", "after",
        "between", "each", "all", "both", "such", "than", "also", "so",
        "only", "very", "just", "where", "when", "how", "what", "there",
        "then", "these", "those", "some", "any", "more", "most", "other",
    }

    def _tokens(text):
        return [w for w in re.findall(r"\w+", text.lower()) if w not in stop and len(w) > 1]

    def _bigrams(tokens):
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    def _high_value_tokens(text):
        """Numbers, percentages, capitalized proper nouns, abbreviations."""
        nums = set(re.findall(r"\d+\.?\d*%?", text))
        caps = {w for w in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text)}
        abbrs = {w for w in re.findall(r"\b[A-Z]{2,}\b", text)}
        return nums | caps | abbrs

    answer_tokens = _tokens(answer_text)
    answer_bigrams = _bigrams(answer_tokens)
    answer_unigrams = set(answer_tokens)
    answer_hv = _high_value_tokens(answer_text)

    scored = []
    for ev in aligned_evidence:
        ev_text = ev.get("text", "")
        ev_tok = _tokens(ev_text)
        ev_uni = set(ev_tok)
        ev_bi = _bigrams(ev_tok)
        ev_hv = _high_value_tokens(ev_text)

        # Unigram overlap ratio
        uni_overlap = len(answer_unigrams & ev_uni)
        uni_score = uni_overlap / max(len(answer_unigrams), 1)

        # Bigram overlap (stronger signal of phrase-level attribution)
        bi_overlap = len(answer_bigrams & ev_bi)
        bi_score = bi_overlap / max(len(answer_bigrams), 1)

        # High-value token overlap (numbers, proper nouns — very strong signal)
        hv_overlap = len(answer_hv & ev_hv) if answer_hv else 0
        hv_score = hv_overlap / max(len(answer_hv), 1)

        # Retrieval alignment score as prior
        align_prior = ev.get("alignment_score", 0.0)

        # Combined attribution score
        attr_score = (
            0.25 * uni_score
            + 0.30 * bi_score
            + 0.25 * hv_score
            + 0.20 * align_prior
        )

        if attr_score > 0.05 or uni_overlap >= 3:
            scored.append((ev["line_id"], attr_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    hits = [lid for lid, _ in scored]
    return hits or [ev["line_id"] for ev in aligned_evidence[:3]]


def _deterministic_fallback(structured_query, aligned_evidence, doc_meta):
    """Keyword-based sentence selection when the LLM is unreachable."""
    query_kw = set(structured_query["original_text"].lower().split())
    scored = []
    for item in aligned_evidence:
        text = item.get("text", "")
        for sent in text.split("."):
            sent = sent.strip()
            if not sent:
                continue
            overlap = len(set(sent.lower().split()) & query_kw)
            if overlap:
                scored.append((sent, overlap, item["line_id"]))
    scored.sort(key=lambda x: x[1], reverse=True)

    chosen, seen, cites = [], set(), []
    for s, _, lid in scored[:3]:
        if s not in seen:
            chosen.append(s)
            cites.append(lid)
            seen.add(s)

    if chosen:
        text = ". ".join(chosen)
        if text and text[-1] not in ".!?":
            text += "."
        return {"answer_text": text, "contributing_lines": cites}

    # Last resort: excerpt top evidence
    top = aligned_evidence[:5]
    title = (doc_meta or {}).get("detected_title", "Document")
    parts = []
    for i, ev in enumerate(top):
        sec = ev.get("metadata", {}).get("section", "Detail")
        parts.append(f"{i+1}. **{sec}**: {ev['text'][:180].strip()}...")
    text = (
        f"### [Local Reliability Mode]\n\n"
        f"**Document:** {title}\n\n"
        f"**Extracted Highlights:**\n" + "\n".join(parts) + "\n\n"
        "*(AI synthesis unavailable; showing direct excerpts.)*"
    )
    return {"answer_text": text, "contributing_lines": [e["line_id"] for e in top]}
