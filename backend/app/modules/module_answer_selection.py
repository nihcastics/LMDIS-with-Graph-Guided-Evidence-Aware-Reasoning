"""
Module ASL: Answer Selection & Compensation Layer
=====================================================
Dual-path answer arbitration:

Path A — Evidence-graph answer  (structured retrieval → LLM)
Path B — Direct-LLM answer      (stored knowledge → LLM)

The layer:
1. Extracts atomic facts from **both** answers.
2. Scores each answer on coverage, specificity, and grounding.
3. Merges the best facts from both into a unified fact set.
4. Sends the merged facts + original question to the conversational
   LLM for final, polished answer framing.
"""

import os
import re
import requests
from openai import OpenAI
from backend.app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_STRONG_MODEL,
    OPENROUTER_STRONG_FALLBACK_MODELS,
    GOOGLE_API_KEY,
    GOOGLE_STRONG_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_CRISP_MODEL,
    ANTHROPIC_BEST_MODEL,
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


def _strong_models():
    """Return the ordered model list starting with the strongest available."""
    return [OPENROUTER_STRONG_MODEL] + OPENROUTER_STRONG_FALLBACK_MODELS


def _compute_max_tokens(query_text: str, facts_or_evidence: list) -> int:
    """Return an adaptive max_tokens budget based on query length and input size."""
    words = len(query_text.split())
    item_count = len(facts_or_evidence)

    # Base budget grows with the number of evidence/fact items
    if item_count >= 20:
        base = 3200
    elif item_count >= 12:
        base = 2400
    elif item_count >= 6:
        base = 1800
    else:
        base = 1400

    # Longer queries typically expect more elaborate synthesis
    if words > 30:
        base = int(base * 1.35)
    elif words > 15:
        base = int(base * 1.15)

    return min(base, 4096)


def select_and_merge(
    query: str,
    evidence_answer: dict | None,
    llm_direct: dict | None,
    doc_meta: dict | None = None,
    aligned_evidence: list | None = None,
    graph=None,
):
    """
    Main entry-point.  Accepts:
      * evidence_answer: {"answer_text": str, "contributing_lines": [...]}
                         (output of module16_19_generation.generate_answer)
      * llm_direct:      {"answer": str, "facts": [str]}
                         (output of module_llm_knowledge.query_llm_directly)

    Returns a dict compatible with module20_formatter:
        {"answer_text": str, "contributing_lines": [...], "source_path": str,
         "dual_path_comparison": {...}}
    """
    ev_text = (evidence_answer or {}).get("answer_text", "")
    ev_lines = (evidence_answer or {}).get("contributing_lines", [])

    llm_text = (llm_direct or {}).get("answer", "")
    llm_facts = (llm_direct or {}).get("facts", [])

    # ── 1. Extract facts from both paths ──────────────────────────────────
    ev_facts = _extract_facts_from_text(ev_text)
    # llm_facts already extracted by the knowledge module; supplement
    llm_facts = llm_facts or _extract_facts_from_text(llm_text)

    # ── 2. Score both answers ─────────────────────────────────────────────
    ev_score = _score_answer(ev_text, ev_facts, query)
    llm_score = _score_answer(llm_text, llm_facts, query)

    # Detect refusal-like answers (LLM said "no info" despite evidence existing)
    _is_ev_refusal = ev_score <= 0.10
    _is_llm_refusal = llm_score <= 0.10

    # Detect metadata / document-bibliographic queries & explicit LLM "no doc evidence" claims
    _is_metadata = _is_metadata_query(query)
    _is_authorship = _is_authorship_query(query)
    _llm_no_doc = _llm_claims_no_document_evidence(llm_text)  # explicit "not in document" phrasing

    # ── Build dual-path comparison (always present) ───────────────────────
    dual_path = {
        "path_a": {
            "name": "Evidence Graph (Path A)",
            "answer_text": ev_text or "(No answer produced)",
            "facts": ev_facts,
            "score": ev_score,
            "status": "refused" if _is_ev_refusal else ("empty" if not ev_text else "ok"),
        },
        "path_b": {
            "name": "LLM Knowledge (Path B)",
            "answer_text": llm_text or "(No answer produced)",
            "facts": llm_facts,
            "score": llm_score,
            "status": "refused" if _is_llm_refusal else ("empty" if not llm_text else "ok"),
        },
        "merged_facts": [],
        "decision_reason": "",
        "winner": "",
    }

    # ── 3. Decision logic ─────────────────────────────────────────────────
    title = (doc_meta or {}).get("detected_title", "Document")

    def _make_result(answer_text, contributing_lines, source_path, reason, winner, crisp=None):
        dual_path["decision_reason"] = reason
        dual_path["winner"] = winner
        dual_path["merged_facts"] = _merge_facts(ev_facts, llm_facts) if (ev_facts or llm_facts) else []
        return {
            "answer_text": answer_text,
            "contributing_lines": contributing_lines,
            "source_path": source_path,
            "dual_path_comparison": dual_path,
            "claude_crisp_answer": crisp,
        }

    # Authorship-style fallback: when evidence retrieval is weak, inspect document text
    # directly with Claude's strongest model to recover names safely.
    if _is_authorship and (not aligned_evidence or _is_ev_refusal or ev_score <= 0.10):
        failsafe = _claude_authorship_failsafe(query, graph, doc_meta)
        if failsafe:
            return _make_result(
                failsafe["answer_text"],
                failsafe.get("contributing_lines", []),
                "document_failsafe",
                "Authorship query had weak graph evidence; document-level failsafe synthesis applied.",
                "document_failsafe",
                None,
            )

    # ── Metadata query priority: graph evidence always wins over LLM ──────
    # For author / title / date / affiliation / doc-detail questions the
    # evidence graph is the ground truth. The LLM may hallucinate or
    # correctly report it has no knowledge — in both cases the graph wins.
    if _is_metadata:
        _debug_log("[ASL] Metadata query detected: prioritising evidence graph over LLM.")

        # 1. Evidence graph has content → use it (possibly via deep analysis for precision)
        if ev_score > 0 and not _is_ev_refusal:
            if aligned_evidence:
                deep = _deep_analysis_answer(query, aligned_evidence, title)
                if deep:
                    crisp = _claude_crisp_answer(query, deep, title)
                    return _make_result(deep, ev_lines, "deep_analysis",
                        f"Metadata query: evidence graph prioritised (ev={ev_score:.2f}, llm={llm_score:.2f}) — deep analysis for maximum precision.",
                        "deep_analysis", crisp)
            crisp = _claude_crisp_answer(query, ev_text, title)
            return _make_result(ev_text, ev_lines, "evidence_graph",
                f"Metadata query: evidence graph used directly (ev={ev_score:.2f}) — LLM score ({llm_score:.2f}) ignored.",
                "path_a", crisp)

        # 2. LLM explicitly says info isn't in the document OR ev path empty/refused
        #    → deep-analyse raw evidence as the definitive final check
        if aligned_evidence:
            deep = _deep_analysis_answer(query, aligned_evidence, title)
            if deep:
                crisp = _claude_crisp_answer(query, deep, title)
                reason = (
                    "Metadata query: LLM claimed no document evidence — evidence graph examined as final authority."
                    if _llm_no_doc else
                    "Metadata query: evidence path empty/refused — deep analysis of raw evidence as final check."
                )
                return _make_result(deep, ev_lines, "deep_analysis", reason, "deep_analysis", crisp)

        # 3. Raw evidence exhausted falling through: prefer LLM only as last resort
        _debug_log("[ASL] Metadata query: no usable graph evidence found; falling through to standard path.")
        # (falls through to standard decision logic below)

    # ── LLM explicitly denies document evidence → force evidence graph check ──
    # Applied to all query types: when the LLM says "not found in document"
    # and the standard evidence score is zero, try deep analysis before
    # accepting that the answer truly doesn't exist.
    elif _llm_no_doc and (ev_score == 0 or _is_ev_refusal) and aligned_evidence:
        _debug_log("[ASL] LLM claims no document evidence; running final evidence graph check.")
        deep = _deep_analysis_answer(query, aligned_evidence, title)
        if deep:
            crisp = _claude_crisp_answer(query, deep, title)
            return _make_result(deep, ev_lines, "deep_analysis",
                "LLM reported no document evidence — evidence graph examined as final check and recovered an answer.",
                "deep_analysis", crisp)
        # Deep analysis also empty: fall through to standard path which may still
        # surface something or produce a clean "not found" message.

    # If both paths failed or produced refusals, go straight to deep analysis
    if (ev_score == 0 and llm_score == 0) or (_is_ev_refusal and _is_llm_refusal):
        if aligned_evidence:
            deep = _deep_analysis_answer(query, aligned_evidence, title)
            if deep:
                crisp = _claude_crisp_answer(query, deep, title)
                return _make_result(deep, ev_lines, "deep_analysis",
                    "Both paths returned refusals — deep analysis used raw evidence to produce a comprehensive answer.",
                    "deep_analysis", crisp)
        if llm_score > ev_score and llm_text:
            crisp = _claude_crisp_answer(query, llm_text, title)
            return _make_result(llm_text, [], "llm_direct",
                "Both paths weak — LLM path had slightly higher score.",
                "path_b", crisp)
        if ev_text and ev_score > 0:
            crisp = _claude_crisp_answer(query, ev_text, title)
            return _make_result(ev_text, ev_lines, "evidence_graph",
                "Both paths weak — evidence path had slightly higher score.",
                "path_a", crisp)
        return _make_result(
            "No information found in the document.", [], "none",
            "Neither path produced usable content.", "none")

    # If evidence path is a refusal but LLM path has content
    if _is_ev_refusal and not _is_llm_refusal:
        if aligned_evidence:
            deep = _deep_analysis_answer(query, aligned_evidence, title)
            if deep:
                crisp = _claude_crisp_answer(query, deep, title)
                return _make_result(deep, ev_lines, "deep_analysis",
                    f"Evidence path refused (score {ev_score:.2f}) but raw evidence existed — deep analysis recovered a strong answer.",
                    "deep_analysis", crisp)
        crisp = _claude_crisp_answer(query, llm_text, title)
        return _make_result(llm_text, [], "llm_direct",
            f"Evidence path refused (score {ev_score:.2f}), LLM path produced content (score {llm_score:.2f}).",
            "path_b", crisp)

    if ev_score == 0:
        # If the LLM also explicitly says it found nothing in the document,
        # make one final attempt using raw evidence before accepting defeat.
        if _llm_no_doc and aligned_evidence:
            deep = _deep_analysis_answer(query, aligned_evidence, title)
            if deep:
                crisp = _claude_crisp_answer(query, deep, title)
                return _make_result(deep, ev_lines, "deep_analysis",
                    f"Evidence path empty and LLM found no document evidence — deep analysis of raw evidence used as final fallback.",
                    "deep_analysis", crisp)
        crisp = _claude_crisp_answer(query, llm_text, title)
        return _make_result(llm_text, [], "llm_direct",
            f"Evidence path empty — LLM path used (score {llm_score:.2f}).",
            "path_b", crisp)

    if llm_score == 0:
        crisp = _claude_crisp_answer(query, ev_text, title)
        return _make_result(ev_text, ev_lines, "evidence_graph",
            f"LLM path empty — evidence path used (score {ev_score:.2f}).",
            "path_a", crisp)

    # Both available with real content — decide approach
    # If evidence clearly dominates, use it directly
    if ev_score >= 0.55 and ev_score > llm_score * 1.4:
        crisp = _claude_crisp_answer(query, ev_text, title)
        return _make_result(ev_text, ev_lines, "evidence_graph",
            f"Evidence path clearly dominates: {ev_score:.2f} vs {llm_score:.2f} (>{1.4:.1f}x). Used directly.",
            "path_a", crisp)

    # ── Deep Analysis: use raw evidence for comprehensive answer ──
    if aligned_evidence:
        deep = _deep_analysis_answer(query, aligned_evidence, title)
        if deep:
            crisp = _claude_crisp_answer(query, deep, title)
            return _make_result(deep, ev_lines, "deep_analysis",
                f"Both paths viable (A={ev_score:.2f}, B={llm_score:.2f}) — deep analysis produced strongest answer from raw evidence.",
                "deep_analysis", crisp)

    # Fallback: traditional fact-merge + reframe
    merged_facts = _merge_facts(ev_facts, llm_facts)
    dual_path["merged_facts"] = merged_facts
    final_text = _reframe_answer(query, merged_facts, title)

    if final_text:
        crisp = _claude_crisp_answer(query, final_text, title)
        return _make_result(final_text, ev_lines, "merged",
            f"Both paths merged: extracted {len(ev_facts)} evidence facts + {len(llm_facts)} LLM facts → {len(merged_facts)} unique facts reframed.",
            "merged", crisp)

    # Fallback: pick the higher-scoring raw answer
    if ev_score >= llm_score:
        crisp = _claude_crisp_answer(query, ev_text, title)
        return _make_result(ev_text, ev_lines, "evidence_graph",
            f"Merge failed — used higher-scoring evidence path ({ev_score:.2f} vs {llm_score:.2f}).",
            "path_a", crisp)
    else:
        crisp = _claude_crisp_answer(query, llm_text, title)
        return _make_result(llm_text, [], "llm_direct",
            f"Merge failed — used higher-scoring LLM path ({llm_score:.2f} vs {ev_score:.2f}).",
            "path_b", crisp)


# ═══════════════════════════════════════════════════════════════════════════
# Fact extraction & scoring
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Claude crisp-answer: one precise sentence via Anthropic API (budget-safe)
# ═══════════════════════════════════════════════════════════════════════════

def _claude_crisp_answer(query: str, elaborated_text: str, title: str) -> str | None:
    """
    Send the already-generated elaborate answer (from Gemini/OpenRouter) to
    Claude with a single instruction: distil it into ONE crisp, 100% accurate
    sentence.  Max ~60 output tokens → cost ≈ $0.00024 per call.
    """
    if not ANTHROPIC_API_KEY or not elaborated_text:
        return None

    # Truncate to keep input tokens minimal (cost control)
    snippet = elaborated_text[:1500].strip()

    prompt = (
        f"Document: {title}\n\n"
        f"Detailed answer available:\n{snippet}\n\n"
        f"Question: {query}\n\n"
        "Distil the answer above into ONE single crisp sentence that is "
        "100% factually accurate and directly answers the question. "
        "No preamble, no filler, no markdown — just the sentence."
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_CRISP_MODEL,
                "max_tokens": 80,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            text = data["content"][0]["text"].strip()
            _debug_log(f"[ASL] Claude crisp answer ({ANTHROPIC_CRISP_MODEL}): {text[:80]}")
            return text or None
        else:
            _debug_log(f"[ASL] Claude API error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        _debug_log(f"[ASL] Claude crisp answer failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Metadata query detection & LLM "no document evidence" detection
# ═══════════════════════════════════════════════════════════════════════════

_METADATA_KEYWORDS = {
    'author', 'authors', 'writer', 'written', 'wrote',
    'student', 'students', 'team', 'member', 'members', 'developer', 'developers',
    'made', 'built', 'created', 'developed',
    'title', 'name', 'heading',
    'date', 'year', 'published', 'publication', 'issued', 'submitted', 'received', 'accepted',
    'publisher', 'journal', 'venue', 'conference',
    'editor', 'editors',
    'affiliation', 'institution', 'university', 'department', 'school', 'college', 'organization',
    'email', 'contact', 'doi', 'isbn', 'issn', 'identifier',
    'abstract', 'keywords', 'subject',
    'version', 'revision', 'edition',
    'creator', 'contributor', 'owner',
    'document', 'paper', 'report', 'manuscript', 'article',
    'metadata', 'details', 'information about',
}

_AUTHORSHIP_KEYWORDS = {
    'author', 'authors', 'student', 'students', 'team', 'developer', 'developers',
    'creator', 'creators', 'contributor', 'contributors', 'member', 'members',
}

_AUTHORSHIP_PHRASES = (
    'who is the author',
    'who are the authors',
    'who made this project',
    'who built this project',
    'who created this project',
    'students who made this project',
    'student who made this project',
    'project team',
    'team members',
    'developed by',
    'built by',
    'created by',
)

_NO_DOC_EVIDENCE_PHRASES = (
    "not found in the document",
    "no information found in the document",
    "does not contain",
    "not mentioned in the document",
    "no relevant information",
    "not available in the document",
    "is not provided",
    "unable to find",
    "could not find",
    "not present in the document",
    "no evidence found",
    "the document does not",
    "not discussed in the document",
    "cannot find",
    "couldn't find",
    "the provided evidence does not",
    "no information available",
    "the document doesn't contain",
    "based on the document, there is no",
    "insufficient document evidence",
)


def _is_metadata_query(query: str) -> bool:
    """Return True if the query is asking for document metadata / bibliographic details."""
    q_lower = query.lower()
    q_words = set(re.findall(r'\w+', q_lower))
    # Direct keyword match
    if q_words & _METADATA_KEYWORDS:
        return True
    # Phrase patterns
    meta_phrases = (
        'who wrote', 'who is the author', 'who are the authors',
        'who made this project', 'who built this project', 'who created this project',
        'students who made this project', 'student contributors', 'project team',
        'what is the title', 'when was this', 'when was it',
        'where was this published', 'what journal', 'which journal',
        'document details', 'document information', 'document metadata',
        'paper details', 'what year', 'publication year',
    )
    return any(p in q_lower for p in meta_phrases)


def _is_authorship_query(query: str) -> bool:
    """Return True for author/student/team ownership questions."""
    q_lower = query.lower()
    q_words = set(re.findall(r'\w+', q_lower))
    if q_words & _AUTHORSHIP_KEYWORDS:
        return True
    if ({"who", "name", "names"} & q_words) and ({"made", "built", "created", "developed", "wrote", "written"} & q_words):
        return True
    return any(phrase in q_lower for phrase in _AUTHORSHIP_PHRASES)


def _collect_authorship_context(graph, max_chars: int = 9000):
    """Collect likely authorship snippets from early-page metadata text nodes."""
    if graph is None:
        return "", []

    ranked = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") not in ("text_component", "paragraph"):
            continue

        text = (data.get("text") or "").strip()
        if len(text) < 3:
            continue

        page = data.get("page", 999)
        try:
            page_num = int(page)
        except Exception:
            page_num = 999

        info_type = (data.get("information_type") or "").lower()
        text_lower = text.lower()

        score = 0
        if page_num <= 2:
            score += 4
        if info_type in ("metadata", "fact", "header"):
            score += 3
        if any(term in text_lower for term in (
            "author", "authors", "written by", "created by", "built by",
            "developed by", "team", "student", "students", "contributor",
            "developer", "project",
        )):
            score += 4
        if re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text):
            score += 2
        if 2 <= len(text.split()) <= 24:
            score += 1

        ranked.append((score, page_num, len(text), node_id, text))

    if not ranked:
        return "", []

    ranked.sort(key=lambda x: (-x[0], x[1], -x[2]))

    picked_lines = []
    picked_ids = []
    used = set()
    total = 0

    for _, page_num, _, node_id, text in ranked:
        key = text[:120].lower()
        if key in used:
            continue
        used.add(key)

        line = f"[Page {page_num}] {text}"
        if total + len(line) + 1 > max_chars:
            break

        picked_lines.append(line)
        picked_ids.append(node_id)
        total += len(line) + 1

        if len(picked_lines) >= 45:
            break

    return "\n".join(picked_lines), picked_ids


def _claude_authorship_failsafe(query: str, graph, doc_meta: dict | None):
    """Use Claude best model for low-evidence authorship questions."""
    if not ANTHROPIC_API_KEY or not ANTHROPIC_BEST_MODEL or graph is None:
        return None

    context, line_ids = _collect_authorship_context(graph)
    if not context:
        return None

    title = (doc_meta or {}).get("detected_title", "Document")
    prompt = (
        f"Document: {title}\n\n"
        "You are extracting who made the project or who authored the document.\n"
        "Use ONLY the provided document excerpts.\n"
        "Rules:\n"
        "1) Return only the final answer text, no explanation.\n"
        "2) If multiple names exist, list all names separated by commas.\n"
        "3) If no names are present in the excerpts, return exactly: Insufficient document evidence.\n\n"
        f"Question: {query}\n\n"
        f"Document Excerpts:\n{context}\n\n"
        "Answer:"
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_BEST_MODEL,
                "max_tokens": 180,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
    except Exception as e:
        _debug_log(f"[ASL] Authorship failsafe request failed: {e}")
        return None

    if resp.status_code != 200:
        _debug_log(f"[ASL] Authorship failsafe API error {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        payload = resp.json()
        content = payload.get("content", [])
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_val = (item.get("text") or "").strip()
                if text_val:
                    text_parts.append(text_val)
        answer_text = "\n".join(text_parts).strip()
    except Exception:
        return None

    if not answer_text:
        return None

    low = answer_text.lower().strip()
    if low.startswith("insufficient document evidence"):
        return None
    if _llm_claims_no_document_evidence(answer_text):
        return None

    return {
        "answer_text": answer_text,
        "contributing_lines": line_ids[:8],
    }


def _llm_claims_no_document_evidence(text: str) -> bool:
    """Return True if the LLM answer explicitly claims the info isn't in the document."""
    if not text:
        return True
    low = text.lower()
    return any(phrase in low for phrase in _NO_DOC_EVIDENCE_PHRASES)


def _extract_facts_from_text(text: str) -> list[str]:
    """Split an answer into atomic fact sentences."""
    if not text:
        return []
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    facts = []
    for s in sentences:
        s = s.strip().strip("-•*").strip()
        if len(s) > 10:
            facts.append(s)
    return facts


def _score_answer(text: str, facts: list[str], query: str) -> float:
    """
    Improved scoring using content-word overlap (stopword-filtered),
    fact density, named-entity/number presence, and refusal detection.
    """
    if not text:
        return 0.0

    low = text.lower()
    # Penalty for refusals
    if any(p in low for p in ("insufficient knowledge", "no information found",
                               "i don't have", "not available", "does not contain",
                               "no relevant information")):
        return 0.05

    stop = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "with", "that", "this", "for", "it", "be",
        "has", "have", "had", "not", "as", "at", "by", "from", "can", "do",
        "does", "will", "would", "could", "should", "its", "their", "they",
        "we", "he", "she", "been", "about", "into", "also", "so", "very",
        "just", "there", "then", "these", "those", "some", "any",
    }

    def _content_words(t):
        return {w for w in re.findall(r"\w+", t.lower()) if w not in stop and len(w) > 2}

    score = 0.0
    score += min(len(text) / 500, 1.0) * 0.20                  # length
    score += min(len(facts) / 5, 1.0) * 0.25                   # fact richness

    # Content-word overlap (filtered)
    q_content = _content_words(query)
    a_content = _content_words(text)
    overlap = len(q_content & a_content)
    score += min(overlap / max(len(q_content), 1), 1.0) * 0.30

    # High-value token presence (numbers, proper nouns in answer)
    nums = len(re.findall(r"\d+\.?\d*%?", text))
    caps = len(re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text))
    score += min((nums + caps * 0.5) / 10, 1.0) * 0.15

    # Specificity bonus: answer uses terms from query directly
    query_bigrams = set()
    q_words = query.lower().split()
    for i in range(len(q_words) - 1):
        query_bigrams.add((q_words[i], q_words[i + 1]))
    a_words = low.split()
    a_bigrams = set()
    for i in range(len(a_words) - 1):
        a_bigrams.add((a_words[i], a_words[i + 1]))
    bigram_overlap = len(query_bigrams & a_bigrams)
    score += min(bigram_overlap / max(len(query_bigrams), 1), 1.0) * 0.10

    return round(min(score, 1.0), 3)


def _merge_facts(ev_facts: list[str], llm_facts: list[str]) -> list[str]:
    """De-duplicate and merge facts, preferring evidence-graph ones first."""
    seen_lower = set()
    merged = []
    for f in ev_facts + llm_facts:
        key = f.lower().strip()
        if key in seen_lower:
            continue
        # Substring dedup
        is_dup = False
        for existing in seen_lower:
            if key in existing or existing in key:
                is_dup = True
                break
        if not is_dup:
            seen_lower.add(key)
            merged.append(f)
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Final reframing LLM call
# ═══════════════════════════════════════════════════════════════════════════

def _reframe_answer(query: str, facts: list[str], title: str) -> str | None:
    """
    Send the merged fact list to the conversational LLM for a polished,
    coherent final answer.
    """
    if not facts or not OPENROUTER_API_KEY:
        return None

    fact_block = "\n".join(f"- {f}" for f in facts[:25])
    max_tokens = _compute_max_tokens(query, facts)
    _debug_log(f"[ASL] _reframe_answer adaptive max_tokens={max_tokens} (facts={len(facts)}, words={len(query.split())})")

    system = (
        "You are an expert Document Intelligence Assistant. You will receive a set "
        "of verified facts extracted from a document and a user question. Combine "
        "ALL the facts into a clear, well-structured, comprehensive answer. "
        "Include every relevant detail, number, name, and date from the facts. "
        "Use bold for key terms. Never refuse to answer — always synthesize what is available. "
        "Do NOT show reasoning, <think> tags, or chain-of-thought."
    )
    prompt = (
        f"### Document: {title}\n\n"
        f"### Verified Facts:\n{fact_block}\n\n"
        f"### User Question: {query}\n\n"
        "### Final Answer:"
    )

    # Try Google Gemini 2.5 Pro directly first
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
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content.strip()
                content = re.sub(r"<(think|thought|reasoning)>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<(think|thought|reasoning)>.*", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = content.strip()
                if content:
                    _debug_log(f"[ASL] Reframe via Google {GOOGLE_STRONG_MODEL}")
                    return content
        except Exception as e:
            _debug_log(f"[ASL] Google error, falling back to OpenRouter: {e}")

    models = _strong_models()
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
                temperature=0.1,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Document Intelligence System",
                },
            )
            content = resp.choices[0].message.content.strip()
            # Strip CoT artifacts
            content = re.sub(
                r"<(think|thought|reasoning)>.*?</\1>", "", content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            content = re.sub(
                r"<(think|thought|reasoning)>.*", "", content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if content:
                _debug_log(f"[ASL] Reframe answer via strong model: {model}")
            return content.strip() or None
        except Exception as e:
            err = str(e)
            if any(c in err for c in ("429", "404", "400", "401", "402", "503", "502")):
                continue
            _debug_log(f"[ASL] Critical error with {model}: {e}")
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Deep Analysis: strong LLM fallback on raw evidence (NotebookLM-quality)
# ═══════════════════════════════════════════════════════════════════════════

def _deep_analysis_answer(
    query: str, aligned_evidence: list, title: str
) -> str | None:
    """
    Send the RAW retrieved evidence chunks to the LLM with a comprehensive
    system prompt for high-quality, richly detailed answers.

    This is the secret weapon: when standard fact-extraction loses detail,
    the deep analysis path preserves the original document text and lets
    the LLM produce NotebookLM-caliber responses.
    """
    if not aligned_evidence or (not OPENROUTER_API_KEY and not GOOGLE_API_KEY):
        return None

    # Build evidence block from raw retrieval chunks — use ALL available
    evidence_parts = []
    for item in aligned_evidence[:15]:
        text = item.get("text", "").strip()
        if not text:
            continue
        section = item.get("metadata", {}).get("section", "")
        page = item.get("metadata", {}).get("page", "")
        prefix = ""
        if section:
            prefix += f"[{section}]"
        if page:
            prefix += f" (p.{page})"
        evidence_parts.append(f"{prefix} {text}" if prefix else text)

    if not evidence_parts:
        return None

    evidence_block = "\n\n".join(evidence_parts)
    max_tokens = _compute_max_tokens(query, evidence_parts)
    _debug_log(f"[ASL] _deep_analysis adaptive max_tokens={max_tokens} (ev={len(evidence_parts)}, words={len(query.split())})")

    system = (
        "You are an expert document research assistant. Your job is to answer "
        "questions using the evidence excerpts provided from a document. "
        "CRITICAL RULES:\n"
        "- ALWAYS provide an answer if ANY relevant information exists in the evidence\n"
        "- Be thorough: include ALL relevant details, numbers, dates, and names\n"
        "- Structure clearly with headings, bullet points, or numbered lists when helpful\n"
        "- Use **bold** for key terms, entities, and important values\n"
        "- Quote exact figures, percentages, and proper nouns from the evidence\n"
        "- Be confident and direct — present information as established fact\n"
        "- Even if the evidence is partial, extract and present what IS available\n"
        "- NEVER refuse to answer — always synthesize whatever evidence is given\n"
        "- NEVER say 'the document does not contain' or 'no information available'\n"
        "- NEVER say 'according to the document' — present facts directly\n"
        "- Do NOT show reasoning, <think> tags, or chain-of-thought.\n"
    )

    prompt = (
        f"### Document: {title}\n\n"
        f"### Evidence Excerpts:\n{evidence_block}\n\n"
        f"### Question: {query}\n\n"
        "Using the evidence above, provide a comprehensive, well-structured answer. "
        "Include every specific detail found in the evidence.\n\n"
        "### Answer:"
    )

    # Try Google Gemini 2.5 Pro directly first
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
                content = re.sub(r"<(think|thought|reasoning)>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<(think|thought|reasoning)>.*", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = content.strip()
                if content:
                    _debug_log(f"[ASL] Deep analysis via Google {GOOGLE_STRONG_MODEL}")
                    return content
        except Exception as e:
            _debug_log(f"[ASL] Google error, falling back to OpenRouter: {e}")

    models = _strong_models()
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
            content = re.sub(
                r"<(think|thought|reasoning)>.*?</\1>", "", content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            content = re.sub(
                r"<(think|thought|reasoning)>.*", "", content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if content:
                _debug_log(f"[ASL] Deep analysis via strong model: {model}")
            return content.strip() or None
        except Exception as e:
            err = str(e)
            if any(c in err for c in ("429", "404", "400", "401", "402", "503", "502")):
                continue
            _debug_log(f"[ASL] Deep analysis error with {model}: {e}")
            return None
    return None
