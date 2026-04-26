import os
import json
import re
from openai import OpenAI
from backend.app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS

# Cached OpenAI client
_openai_client = None

def _get_client():
    global _openai_client
    if _openai_client is None and OPENROUTER_API_KEY:
        _openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    return _openai_client


def _call_llm(prompt, system_msg="You are a precise query analysis assistant.", max_tokens=1000):
    """Call OpenRouter with automatic model fallback."""
    if not OPENROUTER_API_KEY:
        return None
    models = [OPENROUTER_MODEL] + OPENROUTER_FALLBACK_MODELS
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
            if any(c in str(e) for c in ["429", "404", "400", "503", "502"]):
                continue
            return None
    return None


def interpret_query(query_text):
    """
    Module 12: Deep Query Interpretation & Semantic Decomposition

    Performs LLM-based deep analysis of the query to extract:
    - Semantic intent (not just keyword matching)
    - Key entities and concepts to search for
    - Expected answer type (definition, number, list, explanation, etc.)
    - Multiple search claims (paraphrased for broader retrieval)
    - Query expansion terms (synonyms, related concepts)
    """
    query_lower = query_text.lower()

    # ── 1. Heuristic intent pre-classification (fast fallback) ──
    intent = "factual_lookup"
    if any(k in query_lower for k in ["compare", "difference", "versus", "vs", "differ", "contrast"]):
        intent = "comparative"
    elif any(k in query_lower for k in ["summarize", "summary", "overview", "brief", "describe overall"]):
        intent = "summarization"
    elif any(k in query_lower for k in ["how many", "what percentage", "what number", "how much", "count of"]):
        intent = "quantitative"
    elif any(k in query_lower for k in ["what is", "define", "meaning of", "what does", "what are"]):
        intent = "definition"
    elif any(k in query_lower for k in ["why", "reason", "cause", "because", "explain why"]):
        intent = "causal"
    elif any(k in query_lower for k in ["how does", "how do", "how is", "process", "method", "procedure", "steps"]):
        intent = "procedural"
    elif any(k in query_lower for k in ["list", "enumerate", "what are the", "name the", "which are"]):
        intent = "enumerative"

    # ── 2. LLM-Based Deep Query Analysis ──
    llm_analysis = _deep_query_analysis(query_text)

    if llm_analysis:
        intent = llm_analysis.get("intent", intent)
        claims = llm_analysis.get("claims", [query_text])
        key_entities = llm_analysis.get("key_entities", [])
        expected_answer_type = llm_analysis.get("expected_answer_type", "text")
        expanded_terms = llm_analysis.get("expanded_terms", [])
    else:
        claims = [query_text]
        key_entities = _extract_key_entities_heuristic(query_text)
        expected_answer_type = _infer_answer_type(query_lower)
        expanded_terms = []

    # Always include the original query as a claim for direct matching
    if query_text not in claims:
        claims.insert(0, query_text)

    # ── 3. Heuristic relationship expansions for common queries ──
    # When LLM expansion is limited, add relationship-based claims and terms
    claims, expanded_terms = _add_relationship_expansions(
        query_lower, claims, expanded_terms, key_entities
    )

    return {
        "intent": intent,
        "original_text": query_text,
        "claims": claims,
        "key_entities": key_entities,
        "expected_answer_type": expected_answer_type,
        "expanded_terms": expanded_terms,
        "requires_reasoning": intent in ("comparative", "summarization", "causal"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Relationship-based query expansion (heuristic supplement)
# ═══════════════════════════════════════════════════════════════════════════

_RELATIONSHIP_EXPANSIONS = {
    "author": {
        "claims": ["author of the document", "written by", "paper by", "contributed by"],
        "terms": ["author", "authors", "written", "writer", "contributor", "name", "by", "affiliation"],
    },
    "student": {
        "claims": ["student contributors", "students who made this project", "project built by students", "project team members"],
        "terms": ["student", "students", "team", "members", "contributors", "developers", "built by", "created by"],
    },
    "team": {
        "claims": ["project team", "who made this project", "who built this project", "developers of this project"],
        "terms": ["team", "members", "developer", "developers", "creator", "contributors", "made", "built", "created"],
    },
    "title": {
        "claims": ["title of the document", "document name", "paper title"],
        "terms": ["title", "name", "heading", "subject", "topic"],
    },
    "date": {
        "claims": ["publication date", "when was this published", "year of publication"],
        "terms": ["date", "published", "year", "issued", "received", "accepted", "submitted"],
    },
    "published": {
        "claims": ["publication date", "when published", "date of publication"],
        "terms": ["published", "date", "year", "journal", "venue", "issued"],
    },
    "abstract": {
        "claims": ["abstract of the paper", "paper summary", "document overview"],
        "terms": ["abstract", "summary", "overview", "introduction"],
    },
    "conclusion": {
        "claims": ["conclusion of the paper", "main findings", "key results"],
        "terms": ["conclusion", "findings", "results", "summary", "outcome", "determined"],
    },
    "methodology": {
        "claims": ["methodology used", "research method", "approach described"],
        "terms": ["method", "methodology", "approach", "technique", "procedure", "algorithm"],
    },
    "result": {
        "claims": ["research results", "findings of the study", "outcomes achieved"],
        "terms": ["result", "results", "finding", "findings", "outcome", "achieved", "performance", "accuracy"],
    },
    "purpose": {
        "claims": ["purpose of the document", "goal of the research", "objective"],
        "terms": ["purpose", "goal", "objective", "aim", "motivation", "research question"],
    },
    "reference": {
        "claims": ["references cited", "bibliography", "cited works"],
        "terms": ["reference", "references", "bibliography", "cited", "citation"],
    },
}


def _add_relationship_expansions(query_lower, claims, expanded_terms, key_entities):
    """Add heuristic relationship-based expansions for common query concepts."""
    q_words = set(query_lower.split())
    entity_words = {e.lower() for e in (key_entities or [])}
    all_words = q_words | entity_words

    for trigger, expansion in _RELATIONSHIP_EXPANSIONS.items():
        if trigger in all_words or any(trigger in w for w in all_words if len(w) >= 4):
            for claim in expansion["claims"]:
                if claim not in claims:
                    claims.append(claim)
            for term in expansion["terms"]:
                if term not in expanded_terms:
                    expanded_terms.append(term)

    return claims, expanded_terms


def _deep_query_analysis(query_text):
    """Use LLM for comprehensive semantic query decomposition.

    Generates diverse search queries that capture the semantic intent,
    not just keywords. For example, 'Who is the author?' generates claims
    like 'written by', 'author name', 'contributor', etc.
    """
    prompt = (
        "Analyze this document question and return a JSON object with these fields:\n\n"
        "1. \"intent\": one of [factual_lookup, comparative, summarization, quantitative, "
        "definition, causal, procedural, enumerative]\n"
        "2. \"claims\": array of 3-6 diverse search queries that capture the SEMANTIC MEANING "
        "of the question. Include:\n"
        "   - The original phrasing\n"
        "   - Paraphrases using DIFFERENT vocabulary (e.g. 'author' → 'written by', 'name of the writer')\n"
        "   - Relationship-based queries (e.g. 'Who is the author?' → 'paper by', 'contributed by')\n"
        "   - Metadata-style queries when applicable (e.g. 'published date', 'journal name')\n"
        "   - Short keyword-style queries (e.g. 'author name')\n"
        "3. \"key_entities\": array of key nouns/concepts/proper names from the question\n"
        "4. \"expected_answer_type\": one of [number, percentage, date, name, definition, "
        "list, explanation, yes_no, description]\n"
        "5. \"expanded_terms\": array of 5-10 synonyms, related terms, and conceptually "
        "associated words for the key concepts. Think about what words would appear in a "
        "document passage that ANSWERS this question, even if those words don't appear "
        "in the question itself. For example:\n"
        "   - 'author' → ['written', 'by', 'contributor', 'researcher', 'affiliation', 'name']\n"
        "   - 'published' → ['date', 'year', 'journal', 'venue', 'conference', 'volume']\n"
        "   - 'results' → ['findings', 'outcome', 'accuracy', 'performance', 'achieved']\n\n"
        f"Question: \"{query_text}\"\n\n"
        "Return ONLY the JSON object, no explanation."
    )
    result = _call_llm(prompt, max_tokens=600)
    if not result:
        return None
    try:
        m = re.search(r'\{.*\}', result, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            # Validate fields
            if isinstance(parsed.get("claims"), list) and len(parsed["claims"]) >= 1:
                return parsed
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _extract_key_entities_heuristic(query_text):
    """Fallback entity extraction from query using simple heuristics."""
    stop = {
        "what", "which", "how", "where", "when", "who", "why", "does", "do",
        "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "for",
        "to", "and", "or", "this", "that", "with", "from", "about", "can",
        "could", "would", "should", "will", "be", "been", "being", "have",
        "has", "had", "not", "it", "its", "they", "their", "them", "there",
        "describe", "explain", "tell", "me", "give", "find", "show",
    }
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9-]+\b', query_text)
    entities = [w for w in words if w.lower() not in stop and len(w) > 2]
    return entities[:8]


def _infer_answer_type(query_lower):
    """Infer expected answer type from question words."""
    if any(k in query_lower for k in ["how many", "how much", "what number", "count"]):
        return "number"
    if "percentage" in query_lower or "percent" in query_lower:
        return "percentage"
    if any(k in query_lower for k in ["when", "what date", "what year"]):
        return "date"
    if any(k in query_lower for k in ["who", "whose", "whom"]):
        return "name"
    if any(k in query_lower for k in ["define", "definition", "what is a", "what are"]):
        return "definition"
    if any(k in query_lower for k in ["list", "enumerate", "name the"]):
        return "list"
    if any(k in query_lower for k in ["why", "reason", "cause"]):
        return "explanation"
    if any(k in query_lower for k in ["is it", "does it", "can it", "has it", "are there"]):
        return "yes_no"
    return "description"
