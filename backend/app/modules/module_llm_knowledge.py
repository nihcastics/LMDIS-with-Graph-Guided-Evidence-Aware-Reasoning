"""
Module LLM-K: Direct LLM Document Knowledge Store
====================================================
During ingestion, sends representative document text to the LLM and
stores the structured knowledge response on the graph as a backup.

At query time, the LLM is asked the same user question directly
against the stored knowledge, producing a *direct LLM answer* that
runs in parallel with the evidence-graph-based answer.

Both answers are later merged by the Answer Selection Layer.
"""

from openai import OpenAI
import json
import os
import re
from backend.app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_FALLBACK_MODELS,
)

# Cached OpenAI client
_openai_client = None
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


# ═══════════════════════════════════════════════════════════════════════════
# INGESTION-TIME: extract and store LLM knowledge on the graph
# ═══════════════════════════════════════════════════════════════════════════

def store_llm_knowledge(G, doc_meta=None):
    """
    Collect representative text from the graph, send it to the LLM
    once during ingestion, and persist the structured knowledge as
    a special ``llm_knowledge`` node on *G*.

    Returns the (possibly mutated) graph.
    """
    # ── 1. Collect representative document text ──
    text_nodes = []
    for n, d in G.nodes(data=True):
        if d.get("type") in ("text_component", "paragraph"):
            text_nodes.append((n, d))

    # Sort by page then reading order to preserve document flow
    text_nodes.sort(key=lambda x: (x[1].get("page", 0), x[1].get("reading_order", 0)))

    # Build a condensed document representation (~3000 chars max for speed)
    doc_parts = []
    char_budget = 3000
    total = 0
    for _, d in text_nodes:
        txt = d.get("text", "").strip()
        if not txt:
            continue
        if total + len(txt) > char_budget:
            break
        doc_parts.append(txt)
        total += len(txt)

    if not doc_parts:
        return G  # nothing to send

    doc_text = "\n".join(doc_parts)
    title = (doc_meta or {}).get("detected_title", "Unknown Document")

    # ── 2. Ask LLM for structured knowledge ──
    knowledge_json = _extract_knowledge_via_llm(doc_text, title)

    # ── 3. Store on graph as a special node ──
    G.add_node(
        "llm_knowledge_store",
        type="llm_knowledge",
        title=title,
        knowledge=knowledge_json,
        source_char_count=total,
    )

    _debug_log(f"[LLM-K] Stored LLM knowledge ({len(knowledge_json)} chars) for '{title}'")
    return G


# ═══════════════════════════════════════════════════════════════════════════
# QUERY-TIME: ask the LLM directly using stored knowledge
# ═══════════════════════════════════════════════════════════════════════════

def query_llm_directly(G, query, doc_meta=None):
    """
    Retrieve the stored LLM knowledge from the graph and ask the LLM
    to answer the user query using that knowledge.  Returns a dict
    ``{"answer": str, "facts": list[str]}`` or *None* on failure.
    """
    knode = G.nodes.get("llm_knowledge_store")
    if not knode or not knode.get("knowledge"):
        return None

    knowledge_text = knode["knowledge"]
    title = (doc_meta or {}).get("detected_title", knode.get("title", "Document"))

    return _answer_from_knowledge(query, knowledge_text, title)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _call_llm(messages, temperature=0.1, max_tokens=1500):
    """Call OpenRouter with automatic model fallback. Returns content str or None."""
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
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Document Intelligence System",
                },
            )
            content = resp.choices[0].message.content.strip()
            # Strip chain-of-thought artifacts
            content = re.sub(
                r"<(think|thought|reasoning)>.*?</\1>", "", content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            content = re.sub(
                r"<(think|thought|reasoning)>.*", "", content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            return content.strip()
        except Exception as e:
            err = str(e)
            if any(c in err for c in ("429", "404", "400", "401", "402", "503", "502")):
                continue
            _debug_log(f"[LLM-K] Critical error with {model}: {e}")
            return None
    return None


def _extract_knowledge_via_llm(doc_text, title):
    """
    Send representative document text to the LLM and ask for a
    structured knowledge digest.  Returns the raw string response.
    """
    system = (
        "You are a document analysis expert. Given the text from a document, "
        "produce a structured knowledge summary. Include:\n"
        "1. MAIN TOPICS (bullet list)\n"
        "2. KEY FACTS (bullet list of concrete facts, numbers, dates)\n"
        "3. DEFINITIONS (any terms defined in the document)\n"
        "4. RELATIONSHIPS (connections between entities/concepts)\n"
        "5. CONCLUSIONS (main conclusions or findings)\n\n"
        "Be concise and factual. Do NOT add information not present in the text."
    )
    prompt = (
        f"### Document Title: {title}\n\n"
        f"### Document Text:\n{doc_text}\n\n"
        "### Structured Knowledge Summary:"
    )
    result = _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return result or ""


def _answer_from_knowledge(query, knowledge_text, title):
    """
    Ask the LLM to answer the user query given the stored knowledge.
    Returns ``{"answer": str, "facts": list[str]}`` or None.
    """
    system = (
        "You are an expert Document Intelligence Assistant. Answer the question "
        "using ONLY the knowledge provided. After your answer, list the specific "
        "facts you used in a section labelled FACTS_USED (one fact per line, "
        "prefixed with '- '). If the knowledge does not contain the answer, "
        "say 'Insufficient knowledge' and list no facts."
    )
    prompt = (
        f"### Document: {title}\n\n"
        f"### Stored Knowledge:\n{knowledge_text}\n\n"
        f"### Question: {query}\n\n"
        "### Answer:"
    )
    raw = _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1500,
    )
    if not raw:
        return None

    # Parse answer and facts
    answer_part = raw
    facts = []
    if "FACTS_USED" in raw:
        parts = raw.split("FACTS_USED", 1)
        answer_part = parts[0].strip()
        fact_block = parts[1] if len(parts) > 1 else ""
        for line in fact_block.split("\n"):
            line = line.strip().lstrip(":").strip()
            if line.startswith("- "):
                facts.append(line[2:].strip())
            elif line and not line.startswith("#"):
                facts.append(line)

    return {"answer": answer_part, "facts": facts}
