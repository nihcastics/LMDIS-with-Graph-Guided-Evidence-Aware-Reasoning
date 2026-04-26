def format_response(
    generated_result,
    G,
    aligned_evidence=None,
    ev_answer=None,
    llm_direct=None,
    structured_query=None,
):
    """
    Module 20: Final Answer Assembly, Citation Linking & Analysis Payload
    """
    if not generated_result:
        return {
            "answer": "No information found in the document.",
            "citations": [],
            "confidence": 0.0,
            "analysis": _empty_analysis(),
        }

    answer_text = generated_result["answer_text"]
    contributing_lines = set(generated_result.get("contributing_lines", []))
    source_path = generated_result.get("source_path", "evidence_graph")

    # ── Citations: built from aligned_evidence (authoritative scored data) ─
    citations = []
    confidence_sum = 0.0
    seen_keys = set()

    # Build citations from aligned_evidence — the cross-encoder re-ranked,
    # score-bearing evidence items from the retrieval pipeline.
    ev_items = sorted(
        aligned_evidence or [],
        key=lambda x: x.get("alignment_score", 0),
        reverse=True,
    )

    for item in ev_items:
        line_id = item.get("line_id", "")
        meta = item.get("metadata", {})
        page = meta.get("page", "?")
        section = meta.get("section", "Unknown Section")
        info_type = meta.get("information_type", "")
        text = item.get("text", "").strip()
        alignment = item.get("alignment_score", 0.0)
        ce_score = item.get("cross_encoder_score")
        ocr_conf = item.get("ocr_confidence", 1.0)

        if not text:
            continue

        # Dedup by (page, section, text_prefix) — prevents near-duplicate citations
        dedup_key = f"p{page}|{section}|{text[:50]}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        # Boost items that M16-19 independently identified as contributing
        contribution_boost = 0.10 if line_id in contributing_lines else 0.0

        # Composite citation confidence: blends retrieval quality + OCR quality
        cite_confidence = round(
            min(0.40 * alignment + 0.30 * ocr_conf + 0.20 * min(max(
                (ce_score + 5) / 10, 0) if ce_score is not None else 0.5, 1.0
            ) + 0.10 + contribution_boost, 1.0),
            3,
        )

        # Resolve human-readable section title from graph if possible
        section_title = section
        if section in G.nodes:
            section_title = G.nodes[section].get("label", section)

        citations.append({
            "source": f"Page {page} ({section_title})",
            "page": page,
            "section": section_title,
            "snippet": text[:150] + ("..." if len(text) > 150 else ""),
            "confidence": cite_confidence,
            "alignment_score": round(alignment, 3),
            "information_type": info_type,
            "contributed": line_id in contributing_lines,
        })
        confidence_sum += cite_confidence

    # Fallback: if aligned_evidence was empty, build from contributing_lines + graph
    if not citations and contributing_lines:
        for line_id in contributing_lines:
            if line_id not in G:
                continue
            nd = G.nodes[line_id]
            page = nd.get("page", "?")
            sec_id = nd.get("section_id", "Unknown Section")
            sec_title = G.nodes[sec_id].get("label", sec_id) if sec_id in G.nodes else sec_id
            text = nd.get("text", "").strip()
            if not text:
                continue
            dk = f"p{page}|{sec_title}|{text[:50]}"
            if dk in seen_keys:
                continue
            seen_keys.add(dk)
            conf = round(nd.get("ocr_confidence", 1.0), 3)
            citations.append({
                "source": f"Page {page} ({sec_title})",
                "page": page,
                "section": sec_title,
                "snippet": text[:150] + ("..." if len(text) > 150 else ""),
                "confidence": conf,
                "alignment_score": 0.0,
                "information_type": "",
                "contributed": True,
            })
            confidence_sum += conf

    avg_confidence = confidence_sum / len(citations) if citations else 1.0

    # ── Analysis payload ──────────────────────────────────────────────────
    analysis = _build_analysis(
        G,
        aligned_evidence or [],
        ev_answer,
        llm_direct,
        generated_result,
        structured_query or {},
        citations,
        avg_confidence,
    )

    return {
        "answer": answer_text,
        "crisp_answer": generated_result.get("claude_crisp_answer") or None,
        "citations": citations,
        "precision_score": round(avg_confidence, 2),
        "confidence": round(avg_confidence, 2),
        "total_sources": len(citations),
        "analysis": analysis,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis builder
# ═══════════════════════════════════════════════════════════════════════════

def _empty_analysis():
    return {
        "source_path": "none",
        "facts": [],
        "evidence_graph": {"nodes": [], "edges": []},
        "pipeline": [],
        "retrieval_stats": {},
        "confidence_breakdown": {},
        "dual_path_comparison": None,
        "raw_evidence": [],
    }


def _build_analysis(
    G, aligned_evidence, ev_answer, llm_direct, merged, sq, citations, avg_conf
):
    import re

    source_path = merged.get("source_path", "evidence_graph")

    # ── 1. Fact breakdown ─────────────────────────────────────────────────
    facts = _build_facts(merged.get("answer_text", ""), aligned_evidence, G, source_path)

    # ── 2. Evidence graph mini-graph ──────────────────────────────────────
    ev_graph = _build_evidence_subgraph(G, aligned_evidence, sq)

    # ── 3. Pipeline steps ─────────────────────────────────────────────────
    pipeline = _build_pipeline_steps(source_path, aligned_evidence, ev_answer, llm_direct)

    # ── 4. Retrieval statistics ───────────────────────────────────────────
    retrieval_stats = _build_retrieval_stats(aligned_evidence, sq)
    graph_nodes = ev_graph.get("nodes", [])
    graph_edges = ev_graph.get("edges", [])
    retrieval_stats["graph_nodes"] = len(graph_nodes)
    retrieval_stats["graph_edges"] = len(graph_edges)
    retrieval_stats["graph_evidence_nodes"] = sum(
        1 for node in graph_nodes if (node or {}).get("type") == "evidence"
    )

    # ── 5. Confidence breakdown ───────────────────────────────────────────
    confidence_breakdown = _build_confidence_breakdown(
        aligned_evidence, citations, avg_conf, source_path
    )

    # ── 6. Dual-path comparison (from answer selection layer) ─────────────
    dual_path = merged.get("dual_path_comparison", None)

    # ── 7. Raw evidence items for display ─────────────────────────────────
    raw_evidence = _build_raw_evidence(aligned_evidence)

    return {
        "source_path": source_path,
        "facts": facts,
        "evidence_graph": ev_graph,
        "pipeline": pipeline,
        "retrieval_stats": retrieval_stats,
        "confidence_breakdown": confidence_breakdown,
        "dual_path_comparison": dual_path,
        "raw_evidence": raw_evidence,
    }


def _build_facts(answer_text, aligned_evidence, G, source_path):
    """Extract clean fact sentences with source attribution."""
    import re

    facts = []
    if not answer_text:
        return facts

    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    for sent in sentences:
        sent = sent.strip().strip("-•* ").strip()
        if len(sent) < 12:
            continue
        # Find best matching evidence line for attribution
        attribution = _attribute_fact(sent, aligned_evidence, G)
        facts.append({
            "text": sent,
            "source": attribution.get("source", ""),
            "page": attribution.get("page", ""),
            "confidence": attribution.get("confidence", 0.0),
            "grounded": attribution.get("grounded", False),
        })

    return facts


def _attribute_fact(fact_text, aligned_evidence, G):
    """
    Find the best-matching evidence item for a fact sentence.
    Uses multi-signal matching: unigram overlap, number/entity match, alignment score.
    """
    import re as _re

    if not aligned_evidence:
        return {"source": "Direct synthesis", "page": "", "confidence": 0.5, "grounded": False}

    stop = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "with", "which", "that", "this", "for",
        "it", "be", "has", "have", "had", "not", "as", "at", "by", "from",
    }

    def _content_tokens(text):
        return {w for w in _re.findall(r"\w+", text.lower()) if w not in stop and len(w) > 1}

    def _numbers(text):
        return set(_re.findall(r"\d+\.?\d*%?", text))

    fact_words = _content_tokens(fact_text)
    fact_nums = _numbers(fact_text)

    best_score = 0.0
    best_item = None

    for item in aligned_evidence:
        ev_text = item.get("text", "")
        ev_words = _content_tokens(ev_text)
        ev_nums = _numbers(ev_text)

        # Word overlap ratio
        word_overlap = len(fact_words & ev_words)
        word_ratio = word_overlap / max(len(fact_words), 1)

        # Number/entity overlap (very strong signal for factual claims)
        num_overlap = len(fact_nums & ev_nums) if fact_nums else 0
        num_ratio = num_overlap / max(len(fact_nums), 1) if fact_nums else 0.0

        # Alignment prior from retrieval pipeline
        align = item.get("alignment_score", 0.0)

        combined = 0.45 * word_ratio + 0.30 * num_ratio + 0.25 * align
        if combined > best_score:
            best_score = combined
            best_item = item

    if best_item and best_score >= 0.10:
        meta = best_item.get("metadata", {})
        section = meta.get("section", "")
        page = meta.get("page", "")
        source_label = f"Page {page}" if page else "Document"
        if section and section != "Unknown Section":
            source_label += f" ({section})"
        return {
            "source": source_label,
            "page": str(page) if page else "",
            "confidence": round(best_item.get("alignment_score", 0.5), 2),
            "grounded": True,
        }

    return {"source": "Synthesized from context", "page": "", "confidence": 0.4, "grounded": False}


def _build_evidence_subgraph(G, aligned_evidence, sq):
    """Build a visualization-ready mini-graph from retrieved evidence."""
    nodes = []
    edges = []
    seen_nodes = set()

    # Query node
    nodes.append({
        "id": "query",
        "label": (sq.get("original_text", "Query") or "Query")[:40],
        "type": "query",
        "confidence": 1.0,
    })
    seen_nodes.add("query")

    # Collect section and evidence nodes from aligned evidence
    section_map = {}  # section_id -> section_label
    for idx, item in enumerate(aligned_evidence or []):
        if idx >= 15:
            break
        line_id = item.get("line_id", f"ev_{idx}")
        meta = item.get("metadata", {})
        page = meta.get("page", "")
        section = meta.get("section", "")
        section_id = item.get("section_id", "")
        score = round(item.get("alignment_score", 0.5), 2)
        info_type = meta.get("information_type", "")
        text = item.get("text", "")[:80]

        # Section node (group by section)
        if section_id and section_id not in seen_nodes:
            section_label = section if section and section != "Unknown Section" else f"Section (p.{page})"
            nodes.append({
                "id": section_id,
                "label": section_label[:30],
                "type": "section",
                "confidence": score,
            })
            seen_nodes.add(section_id)
            edges.append({
                "source": "query",
                "target": section_id,
                "relationship": "RETRIEVES",
                "weight": score,
            })
            section_map[section_id] = section_label

        # Evidence node
        ev_node_id = f"ev_{idx}"
        ev_label = f"P{page}" if page else f"Evidence {idx+1}"
        if info_type:
            ev_label += f" [{info_type}]"
        nodes.append({
            "id": ev_node_id,
            "label": ev_label[:30],
            "type": "evidence",
            "confidence": score,
            "snippet": text,
            "page": str(page) if page else "",
        })
        seen_nodes.add(ev_node_id)

        # Edge from section to evidence
        if section_id and section_id in seen_nodes:
            edges.append({
                "source": section_id,
                "target": ev_node_id,
                "relationship": "CONTAINS",
                "weight": score,
            })
        else:
            edges.append({
                "source": "query",
                "target": ev_node_id,
                "relationship": "RETRIEVES",
                "weight": score,
            })

    # Add cross-evidence edges for items from the same page
    ev_by_page = {}
    for idx, item in enumerate(aligned_evidence or []):
        if idx >= 15:
            break
        page = (item.get("metadata") or {}).get("page", "")
        if page:
            ev_by_page.setdefault(page, []).append(f"ev_{idx}")
    for page, ev_ids in ev_by_page.items():
        for i in range(len(ev_ids) - 1):
            edges.append({
                "source": ev_ids[i],
                "target": ev_ids[i + 1],
                "relationship": "ADJACENT",
                "weight": 0.4,
            })

    return {"nodes": nodes, "edges": edges}


def _build_pipeline_steps(source_path, aligned_evidence, ev_answer, llm_direct):
    """Describe which processing pipeline steps contributed to the answer."""
    steps = [
        {
            "name": "Query Interpretation",
            "module": "M12",
            "status": "completed",
            "detail": "Parsed query intent and extracted sub-claims",
        },
        {
            "name": "Semantic Retrieval",
            "module": "M13-15",
            "status": "completed",
            "detail": f"Retrieved {len(aligned_evidence or [])} evidence items via bi-encoder + graph traversal + cross-encoder re-ranking",
        },
    ]

    ev_text = (ev_answer or {}).get("answer_text", "")
    if ev_text:
        steps.append({
            "name": "Evidence-Graph Generation",
            "module": "M16-19",
            "status": "completed",
            "detail": "Generated answer from graph-retrieved evidence with source attribution",
        })
    else:
        steps.append({
            "name": "Evidence-Graph Generation",
            "module": "M16-19",
            "status": "skipped",
            "detail": "Insufficient evidence for graph-based answer",
        })

    llm_text = (llm_direct or {}).get("answer", "")
    if llm_text:
        steps.append({
            "name": "LLM Knowledge Backup",
            "module": "LLM-K",
            "status": "completed",
            "detail": "Retrieved stored document knowledge as verification path",
        })
    else:
        steps.append({
            "name": "LLM Knowledge Backup",
            "module": "LLM-K",
            "status": "skipped",
            "detail": "LLM knowledge path not activated",
        })

    path_labels = {
        "merged": "Dual-path fact merge selected — combined evidence-graph + LLM knowledge",
        "evidence_graph": "Evidence-graph path selected — answer grounded in document structure",
        "llm_direct": "LLM knowledge path selected — supplemental knowledge used",
        "deep_analysis": "Deep analysis selected — comprehensive answer from raw evidence via strong LLM",
        "none": "No sufficient path — answer could not be produced",
    }
    steps.append({
        "name": "Answer Selection",
        "module": "ASL",
        "status": "completed" if source_path != "none" else "failed",
        "detail": path_labels.get(source_path, "Answer path resolved"),
    })

    steps.append({
        "name": "Response Assembly",
        "module": "M20",
        "status": "completed",
        "detail": "Citations linked, confidence scored, analysis payload built",
    })

    return steps


def _build_retrieval_stats(aligned_evidence, sq):
    """Aggregate retrieval statistics for the analysis panel."""
    items = aligned_evidence or []
    if not items:
        return {
            "total_retrieved": 0,
            "avg_score": 0,
            "top_score": 0,
            "sections_covered": 0,
            "pages_covered": 0,
            "intent": sq.get("intent", "unknown"),
        }

    scores = [it.get("alignment_score", 0) for it in items]
    sections = set()
    pages = set()
    for it in items:
        meta = it.get("metadata", {})
        if meta.get("section"):
            sections.add(meta["section"])
        if meta.get("page"):
            pages.add(meta["page"])

    return {
        "total_retrieved": len(items),
        "avg_score": round(sum(scores) / len(scores), 3),
        "top_score": round(max(scores), 3),
        "sections_covered": len(sections),
        "pages_covered": len(pages),
        "intent": sq.get("intent", "unknown"),
    }


def _build_confidence_breakdown(aligned_evidence, citations, avg_conf, source_path):
    """Multi-factor confidence breakdown."""
    items = aligned_evidence or []

    # OCR confidence from citations
    ocr_scores = [c["confidence"] for c in citations if c.get("confidence")]
    ocr_avg = round(sum(ocr_scores) / len(ocr_scores), 3) if ocr_scores else 1.0

    # Retrieval alignment scores
    alignment_scores = [it.get("alignment_score", 0) for it in items]
    align_avg = round(sum(alignment_scores) / len(alignment_scores), 3) if alignment_scores else 0.0

    # Cross-encoder scores (if present)
    ce_scores = [it.get("cross_encoder_score", 0) for it in items if "cross_encoder_score" in it]
    ce_avg = round(sum(ce_scores) / len(ce_scores), 3) if ce_scores else None

    # Source grounding factor
    grounding = {"merged": 0.95, "evidence_graph": 1.0, "llm_direct": 0.6, "deep_analysis": 0.92, "none": 0.0}
    grounding_score = grounding.get(source_path, 0.5)

    return {
        "overall": round(avg_conf, 3),
        "ocr_quality": ocr_avg,
        "semantic_alignment": align_avg,
        "cross_encoder": ce_avg,
        "source_grounding": grounding_score,
        "evidence_coverage": min(len(items) / 10.0, 1.0),
    }


def _build_raw_evidence(aligned_evidence):
    """Build a display-friendly list of raw retrieved evidence items."""
    items = []
    for idx, item in enumerate(aligned_evidence or []):
        if idx >= 15:
            break
        text = item.get("text", "").strip()
        if not text:
            continue
        meta = item.get("metadata", {})
        items.append({
            "rank": idx + 1,
            "text": text[:500],
            "page": meta.get("page", ""),
            "section": meta.get("section", ""),
            "information_type": meta.get("information_type", ""),
            "alignment_score": round(item.get("alignment_score", 0), 3),
            "cross_encoder_score": round(item.get("cross_encoder_score", 0), 3) if item.get("cross_encoder_score") is not None else None,
            "ocr_confidence": round(item.get("ocr_confidence", 1.0), 3),
        })
    return items
