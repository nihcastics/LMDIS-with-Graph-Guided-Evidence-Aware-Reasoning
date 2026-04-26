"""
Module 11: Deep Evidence Graph Construction
===============================================
Builds a rich evidence overlay on the document memory graph.

1. Claim nodes with SUPPORTED_BY edges to source lines.
2. Inter-claim edges: SUPPORTS, DEPENDS_ON, RELATED_TO, LINKED_TO, CONTRADICTS.
3. Evidence chains: multi-hop paths from query-relevant nodes through
   structural + semantic edges for deep reasoning.
4. Section-level evidence summary nodes for fast retrieval.
"""

import re
import networkx as nx


def build_evidence_graph(G, claims):
    """Layer claims onto the document graph with deep cross-linking."""

    # ═══ 1. Claim Nodes ═══
    for claim in claims:
        cid = claim["claim_id"]
        G.add_node(cid, **{**claim, "type": "claim"})
        for src in claim["source_lines"]:
            if src in G:
                G.add_edge(cid, src, relationship="SUPPORTED_BY")

    # ═══ 2. Inter-Claim Relationships ═══
    claim_map = {c["claim_id"]: c for c in claims}
    definitions = {}
    for c in claims:
        if c["type"] == "fact_definition":
            parts = c["value"].split("=")
            if len(parts) >= 2:
                definitions[parts[0].strip().lower()] = c["claim_id"]

    for i, c1 in enumerate(claims):
        c1_id = c1["claim_id"]
        c1_text = c1["value"].lower()

        # A. Definition dependency
        for term, def_id in definitions.items():
            if def_id != c1_id and term in c1_text:
                G.add_edge(c1_id, def_id, relationship="DEPENDS_ON")

        for j, c2 in enumerate(claims):
            if i >= j:
                continue
            c2_id = c2["claim_id"]

            s1 = set(c1["source_lines"])
            s2 = set(c2["source_lines"])
            sec1 = _get_section(G, c1["source_lines"][0]) if c1["source_lines"] else None
            sec2 = _get_section(G, c2["source_lines"][0]) if c2["source_lines"] else None
            same_section = sec1 and sec1 == sec2

            # B. Shared source → SUPPORTS
            if s1 & s2:
                G.add_edge(c1_id, c2_id, relationship="SUPPORTS")
            elif same_section:
                G.add_edge(c1_id, c2_id, relationship="RELATED_TO")

            # C. Value overlap
            if len(str(c1["value"])) > 4 and str(c1["value"]) in str(c2["value"]):
                G.add_edge(c1_id, c2_id, relationship="LINKED_TO")

            # D. Contradiction
            if (c1["type"] == c2["type"]
                    and c1["type"] in ("entity_date", "entity_money", "entity_statistic")
                    and same_section and c1["value"] != c2["value"]):
                G.add_edge(c1_id, c2_id, relationship="CONTRADICTS")

    # ═══ 3. Section Evidence Summaries ═══
    _build_section_evidence_summaries(G, claims)

    # ═══ 4. Evidence Chains ═══
    _build_evidence_chains(G)

    return G


# ─── helpers ────────────────────────────────────────────────────────────────

def _get_section(G, line_id):
    if line_id not in G:
        return None
    # First try: look for explicit BELONGS_TO relationship
    for neighbor in G.successors(line_id):
        edge = G.edges.get((line_id, neighbor), {})
        if edge.get("relationship") == "BELONGS_TO" and G.nodes.get(neighbor, {}).get("type") == "section":
            return neighbor
    # Fallback: look for any section-type successor
    for neighbor in G.successors(line_id):
        if G.nodes.get(neighbor, {}).get("type") == "section":
            return neighbor
    # Also try section_id attribute (set during graph construction)
    section_id = G.nodes[line_id].get("section_id")
    if section_id and section_id in G:
        return section_id
    return None


def _build_section_evidence_summaries(G, claims):
    """
    Create per-section evidence summary nodes that aggregate all claims
    belonging to a section.  These act as fast evidence anchors during
    retrieval — instead of traversing every line, the retriever can
    hit the summary first.
    """
    section_claims = {}  # section_id -> [claim dicts]
    for c in claims:
        for src in c["source_lines"]:
            sec = _get_section(G, src)
            if sec:
                section_claims.setdefault(sec, []).append(c)

    for sec_id, sc_list in section_claims.items():
        summary_id = f"evidence_summary_{sec_id}"
        facts = []
        types_counter = {}
        for c in sc_list:
            facts.append(c["value"])
            types_counter[c["type"]] = types_counter.get(c["type"], 0) + 1

        G.add_node(summary_id,
                   type="evidence_summary",
                   section_id=sec_id,
                   claim_count=len(sc_list),
                   claim_types=types_counter,
                   facts_digest="; ".join(facts[:20]))
        G.add_edge(summary_id, sec_id, relationship="SUMMARISES_EVIDENCE_FOR")
        for c in sc_list:
            G.add_edge(summary_id, c["claim_id"], relationship="AGGREGATES")


def _build_evidence_chains(G):
    """
    Build EVIDENCE_CHAIN edges that connect text_components across
    sections when there is a multi-hop reasoning path through
    structural/semantic edges.  This enables the retriever to
    discover evidence that spans multiple sections.
    """
    # Gather high-value nodes (those with claims attached)
    nodes_with_claims = set()
    for n, d in G.nodes(data=True):
        if d.get("type") == "claim":
            for src in d.get("source_lines", []):
                nodes_with_claims.add(src)

    # For each pair of claim-bearing nodes in different sections,
    # check if a short path exists (≤ 4 hops) and create a direct chain edge.
    claim_nodes = list(nodes_with_claims)
    undirected = G.to_undirected(as_view=True)
    for i in range(len(claim_nodes)):
        sec_i = _get_section(G, claim_nodes[i])
        for j in range(i + 1, min(len(claim_nodes), i + 40)):
            sec_j = _get_section(G, claim_nodes[j])
            if sec_i == sec_j:
                continue
            try:
                path = nx.shortest_path(undirected, claim_nodes[i], claim_nodes[j])
                if 2 < len(path) <= 5:
                    G.add_edge(claim_nodes[i], claim_nodes[j],
                               relationship="EVIDENCE_CHAIN",
                               chain_length=len(path),
                               via=[str(p) for p in path[1:-1]])
            except nx.NetworkXNoPath:
                pass
