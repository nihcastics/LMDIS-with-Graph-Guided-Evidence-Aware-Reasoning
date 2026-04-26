import re

def extract_claims(G):
    """
    Module 10: Claim & Entity Extraction
    (Simplified Rule-Based for demo, would be LLM-based in prod)
    """
    claims = []
    claim_counter = 0
    
    for node, data in G.nodes(data=True):
        if data.get("type") == "text_component":
            text = data.get("text", "")
            
            # --- Generic Entity & Fact Extraction Patterns ---

            # 1. Dates (Expanded)
            dates = re.findall(r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})', text)
            for date in dates:
                claim_counter += 1
                claims.append({
                    "claim_id": f"c{claim_counter}",
                    "type": "entity_date",
                    "value": date,
                    "source_lines": [node],
                    "confidence": 1.0
                })

            # 2. Monetary Amounts
            amounts = re.findall(r'(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP|INR|RS|€|£))', text, re.I)
            for amount in amounts:
                claim_counter += 1
                claims.append({
                    "claim_id": f"c{claim_counter}",
                    "type": "entity_money",
                    "value": amount,
                    "source_lines": [node],
                    "confidence": 0.95
                })

            # 3. Percentages / Statistics
            percents = re.findall(r'(\d+(?:\.\d+)?\s?%)', text)
            for pct in percents:
                 claim_counter += 1
                 claims.append({
                    "claim_id": f"c{claim_counter}",
                    "type": "entity_statistic",
                    "value": pct,
                    "source_lines": [node],
                    "confidence": 0.9
                })

            # 4. Durations / Time Periods (Generic)
            # e.g., "30 days", "12 months", "5 years"
            durations = re.findall(r'(\d+\s+(?:day|week|month|year|hr|hour|min|minute)s?)', text, re.I)
            for dur in durations:
                claim_counter += 1
                claims.append({
                    "claim_id": f"c{claim_counter}",
                    "type": "entity_duration",
                    "value": dur,
                    "source_lines": [node],
                    "confidence": 0.85
                })

            # 5. Explicit Definitions (X is defined as Y, "Term" means...)
            def_match = re.search(r'["\']?([\w\s]+)["\']?\s+(?:means|is defined as|refers to)\s+([\w\s,]+)', text, re.I)
            if def_match:
                claim_counter += 1
                claims.append({
                    "claim_id": f"c{claim_counter}",
                    "type": "fact_definition",
                    "value": f"{def_match.group(1).strip()} = {def_match.group(2).strip()}",
                    "source_lines": [node],
                    "confidence": 0.8
                })

            # 6. Key-Value Specifications (Label: Value) where Value is short
            # Avoids matching strictly narrative sentences like "Note: The system..."
            kv_match = re.search(r'^([\w\s]+):\s+([^.]+?)$', text)
            if kv_match and len(kv_match.group(2)) < 50:
                 claim_counter += 1
                 claims.append({
                    "claim_id": f"c{claim_counter}",
                    "type": "fact_specification",
                    "value": f"{kv_match.group(1).strip()}: {kv_match.group(2).strip()}",
                    "source_lines": [node],
                    "confidence": 0.75
                })
                
            # 7. Emails (Entity)
            emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            for email in emails:
                claim_counter += 1
                claims.append({
                    "claim_id": f"c{claim_counter}",
                    "type": "entity_contact",
                    "value": email,
                    "source_lines": [node],
                    "confidence": 1.0
                })

    return claims
