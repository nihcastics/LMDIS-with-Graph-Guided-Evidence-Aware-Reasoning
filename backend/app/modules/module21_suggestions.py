import re
from openai import OpenAI
from backend.app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS

def generate_questions(doc_text_summary, doc_meta=None):
    """Generates 3-5 relevant questions based on document summary."""
    if not OPENROUTER_API_KEY:
        return ["What is this document about?", "Key takeaways?", "Main conclusions?"]
    
    models_to_try = [OPENROUTER_MODEL] + OPENROUTER_FALLBACK_MODELS
    
    doc_info = f"Filename: {doc_meta.get('filename', 'Unknown')}\n"
    if doc_meta and 'detected_title' in doc_meta:
        doc_info += f"Title: {doc_meta['detected_title']}\n"

    prompt = f"""You are a Document Analyst. Based on the following document information and text snippets, generate 4 distinct, highly relevant questions that a user might want to ask about this specific document.
    
Document Context:
{doc_info}

Text Snippets:
{doc_text_summary}

Rules:
1. Generate exactly 4 questions.
2. Ensure questions are specific to the content (not generic like "What is this?").
3. Make them professional and analytical.
4. Output ONLY the questions, one per line, no numbering.

Questions:"""

    for model_name in models_to_try:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Document Intelligence System",
                }
            )
            
            content = response.choices[0].message.content.strip()
            # Strip tags and split by lines
            content = re.sub(r'<(think|thought|reasoning)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            questions = [q.strip() for q in content.split('\n') if q.strip() and '?' in q]
            if questions:
                return questions[:4]
        except Exception as e:
            print(f"Suggestion generation failed for {model_name}: {e}")
            continue
            
    return ["What is the primary objective of this document?", "What are the key results or findings?", "Who authored this document?", "What methodology was used?"]
