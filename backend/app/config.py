import os

# OpenRouter API Configuration
# Set OPENROUTER_API_KEY environment variable for production.
# Falls back to free-tier key for local development only.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_FALLBACK_MODELS = [
    "google/gemini-2.0-flash:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free"
]

# Strong LLM model for critical reasoning tasks (answer selection, semantic alignment)
# google/gemini-2.5-flash: best-in-class for long-document Q&A, 1M context, ~50x cheaper
# than claude-sonnet-4-6. Set OPENROUTER_STRONG_MODEL env var to override.
OPENROUTER_STRONG_MODEL = os.getenv("OPENROUTER_STRONG_MODEL", "google/gemini-2.5-flash")
OPENROUTER_STRONG_FALLBACK_MODELS = [
    "google/gemini-2.0-flash:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-r1",
    "google/gemini-2.0-flash:free",
]

# Google Gemini API Configuration (direct — bypasses OpenRouter for strong model calls)
# Gemini 2.5 Flash: best-in-class long-document Q&A, 1M context, native Google endpoint.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_STRONG_MODEL = os.getenv("GOOGLE_STRONG_MODEL", "gemini-2.5-flash")

# Anthropic Claude API Configuration
# Used ONLY for one-line crisp answer synthesis — budget-preserving (< $0.001 per query).
# claude-haiku-4-5-20251001: fastest + cheapest Claude model.
# claude-sonnet-4-6: balanced performance and cost.
# claude-opus-4-6: highest capability Claude model.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_CRISP_MODEL = os.getenv("ANTHROPIC_CRISP_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_BEST_MODEL = os.getenv("ANTHROPIC_BEST_MODEL", "claude-opus-4-6")

# System Configuration
SYSTEM_NAME = "Lossless Multimodal Document Intelligence System"
