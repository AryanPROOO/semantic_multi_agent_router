import ollama

# Ground truth context for Factual/RAG Agent (Used in DeepEval testing too)
KNOWLEDGE_BASE_CONTEXT = """
Enterprise Platform Policy & Specs (2026):
1. Rate Limits: Free tier allows 60 requests/minute. Enterprise tier allows 5,000 requests/minute.
2. SLA Guarantee: Enterprise tier guarantees 99.95% uptime with automated failover.
3. Refund Policy: Full refunds are processed within 14 days of billing if total token usage is below 100,000 tokens.
"""

AGENT_CONFIGS = {
    "coding_agent": {
        "model": "qwen2.5-coder:3b",
        "system_prompt": "You are an expert software engineer. Provide clean, optimal code with brief time/space complexity.",
    },
    "math_agent": {
        "model": "llama3.2:3b",
        "system_prompt": "You are a mathematical reasoning engine. Solve problems step-by-step concisely.",
    },
    "factual_rag_agent": {
        "model": "llama3.2:3b",
        "system_prompt": (
            f"You are a strict factual assistant. Answer ONLY using the context below. "
            f"Do not assume or extrapolate anything outside this context.\n\nContext:\n{KNOWLEDGE_BASE_CONTEXT}"
        ),
    },
    "general_fallback_agent": {
        "model": "llama3.2:3b",
        "system_prompt": "You are a helpful, concise AI assistant.",
    },
}


def execute_slm_agent(agent_name: str, query: str) -> dict:
    config = AGENT_CONFIGS.get(agent_name, AGENT_CONFIGS["general_fallback_agent"])
    response = ollama.chat(
        model=config["model"],
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": query},
        ],
        options={"temperature": 0.1},  # Low temperature for high factual consistency
    )
    return {
        "model_used": config["model"],
        "answer": response["message"]["content"],
        "context_used": (
            [KNOWLEDGE_BASE_CONTEXT.strip()] if agent_name == "factual_rag_agent" else []
        ),
    }
