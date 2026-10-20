import time
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context
from router import SemanticRouter
from agents import execute_slm_agent

load_dotenv()
router_instance = SemanticRouter()


@observe(name="Qdrant_Semantic_Router")
def traced_route(query: str):
    return router_instance.route(query)


@observe(name="Specialized_SLM_Execution", as_type="generation")
def traced_agent_call(agent_name: str, query: str):
    result = execute_slm_agent(agent_name, query)
    # Log model metadata to Langfuse
    langfuse_context.update_current_observation(
        model=result["model_used"],
        metadata={"agent_type": agent_name},
    )
    return result


@observe(name="Multi_Agent_Workflow")
def process_user_query(query: str) -> dict:
    total_start = time.perf_counter()

    # 1. Semantic Routing via Qdrant
    route_info = traced_route(query)
    selected_agent = route_info["selected_agent"]

    # 2. Route to Specialized SLM
    agent_output = traced_agent_call(selected_agent, query)

    total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)

    return {
        "query": query,
        "routed_to": selected_agent,
        "confidence": route_info["confidence_score"],
        "routing_latency_ms": route_info["routing_latency_ms"],
        "total_latency_ms": total_latency_ms,
        "model_used": agent_output["model_used"],
        "response": agent_output["answer"],
        "retrieval_context": agent_output["context_used"],
    }


if __name__ == "__main__":
    sample_prompts = [
        "Write a Python function for Kadane's algorithm.",
        "What is the refund policy and enterprise rate limit?",
    ]
    for prompt in sample_prompts:
        print("=" * 60)
        res = process_user_query(prompt)
        print(f"Query : {res['query']}")
        print(f"Routed Agent : {res['routed_to']} (Confidence: {res['confidence']})")
        print(f"SLM Model : {res['model_used']}")
        print(f"Routing Latency : {res['routing_latency_ms']} ms")
        print(f"Total Latency : {res['total_latency_ms']} ms")
        print(f"Response Snippet : {res['response'][:180]}...\n")

    # Flush traces to Langfuse Cloud
    langfuse_context.flush()
