import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


class SemanticRouter:
    def __init__(self):
        # Local in-memory Qdrant instance (No Docker required)
        self.client = QdrantClient(":memory:")
        self.collection_name = "intent_router"
        # Fast lightweight embedding model (384 dimensions)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._setup_collection()

    def _setup_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Sample Intents for Routing
        routes = {
            "coding_agent": [
                "Write a Python function to implement binary search",
                "Fix this segmentation fault in my C++ linked list code",
                "Optimize this SQL query and explain time complexity",
                "How do I implement Kadane's algorithm in Python?",
                "Debug this asynchronous JavaScript function",
            ],
            "math_agent": [
                "Solve this system of linear equations step by step",
                "Calculate the probability of drawing two aces from a deck",
                "What is the derivative of sin(x) * cos(x)?",
                "Differentiate this mathematical function with respect to x",
                "Explain Bayes theorem with a numerical example",
                "Find the eigenvalues of a 2x2 matrix",
            ],
            "factual_rag_agent": [
                "What is our company refund and cancellation policy?",
                "Explain the architecture specification of our cloud platform",
                "According to the documentation, what is the rate limit?",
                "Summarize the key compliance rules from the knowledge base",
                "How many days of SLA uptime are guaranteed in the enterprise tier?",
            ],
        }

        points = []
        point_id = 1
        for agent_name, samples in routes.items():
            embeddings = self.encoder.encode(samples)
            for sample, vector in zip(samples, embeddings):
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={"agent": agent_name, "sample_query": sample},
                    )
                )
                point_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)

    def route(self, query: str, threshold: float = 0.45):
        start_time = time.perf_counter()
        query_vector = self.encoder.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1,
        ).points
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if results and results[0].score >= threshold:
            matched_agent = results[0].payload["agent"]
            score = round(results[0].score, 3)
        else:
            matched_agent = "general_fallback_agent"
            score = round(results[0].score, 3) if results else 0.0

        return {
            "selected_agent": matched_agent,
            "confidence_score": score,
            "routing_latency_ms": latency_ms,
        }


if __name__ == "__main__":
    router = SemanticRouter()
    test_queries = [
        "Write C++ code to reverse a linked list",
        "What is the SLA uptime guarantee for enterprise tier?",
        "Differentiate x^3 + 4x with respect to x",
    ]
    for q in test_queries:
        print(f"Query: {q}\nResult: {router.route(q)}\n")
