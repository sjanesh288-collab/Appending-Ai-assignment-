"""Runs the 6 assignment sample queries and saves sample_outputs.json (commit it as proof)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_graph import RAGPipeline  # noqa: E402

QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]

if __name__ == "__main__":
    pipeline = RAGPipeline()
    results = []
    for q in QUERIES:
        r = pipeline.run(q)
        print(json.dumps(r, indent=2, ensure_ascii=False), "\n")
        results.append(r)
    with open("sample_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved sample_outputs.json")
