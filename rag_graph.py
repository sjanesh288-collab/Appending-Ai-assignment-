"""LangGraph RAG pipeline with retrieval gate and hallucination grading.

Graph:
    START -> retrieve -> (relevant?) --no--> refuse -> END
                              |
                             yes
                              v
                          generate -> (refused?) --yes--> finalize -> END
                              ^             |
                              |            no
                              |             v
                              +--retry-- grade_groundedness --pass/exhausted--> finalize -> END
"""
from typing import Any, Dict, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from pydantic import BaseModel, Field

import config

NOT_FOUND = "I couldn't find this information in the Agentic AI eBook, so I can't answer that."


class RAGState(TypedDict, total=False):
    query: str
    chunks: List[dict]        # {"text", "page", "score"}
    top_score: float
    answer: str
    grounded_score: float
    feedback: str
    attempts: int
    confidence: float
    refused: bool


class GroundingGrade(BaseModel):
    score: float = Field(description="0.0-1.0: fraction of the answer's claims fully supported by the context")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims not supported by the context")


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions strictly and ONLY from the provided context taken from the Agentic AI eBook.\n"
     "Rules:\n"
     "1. Use no outside knowledge, even if you know the answer.\n"
     "2. If the context does not contain the answer, reply with exactly: " + NOT_FOUND + "\n"
     "3. Be concise and well structured; cite pages like (p. 12) where useful.\n"
     "{feedback}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Decide whether every claim in the ANSWER is supported by the CONTEXT. "
     "Ignore style. Score 1.0 if fully supported, 0.0 if not supported at all. "
     "List any unsupported claims."),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])


def _format_context(chunks: List[dict]) -> str:
    return "\n\n".join(f"[Chunk {i + 1} | page {c['page']}]\n{c['text']}" for i, c in enumerate(chunks))


class RAGPipeline:
    def __init__(self) -> None:
        config.require_keys()
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
        self.llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
        self.grader = self.llm.with_structured_output(GroundingGrade)
        self.index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
        self.graph = self._build_graph()

    # ---------- nodes ----------
    def retrieve(self, state: RAGState) -> RAGState:
        vec = self.embeddings.embed_query(state["query"])
        res = self.index.query(vector=vec, top_k=config.TOP_K, include_metadata=True)
        chunks = [
            {"text": m.metadata["text"], "page": int(m.metadata.get("page", 0)), "score": float(m.score)}
            for m in res.matches
        ]
        top = chunks[0]["score"] if chunks else 0.0
        return {"chunks": chunks, "top_score": top, "attempts": 0, "feedback": ""}

    def refuse(self, state: RAGState) -> RAGState:
        return {"answer": NOT_FOUND, "confidence": 0.0, "refused": True, "chunks": []}

    def generate(self, state: RAGState) -> RAGState:
        feedback = ""
        if state.get("feedback"):
            feedback = ("4. A previous draft contained unsupported claims. Remove or fix them: "
                        + state["feedback"] + "\n")
        msg = (GENERATE_PROMPT | self.llm).invoke({
            "context": _format_context(state["chunks"]),
            "question": state["query"],
            "feedback": feedback,
        })
        return {"answer": msg.content.strip(), "attempts": state.get("attempts", 0) + 1}

    def grade_groundedness(self, state: RAGState) -> RAGState:
        grade: GroundingGrade = (GRADE_PROMPT | self.grader).invoke({
            "context": _format_context(state["chunks"]),
            "answer": state["answer"],
        })
        score = max(0.0, min(1.0, grade.score))
        return {"grounded_score": score, "feedback": "; ".join(grade.unsupported_claims)}

    def finalize(self, state: RAGState) -> RAGState:
        if state.get("refused") or NOT_FOUND in state.get("answer", ""):
            return {"answer": NOT_FOUND, "confidence": 0.0, "refused": True, "chunks": []}
        g = state.get("grounded_score", 0.0)
        if g < config.FLOOR_GROUNDED_SCORE:  # still ungrounded after retries -> don't risk a hallucination
            return {"answer": NOT_FOUND, "confidence": 0.0, "refused": True, "chunks": []}
        retrieval = min(1.0, state.get("top_score", 0.0) / 0.6)  # cosine ~0.6 == very strong match
        return {"confidence": round(0.7 * g + 0.3 * retrieval, 2)}

    # ---------- routers ----------
    def route_after_retrieve(self, state: RAGState) -> str:
        if not state["chunks"] or state["top_score"] < config.MIN_RETRIEVAL_SCORE:
            return "refuse"
        return "generate"

    def route_after_generate(self, state: RAGState) -> str:
        return "finalize" if NOT_FOUND in state["answer"] else "grade"

    def route_after_grade(self, state: RAGState) -> str:
        if state["grounded_score"] >= config.MIN_GROUNDED_SCORE:
            return "finalize"
        if state["attempts"] < config.MAX_GENERATION_ATTEMPTS:
            return "generate"  # cycle: regenerate with grader feedback
        return "finalize"

    # ---------- graph ----------
    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self.retrieve)
        g.add_node("refuse", self.refuse)
        g.add_node("generate", self.generate)
        g.add_node("grade", self.grade_groundedness)
        g.add_node("finalize", self.finalize)

        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", self.route_after_retrieve, {"refuse": "refuse", "generate": "generate"})
        g.add_conditional_edges("generate", self.route_after_generate, {"finalize": "finalize", "grade": "grade"})
        g.add_conditional_edges("grade", self.route_after_grade, {"generate": "generate", "finalize": "finalize"})
        g.add_edge("refuse", END)
        g.add_edge("finalize", END)
        return g.compile()

    # ---------- public ----------
    def run(self, query: str) -> Dict[str, Any]:
        out = self.graph.invoke({"query": query})
        return {
            "query": query,
            "final_answer": out["answer"],
            "retrieved_context_chunks": [c["text"] for c in out.get("chunks", [])],
            "confidence_score": out.get("confidence", 0.0),
        }
