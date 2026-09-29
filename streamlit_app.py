"""Optional UI. Run: streamlit run streamlit_app.py"""
import streamlit as st

from rag_graph import RAGPipeline

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Answers come only from the eBook. Out-of-scope questions are refused.")


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


query = st.text_input("Ask a question", placeholder="What is Agentic AI?")
if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        result = load_pipeline().run(query.strip())
    st.subheader("Answer")
    st.write(result["final_answer"])
    st.metric("Confidence", result["confidence_score"])
    with st.expander("Retrieved context chunks"):
        for i, chunk in enumerate(result["retrieved_context_chunks"], 1):
            st.markdown(f"**Chunk {i}**")
            st.write(chunk)
    with st.expander("Raw JSON response"):
        st.json(result)
