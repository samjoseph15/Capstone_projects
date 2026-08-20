import streamlit as st
import pickle

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖")

st.title("AI Knowledge Assistant")
st.write("Ask questions from the AI knowledge base.")

with open("rag_model.pkl", "rb") as f:
    config = pickle.load(f)

embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])

vector_db = FAISS.load_local(
    config["vector_db_path"],
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(model=config["llm_model"], temperature=0)
top_k = config["top_k"]


query = st.text_input(
    "🔎 Ask a question",
    placeholder="Example: What is machine learning?"
)

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            # Search in FAISS
            results = vector_db.similarity_search(query, k=top_k)

            # Build context
            context = "\n\n".join(doc.page_content for doc in results)

            # Prompt
            prompt = f"""
Answer ONLY from the given context.
If not found, say: "I couldn't find that information in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

            
            response = llm.invoke(prompt)

        
        st.subheader("💡 Answer")
        st.write(response.content)

        
        st.subheader("📚 Sources")
        for doc in results:
            st.write("•", doc.metadata.get("source", "Unknown"))

st.divider()
st.caption("AI Knowledge Assistant | RAG + FAISS + Hugging Face + Ollama")
