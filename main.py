import os
import uuid
import streamlit as st
from streamlit_local_storage import LocalStorage
from qdrant_client import QdrantClient
from qdrant_operations import upsert_chunks, list_user_docs, delete_document, search, ensure_collection, QdrantConfig
from dotenv import load_dotenv
from st_copy_to_clipboard import st_copy_to_clipboard
from langchain_community.tools.tavily_search import TavilySearchResults

from pipeline.chunk_pdf import chunk_pdf
from pipeline.embed_chunks import embed_texts, embed_query
from src.llm_gemma import GemmaLLM


load_dotenv()

cfg = QdrantConfig(
    url=os.environ.get("QDRANT_CONSOLE_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    collection_name="all_user_docs",
    vector_size=384,       #1536,                  # match your embedding model
    distance="Cosine"                  # or "Dot" / "Euclid"
)

tavily_tool = TavilySearchResults(api_key=os.environ.get("TAVILY_API_KEY"))

qdrant_client = QdrantClient(
    url=cfg.url,
    api_key=cfg.api_key,
)

ensure_collection(qdrant_client, cfg)

def generate_user_id():
    full_uuid = str(uuid.uuid4())
    return full_uuid[:4] + full_uuid[-4:]

def get_active_user_id():
    """Handles device ID persistence, custom ID input, and returns the active user ID at every toggle."""
    storage = LocalStorage()

    # Persistent device ID (never overwritten)
    device_id = storage.getItem("device_id")
    if not device_id:
        device_id = generate_user_id()
        storage.setItem("device_id", device_id)

    # Load custom ID if previously entered
    custom_id = storage.getItem("custom_id")

    # Dropdown selector
    id_choice = st.selectbox(
        "Select which ID to use:",
        ["Device ID", "Custom ID"],
        index=0 if not custom_id else 1
    )

    # Show device ID always
    st.write(f"Your Device ID (auto-generated): {device_id}")

    # Default active ID
    active_user_id = device_id

    # If user selects Custom ID, show input box
    if id_choice == "Custom ID":
        custom_id_input = st.text_input(
            "Enter your Custom ID (8 characters):",
            custom_id or ""
        )
        if custom_id_input and len(custom_id_input.strip()) == 8:
            storage.setItem("custom_id", custom_id_input.strip())
            custom_id = custom_id_input.strip()
            active_user_id = custom_id
        elif custom_id_input:
            st.warning("Custom ID must be exactly 8 characters. Not saved.")
            active_user_id = device_id      # fallback

    st.success(f"Active User ID: {active_user_id}")

    st_copy_to_clipboard(active_user_id, "Copy Active ID")

    return active_user_id


######################################################################################################## MAIN APP
st.title("RAG Chatbot")

active_user_id = get_active_user_id()

def process_pdf(uploaded_file, chunk_size=800, chunk_overlap=200):
    chunks = chunk_pdf(uploaded_file, chunk_size, chunk_overlap)
    texts = [c["text"] for c in chunks]   # flatten for embedding
    vectors = embed_texts(texts)
    return chunks, vectors


uploaded_files = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        chunks, vectors = process_pdf(uploaded_file)

        upsert_chunks(qdrant_client, cfg, active_user_id, filename, vectors, chunks)

        st.success(f"{filename} uploaded and stored!")



docs = list_user_docs(qdrant_client, cfg, active_user_id)
for doc_id, fname in docs:
    col1, col2 = st.columns([4,1])
    with col1:
        st.write(f"📄 {fname}")
    with col2:
        if st.button(f"❌", key=f"del_{doc_id}"):
            delete_document(qdrant_client, cfg, active_user_id, fname)
            st.warning(f"{fname} deleted")




SCORE_THRESHOLD = 0.4   # tune this threshold

query = st.text_input("Ask a question:")
if query:
    query_vector = embed_query(query)
    doc_options = ["All PDFs"] + [fname for _, fname in docs]
    selected_doc = st.selectbox("Search scope:", doc_options)


    context = search(
        qdrant_client, cfg, active_user_id, query_vector,
        filename=None if selected_doc == "All PDFs" else selected_doc
    )

    # Check score threshold
    use_web = not context or context[0]["score"] < SCORE_THRESHOLD   # As context is returned by decreasing similarity score, we just check the first context to confirm whether to infer the web search

    web_context = []
    if use_web:
        st.info("Low similarity score — fetching Tavily results...")
        web_results = tavily_tool.run(query)   # returns list of snippets
        web_context = [
            {"doc_id": "web", "text": snippet, "score": 1.0}
            for snippet in web_results
        ]

    # merge contexts
    final_context = context + web_context

    # Generating answer with combined context
    answer = GemmaLLM().generate(query, final_context)

    # Display Qdrant context
    if context:
        st.write("📄 PDF context:")
        for c in context[:3]:   # show top 3 chunks, but in search operation for Qdrant we are using 5 as limit
            st.write(f"{c['doc_id']} (score={c['score']:.3f})")
            st.write(f"- {c['text']}")

    # Display Tavily context
    if web_context:
        st.write("🌐 Web context:")
        for w in web_context[:3]:
            st.write(f"- {w['text']}")

    st.write(f"\n\nLLM : \n{answer}")