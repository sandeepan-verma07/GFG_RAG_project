import os
import uuid
import streamlit as st
from streamlit_local_storage import LocalStorage
from qdrant_client import QdrantClient
from qdrant_operations import upsert_chunks, list_user_docs, delete_document, search, ensure_collection, QdrantConfig
from dotenv import load_dotenv
from st_copy_to_clipboard import st_copy_to_clipboard
from langchain_community.tools.tavily_search import TavilySearchResults
from src.rag_core import rag_answer
import re
from datetime import datetime


from pipeline.chunk_pdf import chunk_pdf
from src.embeddings import EmbeddingManager
from src.llm_gemma import GemmaLLM
from src.mem0_client import add_user_memories

embedder = EmbeddingManager()



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

# initiate chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []   # list of {"role": "user"/"assistant", "content": str}

active_user_id = get_active_user_id()


def process_pdf(uploaded_file, chunk_size=800, chunk_overlap=200):
    chunks = chunk_pdf(uploaded_file, chunk_size, chunk_overlap)
    texts = [c["text"] for c in chunks]   # flatten for embedding
    vectors = embedder.embed_texts(texts)

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




SCORE_THRESHOLD = 0.35  # threshold  changed 
mode = st.radio(
    "Choose retrieval mode:",
    ["Hybrid (PDF + Web)", "PDF only", "Web only"],
    index=0
)

# existing messages in chat style render
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        

#input for chat
user_query = st.chat_input("Ask a question:")


if user_query:
    # Append user message to history and render
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Embedding for retrieval
    query_vector = embedder.embed_texts([user_query])[0]

    doc_options = ["All PDFs"] + [fname for _, fname in docs]
    selected_doc = st.selectbox("Search scope:", doc_options)

    # always start with empty context
    context = []


    if mode != "Web only":
        context = search(
            qdrant_client, cfg, active_user_id, query_vector,
            filename=None if selected_doc == "All PDFs" else selected_doc
        )

    # Check score threshold
    use_web = False

    if mode == "PDF only":
        use_web = False
    elif mode == "Web only":
        context = []
        use_web = True
    else:
        if not context:
            use_web = True
        elif context[0]["score"] < SCORE_THRESHOLD:
            use_web = True

    web_context = []
    if use_web:
        st.info("Low similarity score — fetching Tavily results...")
        web_results = tavily_tool.run(user_query)[:3]

        web_context = [
    {
        "doc_id": "web",
        "text": snippet.get("content", ""),
        "score": 1.0
    }
    for snippet in web_results
]


   
    

    # merge contexts
    final_context = context + web_context
    if mode == "PDF only" and not context:
        st.warning("No relevant content found in your PDFs.")

    # answer with combined context + Mem0 user memory
    recent_msgs_for_context = st.session_state["messages"]    #[-8:]  # Last 4 turns

    answer = rag_answer(user_query, final_context, active_user_id,recent_messages=recent_msgs_for_context)

    # Display qdrant context
    if context:
        with st.expander("📄 PDF context", expanded=False):
            for c in context[:3]:
                st.write(f"{c['doc_id']} (score={c['score']:.3f})")
                st.write(f"- {c['text']}")

    # Display Tavily context
    if web_context:
        with st.expander("🌐 Web context", expanded=False):
            for w in web_context[:3]:
                st.write(f"- {w['text']}")

    # Append assistant answer to history and render
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Send recent messages to Mem0 as memory
    recent_msgs = st.session_state["messages"][-6:]  # last few turns
    add_user_memories(active_user_id, recent_msgs)
