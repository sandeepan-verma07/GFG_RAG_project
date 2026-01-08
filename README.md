# 🧠 Multi-User RAG Chatbot (PDF + Web + Memory)

This project is a **production-style Retrieval Augmented Generation (RAG) chatbot** built with **Streamlit**, **Qdrant**, **Tavily**, **Mem0**, and **Gemma 3**.

It supports:
- PDF-based question answering
- Automatic web search fallback
- Persistent long-term user memory
- Multi-user isolation using tenant IDs
- Strict context-controlled LLM responses (no hallucination)

---

## 🚀 Key Features

- **RAG with PDFs + Web**
- **Chunk size:** 800  
- **Chunk overlap:** 200
- **Vector DB:** Qdrant (Cosine similarity)
- **Web Search:** Tavily
- **Similarity threshold:** 0.35
- **LLM:** Gemma-3-4B-IT
- **Long-term memory:** Mem0
- **Multi-user support:** Device ID / Custom ID
- **Frontend:** Streamlit

---

## 🧱 Tech Stack

- Python
- Streamlit
- Qdrant
- Tavily Search
- Mem0
- Google Gemma 3
- LangChain
- Sentence Transformers
- FastEmbed
- ChromaDB (offline ingest pipeline)

---

## 🧠 High-Level Architecture

User Query
↓
Embed Query
↓
Qdrant Vector Search (PDFs)
↓
Similarity Check (threshold = 0.35)
↓
┌───────────────┐
│ Score < 0.35? │── Yes ──► Tavily Web Search
└───────┬───────┘
│ No
↓
Merge PDF + Web Context
↓
Fetch User Memories (Mem0)
↓
Inject Recent Chat History
↓
Gemma 3 LLM
↓
Answer
↓
Save Memory to Mem0 
---

## 📁 Project Structure
.
├── main.py # Streamlit app
├── init_qdrant.py # Qdrant collection initialization
├── qdrant_operations.py # Qdrant CRUD + search
├── src/
│ ├── embeddings.py # Embedding manager
│ ├── loader.py # PDF loading & chunking
│ ├── llm_gemma.py # Gemma 3 LLM wrapper
│ ├── mem0_client.py # Mem0 memory handling
│ ├── rag_core.py # RAG orchestration logic
│ ├── retriever.py # Chroma retriever (offline)
│ └── vectore_store.py # Chroma vector store
├── ingest.py # Offline PDF ingest pipeline
├── requirements.txt
├── .env
└── data/ # PDFs for offline ingest

---

## 🧩 Detailed Workflow

### 1️⃣ User Identification (Multi-Tenant)

Each user is assigned:
- An auto-generated **Device ID**, or
- A manually entered **Custom ID (8 characters)**

This ID is used for:
- Qdrant filtering
- Mem0 memory scoping
- Chat history isolation

---

### 2️⃣ PDF Upload & Chunking

- PDFs are split using:
  - **Chunk size:** 800
  - **Overlap:** 200
- Metadata preserved:
  - filename
  - page number
  - chunk index

---

### 3️⃣ Embedding Generation

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output dimension: **384**
- Same model used for:
  - PDF chunks
  - User queries

---

### 4️⃣ Vector Storage

#### 🔹 Qdrant (Main App)
- Single collection
- Multi-tenant using `user_id`
- Payload-indexed
- Used for interactive chat

Each vector payload contains:
user_id, doc_id, filename, page, chunk_index, text

#### 🔹 ChromaDB (Offline Pipeline)
- Used by `ingest.py`
- Local persistent storage
- Converts cosine distance → similarity

---

### 5️⃣ Retrieval Logic (Hybrid RAG)

**Similarity Threshold:** `0.35`

| Mode | Behavior |
|----|----|
| PDF only | Qdrant only |
| Web only | Tavily only |
| Hybrid | Qdrant → Tavily fallback |

**Fallback Rule:**
- If no PDF results
- OR top similarity < 0.35  
→ Tavily Web Search is triggered

---

### 6️⃣ Tavily Web Search

- Top 3 results
- Filtered for:
  - Country relevance
  - Time relevance (current year logic)
- Used only when PDF context is weak

---

### 7️⃣ Memory System (Mem0)

#### Long-Term Memory
- Stores conversation messages
- Scoped per `user_id`
- Used **only for personal information**

#### Memory Retrieval Trigger
Memories are fetched only if query contains:
my, me, I, remember

---

### 8️⃣ RAG Core Orchestration

The RAG core:
1. Extracts readable text from:
   - Qdrant chunks
   - Tavily snippets
2. Fetches user memories from Mem0
3. Injects recent chat history
4. Calls the LLM

---

### 9️⃣ LLM – Gemma 3

**Model:** `gemma-3-4b-it`

#### Strict Context Priority
1. Recent chat history  
2. Long-term memories  
3. PDF context  
4. Web search results  

#### Safety Rules
- ❌ No hallucination
- ❌ No guessing personal info
- ❌ No external knowledge
- ✅ Web results override PDFs when relevant
- ✅ Time-sensitive questions rely on web only

---
