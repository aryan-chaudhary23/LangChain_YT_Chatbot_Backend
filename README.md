<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-🦜-1C3C3C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/FAISS-Vector_Store-0064A5?style=for-the-badge"/>
<img src="https://img.shields.io/badge/HuggingFace-🤗-FFD21E?style=for-the-badge&logoColor=black"/>
<img src="https://img.shields.io/badge/Flask-REST_API-000000?style=for-the-badge&logo=flask&logoColor=white"/>

<br/><br/>

# 🎬 YouTube RAG Application

### *Ask anything about any YouTube video — powered by LangChain, FAISS & Qwen2.5*

<br/>

> **Retrieval-Augmented Generation** pipeline that transforms YouTube transcripts into a searchable semantic index, then uses a state-of-the-art LLM to answer your questions — grounded in the actual video content.

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](/)

</div>

---

## 📸 Demo

```
User:   "What are the main arguments made in this video?"
AI:     "Based on the transcript, the speaker makes three core arguments:
         1. Current AI safety measures are insufficient...
         2. Open-source models present a dual-use risk...
         3. Regulation should focus on compute thresholds..."
```

> ⚡ First query on a new video: **~15–30s** (transcript fetch + embedding)  
> ⚡ Subsequent queries on same video: **~1–3s** (cache hit)

---

## 🧠 How It Works

```
┌─────────────┐     POST /api/ask      ┌─────────────────┐
│    React    │ ──────────────────────▶ │   Flask API     │
│  Frontend   │ ◀────────────────────── │   /api/ask      │
└─────────────┘     {"answer": "..."}   └────────┬────────┘
                                                 │
                        ┌────────────────────────┤
                        ▼                        ▼
               ┌─────────────────┐    ┌──────────────────┐
               │  Video ID       │    │  Cache Hit?      │
               │  Extractor      │    │  vector_store    │
               └────────┬────────┘    │  _cache[id]      │
                        │             └──────────────────┘
                        ▼  (cache miss)
               ┌─────────────────┐
               │  RapidAPI       │  ← YouTube Transcript v3
               │  Transcript     │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Text Splitter  │  ← chunk_size=1000, overlap=200
               │  (LangChain)    │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  BGE-small-en   │  ← HuggingFace Inference API
               │  Embeddings     │    (batches of 10)
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐     top-k=4 chunks
               │  FAISS Vector   │ ──────────────────────▶
               │  Store          │                        │
               └─────────────────┘                        ▼
                                               ┌──────────────────┐
                                               │  PromptTemplate  │
                                               │  + Qwen2.5-7B    │
                                               │  (via HF API)    │
                                               └──────────────────┘
```

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 **Semantic Search** | FAISS vector store finds the *most relevant* transcript chunks — not just keyword matches |
| 🧩 **Smart Chunking** | `RecursiveCharacterTextSplitter` with overlap ensures context is never lost at boundaries |
| 🚀 **In-Memory Cache** | Video indexes are cached per `video_id` — repeat questions are instant |
| 🛡️ **Hallucination Guard** | Prompt explicitly instructs the LLM: *"Answer ONLY from the provided transcript context"* |
| 🌐 **REST API** | Clean Flask API with CORS support — plug in any frontend |
| 💬 **Any YouTube Video** | Paste any YouTube URL — standard, shortened (`youtu.be`), or with timestamps |

---

## 🏗️ Tech Stack

<div align="center">

| Layer | Technology | Role |
|-------|-----------|------|
| **Frontend** | React | User interface |
| **Backend** | Flask + Flask-CORS | REST API server |
| **Transcripts** | RapidAPI (YouTube Transcript v3) | Fetch raw transcript |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Split transcript into chunks |
| **Embeddings** | `BAAI/bge-small-en-v1.5` via HuggingFace | Encode chunks to vectors |
| **Vector Store** | FAISS | Similarity search |
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace | Answer generation |
| **Chain** | LangChain LCEL (`RunnableParallel`) | Orchestrate the RAG pipeline |

</div>

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/youtube-rag-app.git
cd youtube-rag-app
```

### 2. Install dependencies

```bash
pip install flask flask-cors requests python-dotenv \
  langchain-text-splitters langchain-community \
  langchain-core langchain-huggingface faiss-cpu
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
RAPIDAPI_KEY=your_rapidapi_key_here
```

| Variable | Where to Get |
|----------|-------------|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — free account |
| `RAPIDAPI_KEY` | [rapidapi.com/dashboard](https://rapidapi.com/dashboard) — subscribe to *YouTube Transcript v3* |

### 4. Run the backend

```bash
python app.py
# Server starts at http://localhost:5000
```

### 5. Run the frontend

```bash
cd frontend
npm install && npm start
# React app starts at http://localhost:3000
```

---

## 🔌 API Reference

### `POST /api/ask`

Ask a question about a YouTube video.

**Request Body**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "question": "What is the main topic of this video?"
}
```

**Response**
```json
{
  "answer": "The video is about..."
}
```

**Error Response**
```json
{
  "error": "Could not fetch transcript for this video."
}
```

### `GET /api/health`

```json
{ "status": "ok" }
```

---

## 📊 Pipeline Stages

```
Stage 1: FETCH        GET transcript from RapidAPI → raw text string
Stage 2: CHUNK        Split into ~1000-char chunks (200-char overlap)
Stage 3: EMBED        BGE-small encodes each chunk → 384-dim float vector
Stage 4: INDEX        FAISS builds L2 similarity index over all vectors
Stage 5: RETRIEVE     User question encoded → top-4 nearest chunks returned
Stage 6: GENERATE     Chunks + question → PromptTemplate → Qwen2.5 → answer
```

---

## ⚡ Caching Behaviour

| Scenario | Result |
|----------|--------|
| First question on video A | Full pipeline runs (~15–30s). Index cached. |
| Second question on video A | Cache hit — skip to retrieval (~1–3s) |
| Question on video B | Cache miss — full pipeline for B; A stays cached |
| Server restart | Cache cleared — all videos re-indexed on next request |

> **Production tip:** Use `FAISS.save_local()` / `FAISS.load_local()` to persist indexes to disk and survive restarts.

---

## 🗂️ Project Structure

```
youtube-rag-app/
├── app.py                  # Flask API + RAG pipeline
├── .env                    # API keys (never commit this!)
├── .env.example            # Template for .env
├── requirements.txt        # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   └── ...
│   └── package.json
└── README.md
```

---

## 🔭 Roadmap & Improvements

- [ ] **Persistent vector store** — save FAISS indexes to disk between restarts
- [ ] **Async / streaming** — stream progress updates during first-time indexing
- [ ] **Conversation memory** — `ConversationBufferMemory` for follow-up questions
- [ ] **Retry logic** — `tenacity` exponential backoff for HF API timeouts
- [ ] **Cross-encoder reranking** — retrieve k=10, rerank with `bge-reranker-base`, keep top 4
- [ ] **Frontend loading indicator** — real-time status via WebSocket or SSE
- [ ] **Docker support** — containerised deployment
- [ ] **Multi-video comparison** — ask questions across several videos at once

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
# Fork the repo, then:
git checkout -b feature/my-improvement
git commit -m "feat: add my improvement"
git push origin feature/my-improvement
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with 🦜 LangChain · 🤗 HuggingFace · ⚡ FAISS · 🐍 Python

*If this helped you, consider giving it a ⭐*

</div>
