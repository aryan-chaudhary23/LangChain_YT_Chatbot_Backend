from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)
CORS(app)

splitter = None
chunks = None
vector_store = None

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",                  
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),    
    task="feature-extraction"
)
llm_backend = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.2,
        max_new_tokens=1024,
        task="conversational"   # IMPORTANT
    )
llm = ChatHuggingFace(llm=llm_backend)
prompt = PromptTemplate(
    template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
        """,
    input_variables = ['context', 'question']
    )
vector_store_cache = set()

def extract_video_id(url):
    parsed_url = urlparse(url)

    # Case 1: https://www.youtube.com/watch?v=VIDEO_ID
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    # Case 2: https://youtu.be/VIDEO_ID
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    # Case 3: https://www.youtube.com/embed/VIDEO_ID
    if "embed" in parsed_url.path:
        return parsed_url.path.split("/")[-1]

    return None

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def final_work(url,question):
    video_id = extract_video_id(url)
    if not (video_id in vector_store_cache):
        vector_store_cache.clear() 
        vector_store_cache.add(video_id)
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id, languages=["en"])
            transcript = " ".join(chunk.text for chunk in transcript_list)

        except TranscriptsDisabled:
            print("No captions available for this video.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        # 1. Initialize FAISS with just the first chunk
        vector_store = FAISS.from_documents([chunks[0]], embeddings)

        # 2. Feed the remaining chunks in batches of 10
        batch_size = 10
        for i in range(1, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            # Send the small batch to Hugging Face API
            vector_store.add_documents(batch)
            
            # Pause for 1 second to respect free-tier rate limits
            time.sleep(1)
    global retriever, parallel_chain, parser, main_chain
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain.invoke(question)
    

#url = "https://www.youtube.com/watch?v=rL6uo5FRnKY"
@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    question = (data.get("question") or "").strip()
 
    if not url:
        return jsonify({"error": "YouTube URL is required."}), 400
    if not question:
        return jsonify({"error": "Question is required."}), 400
    answer=final_work(url,question)
    return jsonify({"answer": answer})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
