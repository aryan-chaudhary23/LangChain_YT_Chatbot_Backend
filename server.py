import os
import time
import requests # NEW: Added the requests library for RapidAPI
from urllib.parse import urlparse, parse_qs
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# REMOVED: youtube_transcript_api imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings

load_dotenv()

app = Flask(__name__)
CORS(app)

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",                  
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),    
    task="feature-extraction"
)

llm_backend = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.2,
    max_new_tokens=1024,
    task="conversational"   
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

vector_store_cache = {}

from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""

    if "youtube.com" in hostname:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]

        if parsed_url.path.startswith("/embed/") or parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]

    if "youtu.be" in hostname:
        return parsed_url.path[1:]

    return None

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def final_work(url, question):
    video_id = extract_video_id(url)
    
    if not video_id:
        return "Invalid YouTube URL."

    if video_id not in vector_store_cache:
        # --- NEW RAPIDAPI FETCH LOGIC ---
        try:
            api_url = "https://youtube-transcript3.p.rapidapi.com/api/transcript"
            querystring = {"videoId": video_id}
            headers = {
                "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
                "x-rapidapi-host": "youtube-transcript3.p.rapidapi.com"
            }

            response = requests.get(api_url, headers=headers, params=querystring)
            
            if response.status_code != 200:
                return f"API Error: Could not fetch transcript. Status {response.status_code}. Make sure your RapidAPI key is valid."
                
            data = response.json()

            # Safely extract text from the API's JSON response
            if isinstance(data, list):
                transcript = " ".join([item.get('text', '') for item in data if 'text' in item])
            elif isinstance(data, dict) and 'transcript' in data:
                transcript = " ".join([item.get('text', '') for item in data['transcript'] if 'text' in item])
            else:
                transcript = str(data)

            if not transcript or len(transcript) < 10:
                return "No captions available for this video."

        except Exception as e:
             return f"RapidAPI Connection Error: {str(e)}"
        # --- END RAPIDAPI LOGIC ---
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        if not chunks:
            return "Transcript is empty."

        vector_store = FAISS.from_documents([chunks[0]], embeddings)

        batch_size = 10
        for i in range(1, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vector_store.add_documents(batch)
            time.sleep(1)
            
        vector_store_cache[video_id] = vector_store

    vector_store = vector_store_cache[video_id]

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain.invoke(question)
    

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        url = (data.get("url") or "").strip()
        question = (data.get("question") or "").strip()
     
        if not url:
            return jsonify({"error": "YouTube URL is required."}), 400
        if not question:
            return jsonify({"error": "Question is required."}), 400
            
        answer = final_work(url, question)
        return jsonify({"answer": answer})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Backend Crash: {str(e)}"}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
