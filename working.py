from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

from urllib.parse import urlparse, parse_qs

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

# Step 1a: Indexing YouTube video transcripts

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

url = "https://www.youtube.com/watch?v=rL6uo5FRnKY"

video_id = extract_video_id(url)
print(video_id)

try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    #print(transcript)
    #print(transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")

# Step 1b: Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
#print(len(chunks))
#print(chunks[10].page_content)

# Step 1c & 1d: Indexing(Embedding generation and Vector Store Creation) 
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}  # important
)
vector_store = FAISS.from_documents(chunks, embeddings)
#print(vector_store.index_to_docstore_id)

# Step 2: Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#print(retriever.invoke('What is deepmind?'))

# Step 3: Augmentation
from langchain_huggingface import HuggingFaceEndpoint

llm_backend = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=512,
    task="conversational"   # IMPORTANT
)
from langchain_huggingface import ChatHuggingFace

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
question          = "I want to understand everything in this video in simple terms."
retrieved_docs    = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#print(context_text)
final_prompt = prompt.format(context=context_text, question=question)

# Step 4: Generation
from langchain_core.messages import HumanMessage

response = llm.invoke([
    HumanMessage(content=final_prompt)
])

#print(response.content)

#Building a CHAIN
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke('Can you summarize the video'))