# """
# Streamlit app to test the CAG (Cache Augmented Generation) system. Using the local DeepSeek-R1:1.5B model, this system generates responses to user queries and caches them for future use. It leverages SQLite for exact match caching and ChromaDB for semantic search. The Ollama Embeddings model is used to encode queries and responses for similarity comparison.
# """

import os
import sqlite3
import hashlib
import json
import ollama  # type: ignore
import chromadb  # type: ignore
import re
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
SIMILARITY_THRESHOLD = 0.70  # Lowered to avoid incorrect cache hits
CACHE_DB = "cache.db"
CHROMA_DB_PATH = "./chroma_db"

# Initialize SQLite for Exact Match Caching
conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS cache (
        id TEXT PRIMARY KEY,
        prompt TEXT UNIQUE,
        response TEXT
    )"""
)
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt ON cache(prompt)")
conn.commit()

def normalize_prompt(prompt):
    """Lowercases, removes special characters, and trims spaces."""
    prompt = prompt.lower().strip()
    prompt = re.sub(r'[^\w\s]', '', prompt)  # Remove punctuation
    return prompt

def get_cache_key(prompt):
    """Generate hash key for caching based on normalized prompt."""
    normalized_prompt = normalize_prompt(prompt)
    return hashlib.sha256(normalized_prompt.encode()).hexdigest()

def check_exact_cache(prompt):
    """Check for exact match in SQLite."""
    key = get_cache_key(prompt)
    cursor.execute("SELECT response FROM cache WHERE id=?", (key,))
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None

def update_exact_cache(prompt, response):
    """Update SQLite cache."""
    key = get_cache_key(prompt)
    cursor.execute("INSERT OR REPLACE INTO cache (id, prompt, response) VALUES (?, ?, ?)", 
                   (key, prompt, json.dumps(response)))
    conn.commit()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="cached_responses")

# Set Up Ollama Embeddings
def embed_text_with_ollama(text):
    """Generate embeddings using an Ollama model."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def store_response_embedding(prompt, response):
    """Store new response in ChromaDB with embeddings, filtering invalid ones."""
    embedding = embed_text_with_ollama(prompt)
    if "hello! how can i assist you today?" not in response.lower():  # Avoid storing generic responses
        collection.add(
            embeddings=[embedding],
            documents=[response],
            metadatas=[{"prompt": prompt}],
            ids=[get_cache_key(prompt)]
        )
    else:
        print("⚠️ Generic response detected. Not storing in ChromaDB.")

def retrieve_similar_response(prompt):
    """Find the most relevant cached response using embeddings, ensuring scaled distances."""
    query_embedding = embed_text_with_ollama(prompt)
    all_documents = collection.get()["documents"]
    n_results = min(3, len(all_documents)) if all_documents else 1  # Ensure n_results is at least 1
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"]
    )
    
    print(f"🔍 Querying ChromaDB for similar responses to: {prompt}")
    print(f"🔍 Found results: {results}")
    
    if not results["documents"] or not results["documents"][0]:
        return None  # No matches found
    best_match = results["documents"][0]
    similarity_score = results["distances"][0]
    
    print(f"🔍 Best Match: {best_match} with Similarity Score: {similarity_score}")
    
    if similarity_score and similarity_score[0] < 10.0:  # Adjusted scaling to avoid huge distances
        return best_match[0]  # ✅ Return best match if similarity score is reasonable
    return None  # No sufficiently similar match found

def generate_response(prompt):
    """Fetch response from cache or generate using DeepSeek."""
    print(f"🔎 Checking exact cache for: {prompt}")
    cached_response = check_exact_cache(prompt)
    if cached_response:
        print("✅ Cache HIT (Exact Match) - Returning from SQLite")
        return cached_response  # Cache HIT (Exact Match)
    
    print(f"🔍 Checking semantic cache for: {prompt}")
    similar_response = retrieve_similar_response(prompt)
    if similar_response:
        print("✅ Cache HIT (Semantic Match) - Returning from ChromaDB")
        return similar_response  # Cache HIT (Semantic Match)
    
    print("❌ Cache MISS - Generating new response from DeepSeek-R1:1.5B")
    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw_text = response["message"]["content"].strip()
        
        print(f"🤖 Generated Response: {raw_text}")
        
        update_exact_cache(prompt, raw_text)
        store_response_embedding(prompt, raw_text)
        return raw_text
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "I'm sorry, but I couldn't process that request."

# Streamlit UI
st.set_page_config(page_title="Cache-Augmented AI", page_icon="🤖", layout="centered")
st.title("💬 Cache-Augmented AI Assistant")
st.write("Ask me anything and I will generate a response!")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

prompt = st.text_input("Enter your query:", placeholder="Type your question here...")

if st.button("Generate Response"):
    if prompt:
        response = generate_response(prompt)
        st.session_state.conversation.insert(0, {"question": prompt, "answer": response})  # Latest message on top
        st.success("Response Generated!")
    else:
        st.warning("Please enter a query before clicking the button.")

st.markdown("---")
st.subheader("📜 Conversation History")
for chat in st.session_state.conversation:
    st.markdown(f"**🗨️ Question:** {chat['question']}")
    st.markdown(f"**🤖 Answer:** {chat['answer']}")
    st.markdown("---")

st.caption("🚀 Built with DeepSeek-R1, ChromaDB, and Streamlit")
