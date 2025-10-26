from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import faiss
import numpy as np
import google.generativeai as genai
from collections import deque
import json
import os
import time
import pickle
from contextlib import asynccontextmanager


# Multiple API keys from different projects (add your keys here)
API_KEYS = [
    "AIzaSyBdXQ_t03Xz-_mXY9IRTNbBdRiEChRk7jU",  # Your original key
    # Add more API keys from different Google Cloud projects
    # "AIzaSyD...",  # Project 2
    # "AIzaSyE...",  # Project 3
]

current_api_key_index = 0

def configure_next_api_key():
    global current_api_key_index
    if current_api_key_index < len(API_KEYS):
        genai.configure(api_key=API_KEYS[current_api_key_index])
        print(f"Using API key {current_api_key_index + 1}/{len(API_KEYS)}")
        return True
    return False

# Configure initial API key
configure_next_api_key()
llm = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")


# MongoDB setup
mongo_uri = "mongodb+srv://mokhithraa2004:moks@cluster0.p5j2bi5.mongodb.net/?appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["knowledge_base"]
collections_to_load = ["Building_Rules", "Amendment_69", "Amendment_0"]


class Query(BaseModel):
    question: str


chunk_map = {}
index = None


# History management
HISTORY_FILE = "history.json"
MAX_HISTORY = 3
history = deque(maxlen=MAX_HISTORY)

# Caching system
EMBEDDINGS_CACHE_FILE = "embeddings_cache.pkl"
CHUNK_LIMIT = 50  # Reduced from 567 to avoid quota issues


def load_history():
    global history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                saved = json.load(f)
                history = deque(saved, maxlen=MAX_HISTORY)
            print("✅ History loaded from file.")
        else:
            print("ℹ️ No history file found, starting fresh.")
    except Exception as e:
        print(f"⚠️ Error loading history file: {e}. Starting fresh.")
        history = deque(maxlen=MAX_HISTORY)


def save_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(list(history), f, indent=4)
        print("✅ History saved.")
    except Exception as e:
        print(f"⚠️ Error saving history: {e}")


def get_embedding_with_retry(text, task_type="retrieval_document", max_retries=3):
    """Get embedding with multiple API key fallback and retry logic"""
    global current_api_key_index
    
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type
            )
            return result["embedding"]
            
        except Exception as e:
            error_str = str(e)
            if "ResourceExhausted" in error_str or "429" in error_str:
                print(f"⚠️ Quota exhausted for API key {current_api_key_index + 1}")
                current_api_key_index += 1
                
                if configure_next_api_key():
                    print(f"🔄 Switching to next API key...")
                    continue
                else:
                    print("❌ All API keys exhausted. Using zero vector.")
                    return [0.0] * 768  # Return zero vector as fallback
            else:
                print(f"⚠️ Embedding error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return [0.0] * 768  # Return zero vector as fallback
    
    return [0.0] * 768


def load_cached_embeddings():
    """Load embeddings from cache file"""
    try:
        if os.path.exists(EMBEDDINGS_CACHE_FILE):
            print("📂 Loading cached embeddings...")
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['chunk_map'], cached_data['vectors']
        return None, None
    except Exception as e:
        print(f"⚠️ Failed to load embedding cache: {e}")
        return None, None


def save_embeddings_cache(chunk_map, vectors):
    """Save embeddings to cache file"""
    try:
        cache_data = {
            'chunk_map': chunk_map,
            'vectors': vectors
        }
        with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        print("💾 Embeddings cached for future use")
    except Exception as e:
        print(f"⚠️ Failed to save embedding cache: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chunk_map, index
    load_history()

    # Try to load cached embeddings first
    cached_chunk_map, cached_vectors = load_cached_embeddings()
    
    if cached_chunk_map is not None and cached_vectors is not None:
        chunk_map = cached_chunk_map
        index = faiss.IndexFlatL2(cached_vectors.shape[1])
        index.add(cached_vectors)
        print(f"✅ Loaded {len(chunk_map)} cached embeddings")
    else:
        print("🔄 Loading chunks from MongoDB collections (LIMITED to avoid quota issues)...")
        chunks = []

        for name in collections_to_load:
            try:
                current_chunks = list(db[name].find({}).limit(CHUNK_LIMIT))
                print(f"✅ Loaded {len(current_chunks)} chunks from '{name}' (limited to {CHUNK_LIMIT})")
                chunks.extend(current_chunks)
            except Exception as e:
                print(f"⚠️ Failed to load from '{name}': {e}")

        # Sort chunks by chunk_number if it exists
        sorted_chunks = sorted(chunks, key=lambda x: x.get("chunk_number", 0))
        texts = [chunk["content"] for chunk in sorted_chunks]
        chunk_map = {i: text for i, text in enumerate(texts)}

        print(f"🔎 Embedding {len(texts)} chunks using Gemini (with rate limiting)...")
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Add rate limiting: wait between requests
                if i > 0 and i % 5 == 0:  # Wait every 5 requests
                    print(f"⏳ Processed {i}/{len(texts)}, pausing to avoid rate limits...")
                    time.sleep(15)  # Wait 15 seconds every 5 requests
                
                embedding = get_embedding_with_retry(text)
                embeddings.append(embedding)
                print(f"✅ Embedded chunk {i + 1}/{len(texts)}")
                
            except Exception as e:
                print(f"⚠️ Error embedding chunk {i}: {e}")
                embeddings.append([0.0] * 768)  # Fallback zero vector

        vectors = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        
        # Save to cache for future use
        save_embeddings_cache(chunk_map, vectors)
        
        print(f"✅ Indexed {len(texts)} chunks with rate limiting")
    
    yield  # Application runs here
    
    # Shutdown code
    print("🔄 Application shutting down...")


# FastAPI setup with lifespan
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def ask_question(query: Query):
    try:
        user_embedding = get_embedding_with_retry(query.question, task_type="retrieval_query")
        user_vector = np.array([user_embedding], dtype="float32")

        D, I = index.search(user_vector, k=3)
        context = "\n".join(chunk_map[i] for i in I[0] if i < len(chunk_map))

        # Build history context
        history_context = ""
        for h in list(history):
            history_context += f"Previous Question: {h['question']}\nPrevious Answer: {h['answer']}\n\n"

        # Complete prompt including history
        prompt = f"""
You are a Real Estate expert chatbot.

Use the following retrieved document context to answer user's question:

Retrieved Context:
{context}

Recent Conversation History:
{history_context}

Now answer the user's current question based on both retrieved context and recent history.

Current Question:
{query.question}

Answer:
"""

        try:
            response = llm.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            if "ResourceExhausted" in str(e) or "429" in str(e):
                answer = "I'm currently experiencing high demand. Please try again in a few minutes. In the meantime, I can tell you that I'm a Real Estate expert designed to help with building rules and regulations."
            else:
                answer = f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"

        # Update history
        history.appendleft({"question": query.question, "answer": answer})
        save_history()

        return {"answer": answer}
        
    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)}. Please try again."}


@app.get("/history")
async def get_history():
    return {"history": list(history)}


@app.get("/status")
async def get_status():
    """Health check endpoint"""
    return {
        "status": "running",
        "embeddings_cached": os.path.exists(EMBEDDINGS_CACHE_FILE),
        "total_chunks": len(chunk_map),
        "current_api_key": current_api_key_index + 1,
        "total_api_keys": len(API_KEYS)
    }


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Gemini MongoDB Chatbot</title>
  <style>
    body {font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 0; height: 100vh; display: flex;}
    #sidebar {
      width: 300px; background-color: #222831; color: white; padding: 20px;
      box-shadow: 2px 0 5px rgba(0,0,0,0.1); display: flex; flex-direction: column;
    }
    #sidebar h2 {margin-top: 0; font-size: 20px; margin-bottom: 20px;}
    #historyContainer { flex: 1; overflow-y: auto; }
    #historyList { list-style-type: none; padding: 0; margin: 0; }
    .historyItem {
      padding: 10px; background-color: #393e46; margin-bottom: 10px; border-radius: 5px;
      word-break: break-word; position: relative;
    }
    .answer { display: none; margin-top: 10px; }
    .toggleBtn {
      position: absolute; top: 10px; right: 10px;
      background: #0077cc; border: none; color: white; padding: 5px 10px; border-radius: 5px; cursor: pointer;
    }
    .toggleBtn:hover { background: #005fa3; }
    #chat { flex: 1; display: flex; flex-direction: column; padding: 20px; }
    #messages {
      flex: 1; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;
      border-radius: 5px; background-color: white;
    }
    .message {
      margin-bottom: 15px; padding: 10px 15px; border-radius: 10px;
      line-height: 1.6; max-width: 80%;
    }
    .user { background-color: #d0ebff; align-self: flex-end; text-align: right; }
    .bot { background-color: #e6ffe6; align-self: flex-start; white-space: pre-line; }
    textarea {
      width: 100%; height: 60px; padding: 10px; resize: none;
      border-radius: 5px; border: 1px solid #ccc;
    }
    button.askBtn {
      padding: 10px 20px; background-color: #0077cc; color: white;
      border: none; border-radius: 5px; margin-top: 10px; cursor: pointer;
    }
    button.askBtn:hover { background-color: #005fa3; }
    .status-bar {
      background: #333; color: #ccc; padding: 5px 10px; font-size: 12px;
      border-bottom: 1px solid #555;
    }
  </style>
</head>

<body>
  <div id="sidebar">
    <h2>Recent History</h2>
    <div class="status-bar" id="statusBar">Loading status...</div>
    <div id="historyContainer">
      <ul id="historyList"></ul>
    </div>
  </div>

  <div id="chat">
    <h2>Gemini MongoDB Chatbot</h2>
    <div id="messages"></div>
    <textarea id="question" placeholder="Ask something..."></textarea><br/>
    <button class="askBtn" onclick="askQuestion()">Ask</button>
  </div>

  <script>
    async function askQuestion() {
      const questionInput = document.getElementById("question");
      const question = questionInput.value.trim();
      if (!question) return;

      const messages = document.getElementById("messages");
      messages.innerHTML += `<div class="message user"><strong>You:</strong><br/>${question}</div>`;
      questionInput.value = "";

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });

        const data = await response.json();
        const formattedAnswer = data.answer
          .replace(/\\n/g, "<br/>")
          .replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>")
          .replace(/\\d+\\.\\s/g, "<br/><strong>$&</strong>");

        messages.innerHTML += `<div class="message bot"><strong>Bot:</strong><br/>${formattedAnswer}</div>`;
        messages.scrollTop = messages.scrollHeight;

        updateHistory();
      } catch (error) {
        messages.innerHTML += `<div class="message bot"><strong>Error:</strong><br/>Failed to get response. Please try again.</div>`;
      }
    }

    async function updateHistory() {
      try {
        const response = await fetch("/history");
        const data = await response.json();
        const historyList = document.getElementById("historyList");
        historyList.innerHTML = "";

        data.history.forEach((item, idx) => {
          const listItem = document.createElement("li");
          listItem.className = "historyItem";
          listItem.innerHTML = `<strong>Q:</strong> ${item.question}
            <button class="toggleBtn" onclick="toggleAnswer(${idx})">Show</button>
            <div class="answer" id="answer-${idx}"><strong>A:</strong> ${item.answer}</div>`;
          historyList.appendChild(listItem);
        });
      } catch (error) {
        console.error("Failed to update history:", error);
      }
    }

    async function updateStatus() {
      try {
        const response = await fetch("/status");
        const data = await response.json();
        document.getElementById("statusBar").textContent = 
          `Chunks: ${data.total_chunks} | API Key: ${data.current_api_key}/${data.total_api_keys} | Cached: ${data.embeddings_cached ? 'Yes' : 'No'}`;
      } catch (error) {
        document.getElementById("statusBar").textContent = "Status: Error";
      }
    }

    function toggleAnswer(idx) {
      const answerDiv = document.getElementById(`answer-${idx}`);
      const button = answerDiv.previousElementSibling;
      if (answerDiv.style.display === "none" || answerDiv.style.display === "") {
        answerDiv.style.display = "block";
        button.textContent = "Hide";
      } else {
        answerDiv.style.display = "none";
        button.textContent = "Show";
      }
    }

    // Enter key support
    document.getElementById("question").addEventListener("keypress", function(e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
      }
    });

    window.onload = function() {
      updateHistory();
      updateStatus();
    };
  </script>
</body>
</html>
"""

# Add this at the end to run the server when script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
