# A-Loacl-AI-Chatbot-using-Python-Flask-ollama-LLM-and-langchain.
This project is a fully functional AI chatbot powered by Ollama’s local LLM, LangChain for context and prompt management, and a Flask backend for seamless web integration. It demonstrates how to run an intelligent conversational agent locally, ensuring data privacy, fast inference, and modular scalability. 


## Features

- Local LLM hosting (Ollama) — no external API calls required.
- Prompt / context management using **LangChain**.
- Simple REST API built with **Flask** for driving chat clients.
- Example client usage (curl / simple front-end).
- Extensible architecture (add retrieval, RAG, tool use, etc).
---

## Prerequisites

- A machine with enough resources to run the chosen local model (RAM/CPU/GPU as required by the model).  
- Python 3.9 or later.  
- Ollama installed (follow official Ollama docs for installation and model download).  
- Basic familiarity with Python and REST APIs.

> **Note:** Ollama install and model pulling instructions change over time. Please consult the official Ollama docs for your OS: **https://ollama.ai/docs** (or your preferred official source).

## Quick Start (Step-by-step)

Below are the minimal steps to get the project running locally.

## 1. Install Ollama and pull a model
Install Ollama according to official instructions for your OS.
<img width="1151" height="482" alt="image" src="https://github.com/user-attachments/assets/1fee410c-dc85-4d4b-aa72-a157c0cbaec7" />

---
## After downloading and installing Ollama, we get an interface like the one shown below:
---
<img width="1235" height="572" alt="image" src="https://github.com/user-attachments/assets/f7d0eddd-9eb1-4f4f-8cd7-155b457e40dc" />

## 2. Pull a model to run locally, e.g.:
---
## In ths project i use "gemma3:1b" This LLM

   
   ```
   ollama pull <model-name>
   ```
<img width="1468" height="135" alt="image" src="https://github.com/user-attachments/assets/3a61fb53-9178-4985-a88f-bfc1fa81f949" />

---
After successfully Installed the LLM, verify that it’s working properly.
---

<img width="1372" height="290" alt="image" src="https://github.com/user-attachments/assets/3f33674d-4d63-46e6-a32d-224e0771388a" />


## File structure:
---
<img width="811" height="346" alt="image" src="https://github.com/user-attachments/assets/1bcc249b-f97c-4856-b861-9e07a3ae4ff8" />

---
## /Project/serve.py
---

```
# server.py - Flask backend (Python)
# backend with Ollama + LangChain + Chromadb
# -----------------------------

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "context")
ALLOWED_EXT = {'pdf'}
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vectorstore = None  # global vectorstore

# -----------------------------
# Initialize Embeddings + LLM
# -----------------------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="phi3.5")

# CHANGED: Removed “Phi introduction” and made it context-only
system_prompt = """You are a medical document analysis assistant.
Answer ONLY based on the provided PDF context below.
If the answer is not found in the documents, reply:
'Sorry, I could not find that information in the uploaded documents.'
Keep answers short, factual, and relevant.
"""

qa_prompt = ChatPromptTemplate.from_template(
    "{context}\n\nQuestion: {input}\n\nAnswer:"
)

# -----------------------------
# Build Vectorstore
# -----------------------------
def build_vectorstore():
    global vectorstore
    all_docs = []

    for fn in os.listdir(UPLOAD_FOLDER):
        if fn.lower().endswith('.pdf'):
            path = os.path.join(UPLOAD_FOLDER, fn)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)

    if not all_docs:
        print("No PDFs found in context folder.")
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIR)
    print("Vector database built successfully.")
    return True


# Try building vectorstore at startup
try:
    built = build_vectorstore()
    print("Vector DB built from context folder:", built)
except Exception as e:
    print("Vector DB build error (ok if empty):", e)


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# List available PDFs
@app.route('/api/files')
def list_files():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.pdf')]
    return jsonify({'files': files})


# Upload PDFs via API
@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    saved = []

    for f in files:
        filename = secure_filename(f.filename)
        if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT:
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            saved.append(filename)

    # Rebuild vectorstore after upload
    ok = build_vectorstore()
    if not ok:
        return jsonify({'detail': 'No valid documents found.'}), 400

    return jsonify({'detail': 'Indexed successfully', 'files': saved})


# -----------------------------
# Query Endpoint (Updated)
# -----------------------------
@app.route('/api/query', methods=['POST'])
def query():
    global vectorstore
    data = request.get_json() or {}
    q = data.get('query')

    if not q:
        return jsonify({'error': 'No query provided'}), 400
    if vectorstore is None:
        return jsonify({'error': 'No documents indexed yet'}), 400

    try:
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.invoke(q)
        context_documents_str = "\n\n".join([d.page_content for d in relevant_docs])

        # CHANGED: No Runnable chain — direct prompt formatting for control
        prompt_input = qa_prompt.format(context=context_documents_str, input=q)
        result = llm.invoke(prompt_input)

        return jsonify({'answer': str(result)})

    except Exception as e:
        print("Query Error:", e)
        return jsonify({'error': 'Server error', 'detail': str(e)}), 500


# Serve the frontend HTML
@app.route('/')
def index():
    return render_template('index.html')


# -----------------------------
# Run Server
# -----------------------------
if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
```
---
## Now I have designed a front-end UI to query the desired information stored in the /context folder, according to the folder structure:
---
/Templates/index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Closed AI</title>
<style>
  body {
    margin: 0;
    font-family: "Segoe UI", sans-serif;
    background: linear-gradient(135deg, #9cecfb, #65c7f7, #0052d4);
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #fff;
  }

  .chat-container {
    width: 95%;
    max-width: 900px;
    height: 90vh;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 6px 30px rgba(0,0,0,0.25);
    overflow: hidden;
  }

  header {
    background-color: rgba(0,0,0,0.3);
    padding: 15px;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 600;
  }

  .upload-box {
    display: flex;
    align-items: center;
    padding: 6px 12px;
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    font-size: 0.9em;
    margin: 10px;
  }

  .upload-box input[type="file"] {
    flex: 1;
    font-size: 0.9em;
  }

  .upload-box button {
    padding: 6px 12px;
    margin-left: 8px;
    font-size: 0.9em;
    cursor: pointer;
    border-radius: 6px;
    border: none;
    background-color: #004aad;
    color: #fff;
  }

  .upload-box button:hover {
    background-color: #007bff;
  }

  #uploadStatus {
    margin-left: 10px;
    font-size: 0.85em;
    color: #fff;
  }

  .messages {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    scroll-behavior: smooth;
    display: flex;
    flex-direction: column;
  }

  .msg {
    padding: 12px 18px;
    border-radius: 14px;
    margin: 6px 0;
    max-width: 75%;
    word-wrap: break-word;
    line-height: 1.5;
  }

  .msg.user {
    background-color: #0066ff;
    align-self: flex-end;
    border-bottom-right-radius: 0;
    color: #fff;
  }

  .msg.bot {
    background-color: #00bcd4;
    align-self: flex-start;
    border-bottom-left-radius: 0;
    color: #fff;
  }

  .input-area {
    display: flex;
    padding: 10px 12px;
    background: rgba(255,255,255,0.1);
    border-top: 1px solid rgba(255,255,255,0.3);
  }

  .input-area input {
    flex: 1;
    padding: 12px;
    border: none;
    border-radius: 10px;
    outline: none;
    font-size: 15px;
  }

  .input-area button {
    background-color: #004aad;
    color: #fff;
    border: none;
    padding: 12px 18px;
    border-radius: 10px;
    margin-left: 8px;
    cursor: pointer;
    font-size: 14px;
  }

  .input-area button:hover {
    background-color: #007bff;
  }

  .input-area #clearBtn {
    background-color: #d9534f;
  }
</style>
</head>
<body>
  <div class="chat-container">
    <header>🤖 Closed AI 🤖</header>

    <div class="upload-box">
      <input type="file" id="files" multiple accept="application/pdf">
      <button id="uploadBtn">Upload</button>
      <div id="uploadStatus"></div>
    </div>

    <div id="messages" class="messages"></div>

    <div class="input-area">
      <input id="queryInput" placeholder="Ask Anything......." autocomplete="off">
      <button id="askBtn">Ask</button>
      <button id="clearBtn">Clear</button>
    </div>
  </div>

<script>
const apiBase = '' // same origin backend

const messagesEl = document.getElementById('messages')
const queryInput = document.getElementById('queryInput')
const askBtn = document.getElementById('askBtn')
const clearBtn = document.getElementById('clearBtn')
const uploadBtn = document.getElementById('uploadBtn')
const uploadStatus = document.getElementById('uploadStatus')
const filesInput = document.getElementById('files')

function appendMessage(text, who='bot'){
  const d = document.createElement('div')
  d.className = 'msg ' + (who==='user' ? 'user' : 'bot')
  d.textContent = text
  messagesEl.appendChild(d)
  messagesEl.scrollTop = messagesEl.scrollHeight
}

async function sendQuery(){
  const q = queryInput.value.trim()
  if(!q) return
  appendMessage(q,'user')
  queryInput.value = ''
  appendMessage('Thinking...','bot')

  try{
    const res = await fetch(apiBase + '/api/query',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q})
    })
    const data = await res.json()
    messagesEl.lastElementChild.textContent = data.answer || 'No answer returned.'
  }catch(err){
    messagesEl.lastElementChild.textContent = 'Error: ' + (err.message||err)
  }
}

askBtn.addEventListener('click', sendQuery)
clearBtn.addEventListener('click', ()=>{ messagesEl.innerHTML='' })

// Send on Enter, Shift+Enter for newline
queryInput.addEventListener('keydown', function(event){
  if(event.key==='Enter' && !event.shiftKey){
    event.preventDefault()
    sendQuery()
  }
})

// Upload PDFs
uploadBtn.addEventListener('click', async ()=>{
  const files = filesInput.files
  if(!files.length){uploadStatus.textContent = 'Select at least one PDF'; return}
  const fd = new FormData()
  for(const f of files) fd.append('files', f)
  uploadStatus.textContent = 'Uploading...'
  uploadBtn.disabled = true

  try{
    const res = await fetch(apiBase + '/api/upload', {method:'POST', body: fd})
    const j = await res.json()
    uploadStatus.textContent = j.detail || 'Indexed successfully.'
  }catch(e){
    uploadStatus.textContent = 'Upload failed: '+ e.message
  }

  uploadBtn.disabled = false
})
</script>
</body>
</html>

```
---
## The output should look like this:
<img width="1398" height="1252" alt="image" src="https://github.com/user-attachments/assets/710b49fb-bde1-45f8-8f62-c875a01ecb22" />

---
## Now, set up all dependencies for this project. According to the folder structure, create a text file named requirements.txt under the project folder.

/project
```
flask==2.3.2
langchain==0.3.0
langchain-text-splitters
langchain-chroma
langchain-community
ollama
sentence-transformers
chromadb
werkzeug

```
---
## Create a virtual environment for this project aftere open the cmd from project folder "PS D:\project>" 
---
```
python -m venv venv
```
## After active the VENV then go the venv mode
```
PS D:\project> .\venv\Scripts\activate
```
<img width="866" height="31" alt="image" src="https://github.com/user-attachments/assets/700c8147-3162-4258-b72d-1fa56d5cffe3" />
---

## to install all depencys use bellow this command from your terminal.
---

```
(venv) PS D:\project> pip install -r requirements.txt
```
---
---
## After installing all dependencies with "pip install pypdf", run the command to read a PDF file.

#Read a PDF file

#Extract text or images

#Merge or split PDFs

#Rotate or rearrange pages

---
```
(venv) PS D:\project> pip install pypdf
```
---
## Fainally Run the application
---

```
(venv) PS D:\project> python server.py
```
---
## Once the main application is executed successfully, a confirmation message appears in the terminal. After that, we can access the application by navigating to the provided link in the web browser.
<img width="1477" height="822" alt="image" src="https://github.com/user-attachments/assets/475c9c58-0358-4628-ac44-5ac4c09cfd4d" />

---
## Starting Flask server on http://127.0.0.1:5000 ...
<img width="1290" height="317" alt="image" src="https://github.com/user-attachments/assets/d3d3934d-1a49-46cb-9320-a2e6de32320d" />

---
## Now I make a query from my local UI, which is actually my frontend.
<img width="2140" height="640" alt="image" src="https://github.com/user-attachments/assets/903eddaa-3ab9-4605-b572-7742d3d0725c" />
This local AI provides answers from the PDF files stored in our local /context folder using natural language. As shown in the folder structure above, all PDF files should be well organized for accurate responses.

---
## ======The END=====
---
