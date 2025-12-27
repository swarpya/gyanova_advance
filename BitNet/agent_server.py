from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import subprocess
import shutil
import json
import os
import sys

app = Flask(__name__)

# --- CONFIGURATION ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = os.path.join(BASE_DIR, "build/bin/llama-cli")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, "models/bitnet_b1_58-large/ggml-model-i2_s.gguf")

# --- OPTIMIZATION: USE /tmp (RAM DISK) ---
# Your 'df -h' shows / is tmpfs (RAM) with 3.9GB free.
# This is much larger than /dev/shm (64MB) and just as fast.
RAM_DISK_MODEL = "/tmp/bitnet_fast_model.gguf" 

if not os.path.exists(BINARY_PATH):
    print(f"❌ CRITICAL: Could not find {BINARY_PATH}.")
    sys.exit(1)

# Copy model to /tmp (RAM) for speed
try:
    if not os.path.exists(RAM_DISK_MODEL):
        print("🚀 Optimizing: Copying model to /tmp (RAM Disk)...")
        shutil.copyfile(ORIGINAL_MODEL_PATH, RAM_DISK_MODEL)
    MODEL_PATH = RAM_DISK_MODEL
    print(f"✅ Model is in RAM ({RAM_DISK_MODEL})")
except Exception as e:
    print(f"⚠️ Copy failed: {e}")
    print("⚠️ Falling back to Hard Drive (Slower)")
    MODEL_PATH = ORIGINAL_MODEL_PATH

# --- LOAD SEARCH ENGINE ---
print("⏳ Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)
corpus_texts = [item['text'] for item in dataset]
corpus_embeddings = embedder.encode(corpus_texts, convert_to_tensor=True)

print("✅ Server Ready on port 5000")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_query = data.get('query')
    
    # 1. Search
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
    
    # 2. Prompt
    prompt_context = ""
    for i, hit in enumerate(hits):
        idx = hit['corpus_id']
        d = dataset[idx]
        prompt_context += f"Example {i+1}:\nUser: {d['text']}\nTool: {d['tool']}\n\n"
        
    final_prompt = f"""Instruction: Identify the correct tool (Calculator, Web Search, Email).
{prompt_context}Example 4:
User: {user_query}
Tool:"""

    # 3. DIRECT EXECUTION
    cmd = [
        BINARY_PATH,
        "-m", MODEL_PATH,
        "-n", "10",            
        "-t", "2",             
        "-p", final_prompt,
        "-ngl", "0",           
        "-c", "2048",          
        "--temp", "0",
        "-b", "1"
    ]
    
    try:
        # Run inference
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='ignore'
        )
        
        full_output = result.stdout
        
        # Parse output
        if "Tool:" in full_output:
            answer = full_output.split("Tool:")[-1].split("\n")[0].strip()
        else:
            answer = full_output.strip().split("\n")[-1].strip()

        return jsonify({"tool": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)