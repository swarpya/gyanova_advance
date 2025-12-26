from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import subprocess
import shutil
import json
import os
import sys

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = os.path.join(BASE_DIR, "build/bin/llama-cli")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")

# --- CRITICAL FIX: DO NOT USE RAM DISK ---
# Your /dev/shm is only 64MB, but the model is 258MB.
# We must use the disk path directly to prevent crashing Airflow.
MODEL_PATH = os.path.join(BASE_DIR, "models/bitnet_b1_58-large/ggml-model-i2_s.gguf")

if not os.path.exists(BINARY_PATH):
    print(f"❌ CRITICAL: Could not find {BINARY_PATH}.")
    sys.exit(1)

# --- LOAD SEARCH ENGINE ---
print("⏳ Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)
corpus_texts = [item['text'] for item in dataset]
corpus_embeddings = embedder.encode(corpus_texts, convert_to_tensor=True)

print("✅ Server Ready on port 5000 (Safe Disk Mode)")

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
    # NOTE: increased -n to 10 to ensure we catch the answer
    cmd = [
        BINARY_PATH,
        "-m", MODEL_PATH,
        "-n", "10",            
        "-t", "2",             
        "-p", final_prompt,
        "-ngl", "0",           
        "-c", "100",          
        "--temp", "0",
        "-b", "1"
    ]
    
    try:
        print(f"--------------\n🧠 RUNNING INFERENCE for: '{user_query}'")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='ignore'
        )
        
        full_output = result.stdout
        # Debug Print
        print(f"📝 RAW OUTPUT:\n{full_output}\n")
        
        if "Tool:" in full_output:
            answer = full_output.split("Tool:")[-1].split("\n")[0].strip()
        else:
            answer = full_output.strip().split("\n")[-1].strip()

        print(f"✅ PARSED ANSWER: '{answer}'\n--------------")
        return jsonify({"tool": answer})
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)