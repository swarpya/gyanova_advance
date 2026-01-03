from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import subprocess
import shutil
import json
import os
import sys
import re
from collections import Counter
from functools import lru_cache  # <--- OPTIMIZATION 1: Caching

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = os.path.join(BASE_DIR, "build/bin/llama-cli")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, "models/bitnet_b1_58-large/ggml-model-i2_s.gguf")
RAM_DISK_MODEL = "/tmp/bitnet_fast_model.gguf"

if not os.path.exists(BINARY_PATH):
    print(f"❌ CRITICAL: Binary not found at {BINARY_PATH}")
    sys.exit(1)

# --- FRESH MODEL COPY ---
try:
    if not os.path.exists(RAM_DISK_MODEL):
        print(f"🚀 Copying model to {RAM_DISK_MODEL} (Please wait)...")
        shutil.copyfile(ORIGINAL_MODEL_PATH, RAM_DISK_MODEL)
    MODEL_PATH = RAM_DISK_MODEL
except Exception as e:
    print(f"⚠️ RAM Copy failed: {e}. Using disk model.")
    MODEL_PATH = ORIGINAL_MODEL_PATH

print("⏳ Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)
corpus_texts = [item['text'] for item in dataset]
corpus_embeddings = embedder.encode(corpus_texts, convert_to_tensor=True)

print("✅ Server Ready (Hybrid + Caching Enabled)")

# --- THE LOGIC CORE (Wrapped in Cache) ---
# This function remembers the last 128 distinct inputs.
# If you ask the same thing twice, it runs instantly.
@lru_cache(maxsize=128)
def get_decision_logic(user_query, mode, tool_filter):
    print(f"\n📨 PROCESSING (No Cache Hit): Mode=[{mode}] Query=['{user_query}']")
    
    # 1. SEARCH (RAG)
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]
    
    # 2. SHORTCUT: MAJORITY VOTING (Only for Router Mode)
    if mode == 'router':
        # Look at the top 3 hits
        top_tools = [dataset[hits[i]['corpus_id']]['tool'] for i in range(min(3, len(hits)))]
        vote_counts = Counter(top_tools)
        winner, count = vote_counts.most_common(1)[0]
        
        # If 2 or more examples agree, trust them and SKIP the LLM
        if count >= 2:
            print(f"⚡ FAST PATH: Majority vote ({count}/3) says '{winner}'")
            return {"tool": winner}
        
        print(f"🤔 AMBIGUOUS: Votes split {vote_counts}. Asking LLM...")

    # 3. BUILD PROMPT (If we couldn't skip)
    prompt_context = ""
    example_count = 0
    for hit in hits:
        if example_count >= 3: break
        d = dataset[hit['corpus_id']]
        
        if mode == 'router':
            prompt_context += f"Example {example_count+1}:\nUser: {d['text']}\nTool: {d['tool']}\n\n"
            example_count += 1
        elif mode == 'extractor' and d['tool'] == tool_filter:
            params = d.get('parameters', 'N/A')
            prompt_context += f"Example {example_count+1}:\nUser: {d['text']}\nParameters: {params}\n\n"
            example_count += 1

    next_num = example_count + 1
    if mode == 'router':
        final_prompt = f"Instruction: Identify the correct tool.\n{prompt_context}Example {next_num}:\nUser: {user_query}\nTool:"
    else:
        final_prompt = f"Instruction: Extract parameters.\n{prompt_context}Example {next_num}:\nUser: {user_query}\nParameters:"

    # 4. RUN BITNET LLM
    cmd = [
        BINARY_PATH, "-m", MODEL_PATH, "-n", "40", "-t", "2", "-p", final_prompt,
        "-ngl", "0", "-c", "2048", "--temp", "0", "-b", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        full_log = result.stderr + result.stdout
        
        # 5. PARSE OUTPUT
        answer = "Unknown"
        marker = f"Example {next_num}:"
        start_idx = full_log.find(marker)
        
        if start_idx != -1:
            relevant_section = full_log[start_idx:]
            splitter = "Tool:" if mode == 'router' else "Parameters:"
            split_idx = relevant_section.find(splitter)
            if split_idx != -1:
                after_splitter = relevant_section[split_idx + len(splitter):]
                answer = after_splitter.split('\n')[0].strip()
        
        answer = answer.strip(' "')
        print(f"🎯 LLM ANSWER: '{answer}'")

        if mode == 'router':
            return {"tool": answer}
        else:
            return {"params": answer}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

# --- FLASK ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_query = data.get('query')
    mode = data.get('mode', 'router') 
    tool_filter = data.get('tool_filter', None)

    # Call the cached function
    # Note: We must ensure arguments are hashable. Strings/None are fine.
    try:
        result = get_decision_logic(user_query, mode, tool_filter)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)