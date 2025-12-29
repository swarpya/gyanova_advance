from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import subprocess
import shutil
import json
import os
import sys
import re
from collections import Counter # <--- NEW: For counting votes

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = os.path.join(BASE_DIR, "build/bin/llama-cli")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, "models/bitnet_b1_58-large/ggml-model-i2_s.gguf")
RAM_DISK_MODEL = "/tmp/bitnet_fast_model.gguf"

if not os.path.exists(BINARY_PATH): sys.exit(1)

try:
    if not os.path.exists(RAM_DISK_MODEL):
        shutil.copyfile(ORIGINAL_MODEL_PATH, RAM_DISK_MODEL)
    MODEL_PATH = RAM_DISK_MODEL
except:
    MODEL_PATH = ORIGINAL_MODEL_PATH

print("⏳ Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)
corpus_texts = [item['text'] for item in dataset]
corpus_embeddings = embedder.encode(corpus_texts, convert_to_tensor=True)

print("✅ Server Ready (Hybrid Classification)")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_query = data.get('query')
    mode = data.get('mode', 'router') 
    tool_filter = data.get('tool_filter', None)

    print(f"\n📨 REQUEST: Mode=[{mode}] Query=['{user_query}']")
    
    # 1. Search
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]
    
    # --- NEW: FAST PATH (Voting Logic) ---
    if mode == 'router':
        # Get the tool names of the top 3 hits
        top_tools = [dataset[hits[i]['corpus_id']]['tool'] for i in range(3)]
        print(f"🗳️  Search Hits: {top_tools}")
        
        # Count the votes
        vote_counts = Counter(top_tools)
        winner, count = vote_counts.most_common(1)[0]
        
        # THRESHOLD: If 3 out of 3 (or 2 out of 3) agree, trust them!
        # Strictness: 3 means unanimous (very safe), 2 means majority (faster).
        if count >= 2: 
            print(f"⚡ FAST PATH: Skipping LLM. Majority vote says '{winner}'")
            return jsonify({"tool": winner})
        
        print(f"🤔 AMBIGUOUS: Votes split {vote_counts}. Asking LLM...")

    # 2. Build Prompt (Only runs for Extractor OR Ambiguous Router)
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

    # 3. RUN BITNET
    cmd = [
        BINARY_PATH, "-m", MODEL_PATH, "-n", "40", "-t", "2", "-p", final_prompt,
        "-ngl", "0", "-c", "2048", "--temp", "0", "-b", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        full_log = result.stderr + result.stdout
        
        # Parsing Logic
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
            return jsonify({"tool": answer})
        else:
            return jsonify({"params": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)