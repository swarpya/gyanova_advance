import json
import subprocess
import sys
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
MODEL_PATH = "models/bitnet_b1_58-large/ggml-model-i2_s.gguf"
DATASET_FILE = "dataset.json"

class VectorAgent:
    def __init__(self):
        print("⏳ Loading Embedding Model (this happens once)...")
        # 'all-MiniLM-L6-v2' is a tiny, fast model perfect for CPUs
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("📂 Loading Dataset...")
        with open(DATASET_FILE, 'r') as f:
            self.dataset = json.load(f)
            
        # Create a list of just the texts for searching
        self.corpus_texts = [item['text'] for item in self.dataset]
        
        print("🧠 Building Vector Index...")
        # Convert all 90 examples into vectors
        self.corpus_embeddings = self.embedder.encode(self.corpus_texts, convert_to_tensor=True)
        print("✅ System Ready!\n")

    def get_relevant_examples(self, user_query, k=3):
        """Finds the top k most similar examples from dataset.json"""
        # Convert user query to vector
        query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
        
        # Search against our dataset
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=k)
        hits = hits[0] # Get the first (and only) query results
        
        # Format the examples for the LLM prompt
        examples_text = ""
        for i, hit in enumerate(hits):
            idx = hit['corpus_id']
            data = self.dataset[idx]
            examples_text += f"Example {i+1}:\nUser: {data['text']}\nTool: {data['tool']}\n\n"
            
        return examples_text

    def ask_bitnet(self, user_query):
        # 1. Retrieve Dynamic Context
        context_examples = self.get_relevant_examples(user_query)
        
        # 2. Construct the Prompt
        # We sandwich the examples between the instruction and the user query
        prompt = f"""Instruction: Identify the correct tool (Calculator, Web Search, Email).
        
{context_examples}Example 4:
User: {user_query}
Tool:"""

        # 3. Run BitNet
        cmd = [
            sys.executable, "run_inference.py",
            "-m", MODEL_PATH,
            "-p", prompt,
            "-n", "6",       # strict limit
            "--temp", "0"    # zero creativity
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            full_output = result.stdout
            
            # 4. Extract Answer
            if "Tool:" in full_output:
                answer = full_output.split("Tool:")[-1].split("\n")[0].strip()
            else:
                answer = full_output.strip()
            
            # Clean up (remove punctuation/spaces)
            return answer.lower()
            
        except Exception as e:
            print(f"Error: {e}")
            return "error"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Initialize the Agent
    agent = VectorAgent()
    
    while True:
        query = input("📝 Enter request (or 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break
            
        # Get decision
        tool_name = agent.ask_bitnet(query)
        print(f"🤖 Model Selection: [{tool_name}]")
        
        # Router
        if "calculator" in tool_name:
            print(">> 🧮 Launching Calculator...")
        elif "web" in tool_name or "search" in tool_name:
            print(">> 🌐 Launching Web Search...")
        elif "email" in tool_name:
            print(">> 📧 Launching Email Client...")
        else:
            print(">> ❌ Unknown Tool.")
        print("-" * 40)