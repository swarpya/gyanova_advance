import subprocess
import re

# The fixed list of tools allowed
ALLOWED_TOOLS = ["Calculator", "Web Search", "Email"]

def get_tool_for_query(user_query):
    # 1. Construct a Strict Prompt
    # We put the options inside the prompt to guide the model
    prompt = f"""
Instruction: Select the correct tool for the user's request.
Options: {', '.join(ALLOWED_TOOLS)}
User Request: "{user_query}"
Answer: The best tool is"""

    # 2. Run the BitNet Binary
    # -n 3: Stop after 3 tokens (prevents rambling)
    # --temp 0: Zero creativity (essential for logic)
    cmd = [
        "./build/bin/main",
        "-m", "models/bitnet_b1_58-large/ggml-model-i2_s.gguf",
        "-p", prompt,
        "-n", "3",
        "--temp", "0"
    ]
    
    # Run command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    # 3. Clean the output (BitNet is chatty, so we look for our keywords)
    # This logic checks if any allowed tool appears in the output
    found_tool = "Unknown"
    for tool in ALLOWED_TOOLS:
        if tool.lower() in output.lower():
            found_tool = tool
            break
            
    return found_tool

# --- Test It ---
query = "What is the square root of 144?"
tool = get_tool_for_query(query)
print(f"Query: {query}")
print(f"Selected Tool: {tool}")

if tool == "Calculator":
    print(">> Triggering Calculator Function...")
elif tool == "Web Search":
    print(">> Triggering Search Function...")