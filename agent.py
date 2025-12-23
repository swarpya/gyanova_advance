import subprocess
import sys

# 1. Define your actual tools
def run_calculator():
    print("\n✅ SUCCESS: Triggering Calculator...")

def run_web_search():
    print("\n✅ SUCCESS: Triggering Web Search...")

def run_email():
    print("\n✅ SUCCESS: Triggering Email...")

# 2. Logic to ask BitNet
def get_tool_name(user_query):
    # --- CHANGE 1: THE PROMPT ---
    # We give it 3 examples so it just has to complete the pattern.
    # We leave the last "Tool:" empty for the model to fill.
    prompt = f"""Example 1:
User: Calculate 10 * 10
Tool: Calculator

Example 2:
User: Who won the match?
Tool: Web Search

Example 3:
User: Send a mail to boss
Tool: Email

Example 4:
User: {user_query}
Tool:"""

    command = [
        sys.executable, "run_inference.py",
        "-m", "models/bitnet_b1_58-large/ggml-model-i2_s.gguf",
        "-p", prompt,
        "-n", "6",       # Give it slightly more room
        "--temp", "0"    # Keep strict
    ]

    try:
        print(f"   (Sending prompt to model...)")
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        
        full_output = result.stdout
        
        # --- CHANGE 2: DEBUG PRINT ---
        # This will show you exactly what the model replied
        print(f"--------------------------------------------------")
        print(f"🔍 DEBUG RAW OUTPUT:\n{full_output}")
        print(f"--------------------------------------------------")

        # --- CHANGE 3: BETTER PARSING ---
        # We look for the LAST "Tool:" in the text and take what comes after
        if "Tool:" in full_output:
            # Split by "Tool:" and take the last part
            answer_part = full_output.split("Tool:")[-1]
            # Take the first line of that answer
            answer = answer_part.split("\n")[0].strip()
        else:
            answer = full_output.strip()
            
        return answer.lower()

    except Exception as e:
        print(f"Error: {e}")
        return None

# 3. Main Loop
if __name__ == "__main__":
    query = input("Enter your request: ")
    tool_name = get_tool_name(query)
    
    # 4. Router
    if "calculator" in tool_name:
        run_calculator()
    elif "web" in tool_name or "search" in tool_name:
        run_web_search()
    elif "email" in tool_name:
        run_email()
    else:
        print(f"❌ FAILED. Model said: '{tool_name}'")


# ... keep all your functions above the same ...

if __name__ == "__main__":
    print("🤖 BitNet Agent Ready! (Type 'exit' to quit)")
    
    while True:
        # 1. Get input
        query = input("\n📝 Enter command: ")
        
        # 2. Check for exit
        if query.lower() in ["exit", "quit"]:
            print("Bye!")
            break
            
        # 3. Get the tool name
        tool_name = get_tool_name(query)
        
        # 4. Router Logic
        if "calculator" in tool_name:
            run_calculator()
        elif "web" in tool_name or "search" in tool_name:
            run_web_search()
        elif "email" in tool_name:
            run_email()
        else:
            print(f"❌ Unsure. Model replied: {tool_name}")