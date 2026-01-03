import requests

def main(query: str):
    print(f"🚀 Phase 1 Input: {query}")
    url = "http://host.docker.internal:5000/predict" 
    # Note: 'host.docker.internal' lets Docker talk to your local python server
    
    try:
        response = requests.post(url, json={"query": query, "mode": "router"})
        data = response.json()
        tool_name = data.get("tool", "Unknown")
        print(f"✅ Decision: {tool_name}")
        return tool_name
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Error"