import requests

def main(query: str, tool: str):
    if tool != "Calculator":
        print(f"⚠️ Skipping extraction. Tool is '{tool}'")
        return "SKIP"

    print(f"⛏️ Phase 2 Extracting: {query}")
    url = "http://host.docker.internal:5000/predict"
    
    response = requests.post(url, json={
        "query": query, 
        "mode": "extractor",
        "tool_filter": tool
    })
    
    params = response.json().get("params", "")
    print(f"✅ Extracted: {params}")
    return params