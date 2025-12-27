import requests
import json

# CONFIG
SERVER_URL = "http://localhost:5000/predict"

# TEST CASE: We simulate what Phase 1 would have passed to Phase 2
TEST_QUERY = "calculate 500 minus 20"
TEST_TOOL = "Calculator"

def test_extraction():
    print(f"🔬 TESTING PHASE 2 (Extraction) for: '{TEST_QUERY}'")
    print(f"🔧 Filter Tool: {TEST_TOOL}")
    print("-" * 40)

    # Payload matches exactly what Airflow sends
    payload = {
        "query": TEST_QUERY,
        "mode": "extractor",
        "tool_filter": TEST_TOOL
    }

    try:
        print("📨 Sending Request to Server...")
        response = requests.post(SERVER_URL, json=payload)
        
        # 1. Check Status Code
        if response.status_code != 200:
            print(f"❌ Server Error (Status {response.status_code}):")
            print(response.text)
            return

        # 2. Print RAW JSON (This is the most important part)
        data = response.json()
        print("\n📦 RAW JSON RESPONSE FROM SERVER:")
        print(json.dumps(data, indent=4))

        # 3. Simulate Parsing
        params = data.get("params", "")
        print(f"\n✅ Extracted Param String: '{params}'")
        
        # 4. Attempt Logic Parse
        if params:
            print("\n🧮 Attempting to parse variables...")
            vars = {}
            for p in params.split(','):
                if '=' in p:
                    k, v = p.split('=')
                    vars[k.strip()] = v.strip()
            print(f"   -> Parsed Dict: {vars}")
        else:
            print("❌ 'params' was empty! Check server logs.")

    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("   (Is 'python agent_server.py' running?)")

if __name__ == "__main__":
    test_extraction()