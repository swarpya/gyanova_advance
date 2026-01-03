def main(params: str):
    if not params or params == "SKIP":
        return "Nothing to calculate"

    vars = {}
    # Simple Parsing Logic
    for p in params.split(','):
        if '=' in p:
            k, v = p.split('=')
            vars[k.strip()] = v.strip()
            
    a = float(vars.get('a', 0))
    b = float(vars.get('b', 0))
    op = vars.get('op', '').lower()
    
    res = 0
    if "add" in op: res = a + b
    elif "sub" in op or "minus" in op: res = a - b
    elif "mul" in op: res = a * b
    elif "div" in op: res = a / b
    
    print(f"🚀 FINAL RESULT: {res}")
    return res