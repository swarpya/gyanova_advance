from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import json

SERVER_URL = "http://localhost:5000/predict"

def task_1_router(**context):
    user_query = context['dag_run'].conf.get('query', 'calculate 500 minus 20')
    print(f"🚀 Phase 1 Input: {user_query}")
    
    response = requests.post(SERVER_URL, json={"query": user_query})
    tool_name = response.json().get("tool", "Unknown")
    print(f"✅ Decision: {tool_name}")
    
    context['ti'].xcom_push(key='tool_name', value=tool_name)
    context['ti'].xcom_push(key='user_query', value=user_query)

def task_2_extractor(**context):
    # 1. DEBUG INPUTS
    tool_name = context['ti'].xcom_pull(key='tool_name', task_ids='decide_tool')
    user_query = context['ti'].xcom_pull(key='user_query', task_ids='decide_tool')
    
    print(f"📥 PHASE 2 RECEIVED:\n - Tool: '{tool_name}'\n - Query: '{user_query}'")
    
    if tool_name != "Calculator":
        print(f"⚠️ Skipping: Tool is not Calculator")
        return "SKIP"

    # 2. CALL SERVER
    print(f"📞 Calling Server (Mode: Extractor)...")
    response = requests.post(SERVER_URL, json={
        "query": user_query, 
        "mode": "extractor",
        "tool_filter": tool_name
    })
    
    params = response.json().get("params", "")
    print(f"✅ RAW SERVER RESPONSE: '{params}'")
    
    context['ti'].xcom_push(key='params', value=params)

def task_3_calculator(**context):
    params = context['ti'].xcom_pull(key='params', task_ids='extract_data')
    print(f"📥 PHASE 3 RECEIVED: '{params}'")
    
    if not params or params == "SKIP":
        print("Nothing to calculate.")
        return

    # Parse
    vars = {}
    try:
        # Simple cleanup to handle potential "Parameters:" prefix if left over
        clean_params = params.replace("Parameters:", "").strip()
        
        for p in clean_params.split(','):
            if '=' in p:
                k, v = p.split('=')
                vars[k.strip()] = v.strip()
            
        a = float(vars.get('a', 0))
        b = float(vars.get('b', 0))
        op = vars.get('op', '').lower()
        
        res = 0
        if "add" in op: res = a + b
        elif "sub" in op or "minus" in op: res = a - b
        elif "mul" in op or "times" in op: res = a * b
        elif "div" in op: res = a / b
        
        print(f"🚀 FINAL CALCULATION: {a} {op} {b} = {res}")
        return res
    except Exception as e:
        print(f"❌ PARSING ERROR: {e}")

with DAG(
    dag_id='04_smart_agent',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    t1 = PythonOperator(task_id='decide_tool', python_callable=task_1_router)
    t2 = PythonOperator(task_id='extract_data', python_callable=task_2_extractor)
    t3 = PythonOperator(task_id='execute_math', python_callable=task_3_calculator)
    t1 >> t2 >> t3