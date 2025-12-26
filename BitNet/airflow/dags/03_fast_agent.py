from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import json

def ask_agent_server(**context):
    user_query = context['dag_run'].conf.get('query', 'What is 10 + 10?')
    
    print(f"🚀 Sending query to Agent Server: {user_query}")
    
    try:
        # This takes milliseconds because the server is already awake
        response = requests.post(
            "http://localhost:5000/predict", 
            json={"query": user_query}
        )
        result = response.json()
        tool = result.get("tool", "Unknown")
        
        print(f"✅ Agent Decision: {tool}")
        return tool
        
    except Exception as e:
        print("❌ Is the server running? Run 'python agent_server.py' first.")
        raise e

with DAG(
    dag_id='fast_agent_inference',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    predict = PythonOperator(
        task_id='get_tool_decision',
        python_callable=ask_agent_server
    )