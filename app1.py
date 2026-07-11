import streamlit as st
import requests

# --- Configuration ---
API_URL = "https://judgingly-datebook-niece.ngrok-free.dev/generate"
API_KEY = "secret123"

# --- Streamlit UI ---
st.set_page_config(page_title="Text-to-SQL Generator", layout="wide")
st.title("📝 Text-to-SQL Generator")

st.markdown("""
Enter your **database schema** and a **natural language query**.  
**Example schema format:
TABLE: orders\n
COLUMNS:
- order_id (PK)
- customer_id (FK)
- store_id (FK)

TABLE: customers\n
COLUMNS:
- customer_id (PK)
- first_name


""")

# --- Inputs ---
user_schema = st.text_area("Enter your database schema here:", height=200)
user_input = st.text_input("Enter your query:", "")

if st.button("Generate SQL"):
    if not user_schema.strip() or not user_input.strip():
        st.error("Please provide both schema and query.")
    else:
        try:
            headers = {"Authorization": f"Bearer {API_KEY}"}
            payload = {"user_schema": user_schema, "user_input": user_input}
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                sql_query = data["response"]["sql_query"]
                explanation = data["response"]["explanation"]

                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")

                st.subheader("Explanation")
                st.write(explanation)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
