import streamlit as st
from groq import Groq
import json
import re

st.set_page_config(page_title="Text-to-SQL", layout="wide")
st.title("📝 Text-to-SQL Generator (Chain & Parser)")

user_schema = st.text_area("Enter your database schema:", height=150)
user_input = st.text_input("Enter your query:", "")

if st.button("Generate SQL"):
    if not user_schema.strip() or not user_input.strip():
        st.error("Please provide both schema and query.")
    else:
        with st.spinner("Processing through Chain..."):
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                prompt = f"""
                You are an SQL expert. Respond ONLY in JSON format.
                Schema: {user_schema}
                Query: {user_input}
                
                The JSON must have the following keys:
                - "sql_query": The actual SQL code.
                - "explanation": A short explanation of how the query works.
                """
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-70b-versatile",
                    response_format={"type": "json_object"} #JSON
                )
                
                result_json = json.loads(response.choices[0].message.content)
                
                st.subheader("Generated SQL")
                st.code(result_json["sql_query"], language="sql")
                
                st.subheader("Explanation")
                st.write(result_json["explanation"])
                
            except Exception as e:
                st.error(f"Error in parsing output: {str(e)}")
