import streamlit as st
from groq import Groq
import json
import csv
from datetime import datetime
from langchain_core.prompts import PromptTemplate

#CSS Style
def css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown, .stTextInput, .stTextArea {
        font-size: 24px !important;
    }
    h1 { color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

local_css()

sql_prompt_template = PromptTemplate(
    input_variables=["schema", "query"],
    template="You are an SQL expert. Schema: {schema}. Query: {query}. Respond ONLY in JSON: {'sql_query': '...', 'explanation': '...'}"
)

#Evaluation
def evaluate_model(generated_sql, ground_truth):
    return 1 if generated_sql.strip().lower() == ground_truth.strip().lower() else 0

st.title("🚀 SQL Generation")

tab1, tab2 = st.tabs(["Generator", "Evaluation"])

with tab1:
    user_schema = st.text_area("Schema:", height=150)
    user_input = st.text_input("Query:")
    
    if st.button("Generate"):
        prompt = sql_prompt_template.format(schema=user_schema, query=user_input)
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile",
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        st.code(result["sql_query"], language="sql")
        st.session_state['last_sql'] = result["sql_query"]

with tab2:
    st.subheader("Model Evaluation")
    ground_truth = st.text_area("Enter expected SQL for testing:")
    if st.button("Run Accuracy Test"):
        if 'last_sql' in st.session_state:
            score = evaluate_model(st.session_state['last_sql'], ground_truth)
            st.metric("Accuracy Score", f"{score * 100}%")
        else:
            st.warning("Generate an SQL first!")
