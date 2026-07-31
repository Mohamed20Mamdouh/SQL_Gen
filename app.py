import streamlit as st
import json
import sqlite3
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from groq import Groq

def style():
    st.set_page_config(page_title="SQLSync", page_icon="https://share.gemini.google/7WvOKpjtuAcu")
    st.markdown("""
    <link href="https://fonts.google.com/specimen/Fira+Code" rel="stylesheet">
    <style>
        .stApp { background-color: #FAF8F5; max-width: 800px; margin: 0 auto; font-family: 'Cairo', Medium 500; }
        h1 { color: #D35400; text-align: left; display: flex; align-items: center; gap: 12px; font-weight: 700; }
        label { font-size: 32px !important; color: #5D4037 !important; font-weight: 600 !important; }
        .stTextInput input, .stTextArea textarea { background-color: #FFFFFF !important; border: 2px solid #F39C12 !important; border-radius: 12px !important; padding: 12px !important; box-shadow: 0 4px 12px rgba(211, 84, 0, 0.08) !important; transition: all 0.3s ease-in-out !important; }
        .stTextInput input:focus, .stTextArea textarea:focus { border-color: #D35400 !important; box-shadow: 0 4px 15px rgba(211, 84, 0, 0.2) !important; }
        .stButton button { background: linear-gradient(135deg, #D35400, #E67E22) !important; color: white !important; width: 100%; border-radius: 10px; font-weight: bold; font-size: 16px; border: none; padding: 10px; box-shadow: 0 4px 10px rgba(211, 84, 0, 0.2); transition: 0.3s; }
        .stButton button:hover { background: linear-gradient(135deg, #BA4A00, #D35400) !important; box-shadow: 0 6px 15px rgba(211, 84, 0, 0.35); }
    </style>

<h1><img src="https://share.gemini.google/uSP60sKxnDi2" width="40" style="vertical-align: middle;">SQLSync</h1>""", unsafe_allow_html=True)
style()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

col1, col2 = st.columns([2, 1])
with col1:
    schema_option = st.radio("How do you want to provide the Schema?", ("Upload .sql", "Text"))
with col2:
    sql_dialect = st.selectbox("SQL Dialect:", ["MySQL", "PostgreSQL", "SQLite", "SQL Server"])

schema_text = ""

if schema_option == "Upload .sql File":
    uploaded_file = st.file_uploader("Upload your .sql file:", type=['sql', 'txt'])
    if uploaded_file:
        schema_text = uploaded_file.read().decode("utf-8")
else:
    schema_text = st.text_area("Paste Your Schema Here (Optional):", height=150, placeholder="CREATE TABLE users (id INT, name VARCHAR(50));")

user_input = st.text_input("Enter your SQL query request:", placeholder="Prompt your query...")

if st.button("Generate SQL"):
    if not user_input:
        st.warning("Please Enter Your SQL Query: ")
    else:
        with st.spinner("Processing..."):
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                
                if schema_text.strip():
                    embeddings = get_embeddings()
                    chunks = [f"CREATE TABLE {c.strip()}" for c in schema_text.split("CREATE TABLE") if c.strip()]
                    docs = [Document(page_content=c) for c in chunks]
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    relevant_schema = "\n".join([doc.page_content for doc in vectorstore.similarity_search(user_input, k=3)])
                    
                    prompt = f"""You are an advanced Text-to-SQL expert specialized in {sql_dialect}. 
Given the following database schema, write an accurate, optimized, and syntax-compliant SQL query for {sql_dialect}.

Database Schema:
{relevant_schema}

User Request: {user_input}

Respond ONLY in valid JSON format with the following keys:
- "sql_query": The generated SQL query string.
- "explanation": A brief explanation of how the query works."""
                else:
                    prompt = f"""You are an advanced Text-to-SQL expert specialized in {sql_dialect}. 
Write a standard SQL query for {sql_dialect} based on the request:

User Request: {user_input}

Respond ONLY in valid JSON format with the following keys:
- "sql_query": The generated SQL query string.
- "explanation": A brief explanation of how the query works."""
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    seed=42
                )
                
                result = json.loads(response.choices[0].message.content)
                st.session_state["last_sql"] = result["sql_query"]
                st.session_state["last_explanation"] = result.get("explanation", "")
            except Exception as e:
                st.error(f"Error: {e}")

if "last_sql" in st.session_state:
    st.markdown("---")
    st.subheader("(SQL Generation)")
    st.code(st.session_state["last_sql"], language="sql")
    
    st.subheader("💡 Explanation")
    st.write(st.session_state["last_explanation"])
    
    file_extension = "sql"
    export_filename = f"query_output.{sql_dialect.lower().replace(' ', '_')}.{file_extension}"
    st.download_button(
        label=f"📥 Export as File ({sql_dialect})",
        data=f"-- Generated by SQLSync for {sql_dialect}\n" + st.session_state["last_sql"],
        file_name=export_filename,
        mime="text/plain")
    st.markdown("---")
    st.subheader("⚙️ Advanced Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Query", use_container_width=True):
            try:
                conn = sqlite3.connect(':memory:')

                df = pd.read_sql_query(st.session_state["last_sql"], conn)
                st.success("Query executed successfully!")
                st.dataframe(df, use_container_width=True)
                conn.close()
            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.info("Note: If you are running a SELECT query, make sure the tables exist and have data. You might need to execute CREATE/INSERT statements first.")
    
    with col2:
        if st.button("⚡ Optimize & Analyze", use_container_width=True):
            with st.spinner("Analyzing Query..."):
                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    
                    optimization_prompt = f"""
                    You are an expert Database Administrator. Analyze the following SQL query based on this schema:
                    
                    Schema: 
                    {schema_text}
                    
                    Query: 
                    {st.session_state["last_sql"]}
                    
                    Please provide a brief, professional analysis covering:
                    1. Potential performance bottlenecks.
                    2. Suggested code modifications for better execution speed.
                    3. Recommended columns for indexing to optimize this specific query.
                    """
                    
                    opt_response = client.chat.completions.create(
                        messages=[{"role": "user", "content": optimization_prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.2
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown(opt_response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Optimization Error: {e}")

st.markdown("---")
with st.expander("💬 SQLSync Chat", expanded=False):
    st.write("Ask the assistant for any code modifications, or request adding new conditions (like an extra JOIN or WHERE clause).")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if chat_input := st.chat_input("Discuss the code or schema with the assistant..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)

        with st.spinner("Generating response..."):
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                
                chat_context = f"Current Schema:\n{schema_text}\n\nCurrent Last SQL Query:\n{st.session_state.get('last_sql', 'None')}\n\n"
                messages_payload = [{"role": "system", "content": f"You are an expert database assistant. {chat_context}"}]
                for m in st.session_state.messages:
                    messages_payload.append({"role": m["role"], "content": m["content"]})
                
                chat_response = client.chat.completions.create(
                    messages=messages_payload,
                    model="llama-3.3-70b-versatile",
                    temperature=0.2
                )
                assistant_reply = chat_response.choices[0].message.content
                
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)
            except Exception as e:
                st.error(f"Chat Error: {e}")
