import streamlit as st
import json
import sqlite3
import time
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from groq import Groq

st.set_page_config( page_title="SQLSync", page_icon="https://github.com/Mohamed20Mamdouh/SQL_Gen/blob/main/Head-Edit.png?raw=true")

def style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap');
    
    code, pre, [data-testid="stCodeBlock"] { 
        font-family: 'Fira Code', monospace !important; 
    }
    
    .stApp { background-color: #FAF8F5; max-width: 800px; margin: 0 auto; }
    h1 { color: #D35400; display: flex; justify-content: center; align-items: center; gap: 12px; font-weight: 700; margin-bottom: 40px; }
    label { font-size: 20px !important; color: #5D4037 !important; font-weight: 600 !important; }
    .stTextInput input, .stTextArea textarea { background-color: #FFFFFF !important; border: 2px solid #F39C12 !important; border-radius: 12px !important; padding: 12px !important; box-shadow: 0 4px 12px rgba(211, 84, 0, 0.08) !important; transition: all 0.3s ease-in-out !important; }
    .stTextInput input:focus, .stTextArea textarea:focus { border-color: #D35400 !important; box-shadow: 0 4px 15px rgba(211, 84, 0, 0.2) !important; }
    .stButton button { background: linear-gradient(135deg, #D35400, #E67E22) !important; color: white !important; width: 100%; border-radius: 10px; font-weight: bold; font-size: 16px; border: none; padding: 10px; box-shadow: 0 4px 10px rgba(211, 84, 0, 0.2); transition: 0.3s; }
    .stButton button:hover { background: linear-gradient(135deg, #BA4A00, #D35400) !important; box-shadow: 0 6px 15px rgba(211, 84, 0, 0.35); }
    .stChatMessage [data-testid="stMarkdownContainer"] { direction: rtl; text-align: right; }
    .stChatMessage [data-testid="stCodeBlock"], .stChatMessage pre { direction: ltr !important; text-align: left !important; }
    .stChatMessage code { direction: ltr !important; unicode-bidi: isolate; }
    </style>
    
    <h1><img src="https://github.com/Mohamed20Mamdouh/SQL_Gen/blob/main/Head-Edit.png?raw=true" width="100" style="vertical-align: middle;">SQLSync</h1>
    """, unsafe_allow_html=True) 

style()

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)
        
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

with st.container(border=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        schema_option = st.radio("How do you want to provide the Schema?", ("Upload .sql", "Text"))
    with col2:
        sql_dialect = st.selectbox("SQL Dialect:", ["MySQL", "PostgreSQL", "SQLite", "SQL Server"])    

schema_text = ""
if schema_option == "Upload .sql":
    uploaded_file = st.file_uploader("Upload your .sql file (Required):", type=['sql', 'txt'])
    if uploaded_file:
        schema_text = uploaded_file.read().decode("utf-8")
else:
    schema_text = st.text_area("Paste Your Schema Here (Required):", height=150, placeholder="CREATE TABLE users (id INT, name VARCHAR(50));")

user_input = st.text_input("Enter your SQL query request:", placeholder="Prompt your query...")

if st.button("Generate SQL"):
    if not schema_text.strip():
        st.warning("Please provide your Database Schema (Upload a file or paste text) to proceed.")
    elif not user_input:
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

IMPORTANT INSTRUCTIONS:
- Generate ONLY what is explicitly requested.
- DO NOT add extra clauses like ORDER BY, WHERE, or LIMIT unless the user specifically asks for them.

CRITICAL RULES:
1. If the User Request is NOT related to databases, SQL, or the provided Schema (e.g., asking for recipes, general knowledge, non-SQL programming), YOU MUST STRICTLY REFUSE TO ANSWER.
2. If the request is out of scope, output EXACTLY "-- ERROR: OUT_OF_SCOPE" as the value for the "sql_query" key.
3. Do not generate fake tables or default queries like 'SELECT * FROM users' unless explicitly requested by the user based on the schema.

Database Schema:
{relevant_schema}

User Request: {user_input}

Respond ONLY in valid JSON format with the following keys:
- "sql_query": The generated SQL query string (or the exact error string if out of scope).
- "explanation": You MUST write this explanation in the EXACT SAME LANGUAGE as the User Request. If the request is out of scope, explain briefly why you cannot answer."""
                else:
                    prompt = f"""You are an advanced Text-to-SQL expert specialized in {sql_dialect}. 
Write a standard SQL query for {sql_dialect} based on the request.

CRITICAL RULES:
1. If the User Request is NOT related to databases or SQL (e.g., asking for recipes, general knowledge), YOU MUST STRICTLY REFUSE TO ANSWER.
2. If the request is out of scope, output EXACTLY "-- ERROR: OUT_OF_SCOPE" as the value for the "sql_query" key.

User Request: {user_input}

Respond ONLY in valid JSON format with the following keys:
- "sql_query": The generated SQL query string (or the exact error string if out of scope).
- "explanation": You MUST write this explanation in the EXACT SAME LANGUAGE as the User Request. If the request is out of scope, explain briefly why you cannot answer."""
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    seed=42
                )
                
                result = json.loads(response.choices[0].message.content)
                
                if "-- ERROR: OUT_OF_SCOPE" in result.get("sql_query", ""):
                    is_arabic = any("\u0600" <= c <= "\u06FF" for c in user_input)                    
                    if is_arabic:
                        st.error("عفواً، هذا السؤال خارج نطاق قواعد البيانات. يرجى طرح أسئلة متعلقة بالـ Schema المتاحة أو الـ SQL فقط.")
                    else:
                        st.error("Sorry, this question is out of database scope. Please ask questions related to the provided Schema or SQL only.")
                    if "last_sql" in st.session_state:
                        del st.session_state["last_sql"]
                    if "last_explanation" in st.session_state:
                        del st.session_state["last_explanation"]
                else:
                    st.session_state["last_sql"] = result["sql_query"]
                    st.session_state["last_explanation"] = result.get("explanation", "")
                    
            except Exception as e:
                st.error(f"Error: {e}")

if "last_sql" in st.session_state:
    st.markdown("---")
    st.subheader("(SQL Generation)")
    st.code(st.session_state["last_sql"], language="sql")
    
    st.subheader("💡 Explanation")
    st.write_stream(stream_data(st.session_state["last_explanation"]))
    
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
                    
                    User's Original Request: 
                    {user_input}
                    
                    Please provide a brief, professional analysis covering:
                    1. Potential performance bottlenecks.
                    2. Suggested code modifications for better execution speed.
                    3. Recommended columns for indexing to optimize this specific query.
                    
                    IMPORTANT INSTRUCTION: 
                    You MUST write the entire analysis in the EXACT SAME LANGUAGE as the "User's Original Request". 
                    If the User's Original Request is in Arabic, your analysis MUST be completely in Arabic (except for SQL keywords or column names).
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
                    messages_payload = [{"role": "system", "content": f"You are an expert database assistant. Always reply in the exact same language the user uses (e.g., if the user asks in Arabic, reply in Arabic; if in English, reply in English). {chat_context}"}]
                    for m in st.session_state.messages:
                        messages_payload.append({"role": m["role"], "content": m["content"]})
                    
                    chat_response = client.chat.completions.create(
                        messages=messages_payload,
                        model="llama-3.3-70b-versatile",
                        temperature=0.2
                    )
                    
                    assistant_reply = chat_response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})                
                    with st.chat_message("assistant", avatar="https://github.com/Mohamed20Mamdouh/SQL_Gen/blob/main/Head-Edit.png?raw=true"):
                        st.write_stream(stream_data(assistant_reply))
                        
                except Exception as e:
                    st.error(f"Chat Error: {e}")
