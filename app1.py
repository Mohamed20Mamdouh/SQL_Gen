import streamlit as st
from groq import Groq
import json
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def style():
    st.markdown("""
    <style>
        .stApp { background-color: #F0F8FF; max-width: 800px; margin: 0 auto; }
        h1 { color: #1A3A5F; text-align: left; }
        label { font-size: 18px !important; color: #1A3A5F !important; }
        .stTextInput input, .stTextArea textarea { border: 2px solid #1A3A5F !important; }
        .stButton button { background: #1A3A5F !important; color: white !important; width: 100%; }
    </style>

<h1>🗄️ SQL Intelligent Assistant</h1>
""", unsafe_allow_html=True)
style()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

schema_option = st.radio("How do you want to provide the Schema?", ("Upload .sql File", "Write Manually"))
schema_text = ""

if schema_option == "Upload .sql File":
    uploaded_file = st.file_uploader("Upload your .sql file:", type=['sql', 'txt'])
    if uploaded_file:
        schema_text = uploaded_file.read().decode("utf-8")
else:
    schema_text = st.text_area("Paste your Schema here (Optional):", height=200)

user_input = st.text_input("Enter your SQL query request:")

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
                    relevant_schema = "\n".join([doc.page_content for doc in vectorstore.similarity_search(user_input, k=2)])
                    
                    prompt = f"Schema: {relevant_schema}\n\nQuery: {user_input}\nRespond ONLY in JSON format with 'sql_query' and 'explanation'."
                else:
                    prompt = f"No Schema provided. Generate a standard SQL query for: {user_input}\nRespond ONLY in JSON format with 'sql_query' and 'explanation'."
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b",
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                st.subheader("Generated SQL")
                st.code(result["sql_query"], language="sql")
                st.subheader("Explanation")
                st.write(result.get("explanation", ""))
                
            except Exception as e:
                st.error(f"Error: {e}")
