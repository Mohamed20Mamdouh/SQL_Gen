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
    .stApp { background-color: #F8F9FA; font-family: 'Segoe UI', sans-serif; }
    h1 { color: #0078D4; font-weight: bold; }
    .stButton>button { background-color: #0078D4; color: white; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

style()

col1, col2 = st.columns([1, 6])
with col1:
    st.image("sql_logo.png", width=80) 
with col2:
    st.title("SQL Server Intelligent Assistant")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def read_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    return file.read().decode("utf-8")

uploaded_file = st.file_uploader("Upload Schema File (.txt or .pdf)", type=['txt', 'pdf'])
user_input = st.text_input("Enter your SQL query request:")

if st.button("Generate SQL"):
    if not uploaded_file or not user_input:
        st.warning("Please upload a file and enter a query.")
    else:
        with st.spinner("Processing..."):
            try:
                schema_text = read_file(uploaded_file)
                
                embeddings = get_embeddings()
                chunks = [f"CREATE TABLE {c.strip()}" for c in schema_text.split("CREATE TABLE") if c.strip()]
                docs = [Document(page_content=c) for c in chunks]
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                relevant_schema = "\n".join([doc.page_content for doc in vectorstore.similarity_search(user_input, k=2)])
                
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                prompt = f"Schema: {relevant_schema}\n\nQuery: {user_input}\nRespond ONLY in JSON format with 'sql_query' and 'explanation'."
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-70b-versatile",
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                st.subheader("Generated SQL")
                st.code(result["sql_query"], language="sql")
                st.subheader("Explanation")
                st.write(result["explanation"])
                
            except Exception as e:
                st.error(f"Error: {e}")
