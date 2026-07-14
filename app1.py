import streamlit as st
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="Text-to-SQL RAG", layout="wide")
st.title("📝 Text-to-SQL Generator (RAG Powered)")

user_schema = st.text_area("Enter your database schema here:", height=200)
user_input = st.text_input("Enter your query:", "")

if st.button("Generate SQL"):
    if not user_schema.strip() or not user_input.strip():
        st.error("Please provide both schema and query.")
    else:
        with st.spinner("Searching schema and generating SQL..."):
            try:
                embeddings = get_embeddings()
                chunks = [d.strip() for d in user_schema.split("CREATE TABLE")]
                chunks = [f"CREATE TABLE {c}" for c in chunks if c.strip()]
                
                docs = [Document(page_content=c) for c in chunks]
                vectorstore = FAISS.from_documents(docs, embeddings)
                

                retrieved_docs = vectorstore.similarity_search(user_input, k=2)
                relevant_schema = "\n".join([doc.page_content for doc in retrieved_docs])

                
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert SQL assistant. Use only the provided relevant schema to generate the SQL query."},
                        {"role": "user", "content": f"Relevant Schema:\n{relevant_schema}\n\nQuestion: {user_input}"}
                    ],
                    model="llama-3.1-8b-instant",
                )
                
                st.subheader("Generated SQL Query")
                st.write(response.choices[0].message.content)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
