import streamlit as st
from groq import Groq
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
st.set_page_config(page_title="SQL Assistant", page_icon="🗄️")
st.title("🗄️ SQL Assistant")

#embedding
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
@st.cache_resource(show_spinner=False)
def build_vectorstore(schema_text):
    embeddings = get_embeddings()
    chunks = re.split(r'(?i)(?=CREATE\s+TABLE)', schema_text)
    chunks = [c.strip() for c in chunks if c.strip()] 
    if not chunks:
        return None      
    docs = [Document(page_content=c) for c in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

#Input
sql_dialect = st.selectbox("Select SQL Dialect:", ["PostgreSQL", "MySQL", "SQL Server", "SQLite", "Generic SQL"])

schema_option = st.radio("How do you want to provide the Schema?", ("Upload .sql File", "Write Manually"))
schema_text = ""

if schema_option == "Upload .sql File":
    uploaded_file = st.file_uploader("Upload your .sql file:", type=['sql', 'txt'])
    if uploaded_file:
        schema_text = uploaded_file.read().decode("utf-8")
else:
    schema_text = st.text_area("Paste your Schema here (Optional):", height=200)

user_input = st.text_input("Enter your SQL query request:")

#processing
if st.button("Generate SQL"):
    if not user_input:
        st.warning("Please Enter Your SQL Query: ")
    else:
        with st.spinner("Processing..."):
            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                
                system_prompt = f"""You are an expert Database Administrator. 
                Write precise and highly optimized {sql_dialect} queries. 
                Respond ONLY in valid JSON format with exactly two keys: 'sql_query' and 'explanation'."""

                prompt = ""
                if schema_text.strip():
                    vectorstore = build_vectorstore(schema_text)
                    if vectorstore:
                        relevant_docs = vectorstore.similarity_search(user_input, k=2)
                        relevant_schema = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        prompt = f"Schema:\n{relevant_schema}\n\nUser Query: {user_input}\n\nBased ONLY on the schema above, generate the {sql_dialect} query."
                    else:
                        prompt = f"User Query: {user_input}\nGenerate a standard {sql_dialect} query."
                else:
                    prompt = f"No Schema provided. Generate a standard {sql_dialect} query for: {user_input}"
                
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    response_format={"type": "json_object"}
                )                
                raw_content = response.choices[0].message.content.strip()
                if raw_content.startswith("```json"):
                    raw_content = raw_content[7:-3].strip()
                elif raw_content.startswith("```"):
                    raw_content = raw_content[3:-3].strip()
                    
                result = json.loads(raw_content)
                
                st.subheader("Generated SQL")
                st.code(result.get("sql_query", "-- No SQL generated"), language="sql")
                st.subheader("Explanation")
                st.write(result.get("explanation", "No explanation provided."))
                
            except Exception as e:
                st.error(f"Error: {e}")
