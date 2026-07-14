import streamlit as st
from groq import Groq
# --- Configuration ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except Exception as e:
    st.error("API Key not found. Please set GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Text-to-SQL Generator", layout="wide")
st.title("📝 Text-to-SQL Generator")

st.markdown("""
    <style>
    html, body, [class*="css"], [class*="st-"] { font-size: 18px !important; }
    .stButton>button {
        font-weight: bold !important;
        font-size: 20px !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        background-color: #D32F2F !important;
        color: white !important;
    }
    .stButton>button:hover { transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# --- Inputs ---
user_schema = st.text_area("Enter your database schema here:", height=200)
user_input = st.text_input("Enter your query:", "")

if st.button("Generate SQL"):
    if not user_schema.strip() or not user_input.strip():
        st.error("Please provide both schema and query.")
    else:
        with st.spinner("Generating..."):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert SQL assistant. Generate clean SQL and a brief explanation."},
                        {"role": "user", "content": f"Schema: {user_schema} \n Query: {user_input}"}
                    ],
                    model="llama-3.1-70b-versatile",
                )
                
                response_text = chat_completion.choices[0].message.content
                st.subheader("Result")
                st.write(response_text)
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
