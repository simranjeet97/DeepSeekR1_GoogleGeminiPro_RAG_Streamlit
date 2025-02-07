import os
os.environ["GOOGLE_API_KEY"] = 
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize DeepSeek R1 model (Ollama)
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.3
)

# Define system prompt for DeepSeek R1
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Function to generate response from DeepSeek R1
def generate_ai_response(user_query):
    prompt_chain = ChatPromptTemplate.from_messages([
        system_prompt,
        HumanMessagePromptTemplate.from_template(user_query)
    ])
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Initialize Google Gemini Pro model
gemini_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

# Function to get response from Google Gemini
def get_gemini_response(user_query):
    return gemini_model.invoke(user_query)

# Streamlit app setup
st.set_page_config(page_title="AI Assistant: DeepSeek & Gemini Pro", page_icon=":brain:", layout="wide")

# Custom CSS styling for a more polished look
st.markdown("""
    <style>
        .main {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #333333;
        }
        .stTextInput textarea {
            color: #ffffff !important;
            background-color: #444444 !important;
        }
        .stSelectbox option {
            background-color: #333333 !important;
            color: white !important;
        }
        .stButton button {
            background-color: #0069d9;
            color: white;
        }   
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("AI Assistant: DeepSeek & Gemini Pro")
st.caption("🚀 Choose your model and ask your question! :brain:")

# Sidebar for model selection with emojis
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_choice = st.selectbox(
        "Select your AI model:",
        ["DeepSeek R1", "Google Gemini Pro"]
    )
    st.divider()
    st.markdown("### Features")
    st.markdown("""
        - 💻 **DeepSeek R1**: Latest LLM that Surpasses open source LLMs, Expert reasoning and coding assistant with debugging capabilities.
        - 🌐 **Google Gemini Pro**: Google state of the art LLM that can do coding, reasoning and provide assistant for any kind of query.
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) and [Google Generative AI](https://genai.google).")

# User input field for querying
user_query = st.text_input("What would you like to ask? :question:")

# Add the submit button
if st.button("Submit"):
    if user_query:
        if model_choice == "DeepSeek R1":
            # Generate response using DeepSeek R1 (Coding Assistant)
            with st.spinner("🤖 Generating coding response..."):
                response = generate_ai_response(user_query)
            st.subheader("DeepSeek R1 Response :computer:")
            st.markdown(f"```python\n{response}\n```")

        elif model_choice == "Google Gemini Pro":
            # Generate response using Google Gemini Pro (General Assistant)
            with st.spinner("🌐 Generating general response..."):
                response = get_gemini_response(user_query)
            st.subheader("Google Gemini Pro Response :rocket:")
            st.write(f"```python\n{response.content}```")

