import os
os.environ["GOOGLE_API_KEY"] = 
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Setup environment variables (API keys, etc.)

# Initialize DeepSeek R1 model (Ollama)
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.3
)

# Define system prompt for DeepSeek R1
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "using retrieved context. If no relevant context is found, make an educated guess."
)

def get_conversational_chain(anime_name):
    """
    Creates a conversational chain for discussing anime fan theories using LangChain and Google Generative AI.
    Dynamically adapts the prompt based on the anime name.
    """
    prompt_template = f"""
    You are an expert assistant for discussing fan theories and answering questions about the anime '{anime_name}'. 
    Provide well-informed and detailed responses based on the provided context. 
    If the answer isn't in the context, offer plausible theories or general insights based on common tropes or themes relevant to '{anime_name}'. 
    If no relevant information can be derived, respond with: "I'm sorry, I don't have enough information to answer that."

    Context:\n{{context}}\n
    Question:\n{{question}}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to generate response from DeepSeek R1 (using RAG)
def generate_ai_response_deepseek(user_query, vector_store):
    # Retrieve relevant documents using similarity search
    retrieved_docs = vector_store.similarity_search(user_query, k=5)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Build prompt sequence with retrieved context for DeepSeek R1
    prompt_chain = ChatPromptTemplate.from_messages([
        system_prompt,
        HumanMessagePromptTemplate.from_template(f"Context:\n{context}\n\nQuestion:\n{user_query}")
    ])
    
    # Process response
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Function to get response from Google Gemini Pro (using RAG)
def generate_ai_response_gemini(user_query, vector_store):
    # Retrieve relevant documents using similarity search
    retrieved_docs = vector_store.similarity_search(user_query, k=5)
    conversational_chain = get_conversational_chain("Solo Leveling")
    # Generate response using Google Gemini with context

    response = conversational_chain.run({"input_documents": retrieved_docs, "question": user_query})
    return response

# Function to load and process documents for RAG
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

# Function to chunk and vectorize documents
@st.cache_resource
def get_vector_store(folder_path):
    # This function will only be run once unless the documents change
    documents = load_documents_from_folder(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local("faiss_index")  # Save locally for future use
    return vector_store

# Set dataset path
file_path = "/Users/simranjeetsingh1497/Downloads/RAG+LLM_Projects/GenAnime/anime_dataset"  # Specify the correct path to your folder of documents

# Streamlit app setup
st.set_page_config(page_title="AI Assistant: DeepSeek & Gemini Pro with RAG: AnimeBot", page_icon=":brain:", layout="wide")

# Custom CSS styling for a polished look
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
st.title("AI Assistant: DeepSeek & Gemini Pro with RAG")
st.caption("🚀 Choose your model and ask your question! :brain:")

# Sidebar for model selection
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
        - 🌐 **Google Gemini Pro**: Google state of the art LLM that can do coding, reasoning, and provide assistant for any kind of query.
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) and [Google Generative AI](https://genai.google).")

# User input field for querying
user_query = st.text_input("What would you like to ask? :question:")

# Add the submit button
if st.button("Submit"):
    with st.spinner("⏳ Checking if vector DB exists..."):
        try:
            # Try loading vector store, if it fails, we'll create it
            vector_store = get_vector_store(file_path)
            st.success("Vector DB loaded successfully.")
        except:
            st.error("Vector DB creation failed or doesn't exist.")

    if user_query:
        if model_choice == "DeepSeek R1":
            # Generate response using DeepSeek R1 (Coding Assistant)
            with st.spinner("🤖 Generating response..."):
                response = generate_ai_response_deepseek(user_query, vector_store)
            st.subheader("DeepSeek R1 Response :computer:")
            st.markdown(response)

        elif model_choice == "Google Gemini Pro":
            # Generate response using Google Gemini Pro (General Assistant)
            with st.spinner("🌐 Generating response..."):
                response = generate_ai_response_gemini(user_query, vector_store)
            st.subheader("Google Gemini Pro Response")
            st.write(f"**Response:** {response}")