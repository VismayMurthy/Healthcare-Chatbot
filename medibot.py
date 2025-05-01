import os
import streamlit as st
from gtts import gTTS
import base64
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Page configuration
st.set_page_config(
    page_title="MediBot - Your Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
    st.session_state.custom_primary = "#1890ff"  # Default primary color
    st.session_state.custom_bg = "#f0f8ff"      # Default background
    st.session_state.custom_text = "#000000"    # Default text color


# Function to apply theme CSS
def apply_theme():
    if st.session_state.theme == "light":
        st.markdown("""
        <style>
            .main {
                background-color: #f8f9fa;
            }
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .chat-message {
                padding: 1.5rem;
                border-radius: 0.8rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
            }
            .chat-message.user {
                background-color: #e6f7ff;
                border-left: 5px solid #1890ff;
            }
            .chat-message.bot {
                background-color: #f6ffed;
                border-left: 5px solid #52c41a;
            }
            .chat-message .message-content {
                margin-top: 0.5rem;
            }
            .stButton>button {
                background-color: #1890ff;
                color: white;
                border-radius: 0.5rem;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }
            .sidebar .sidebar-content {
                background-color: #f0f2f5;
            }
            .header-container {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 1rem;
                background-color: #1890ff;
                border-radius: 0.8rem;
                margin-bottom: 2rem;
            }
            .header-text {
                color: white;
                margin: 0;
            }
            .options-container {
                background-color: #f0f8ff;
                border-radius: 0.8rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid #d9e6f2;
            }
        </style>
        """, unsafe_allow_html=True)
    elif st.session_state.theme == "dark":
        st.markdown("""
       <style>
        .main {
            background-color: #121212;
            color: #cfcfcf;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message.user {
            background-color: #0d2a45;
            border-left: 5px solid #1890ff;
            color: #a3d8ff;
        }
        .chat-message.bot {
            background-color: #0d2a45;
            border-left: 5px solid #0d2a45;
            color: #b7f0b1;
        }
        .chat-message .message-content {
            margin-top: 0.5rem;
        }
        .stButton>button {
            background-color: #1890ff;
            color: white;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            background-color: #1890ff;
            border-radius: 0.8rem;
            margin-bottom: 2rem;
        }
        .header-text {
            color: white;
            margin: 0;
        }
        .options-container {
            background-color: #1e1e2e;
            border-radius: 0.8rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #2a2a3a;
        }
    </style>
        """, unsafe_allow_html=True)
    elif st.session_state.theme == "custom":
        st.markdown(f"""
        <style>
            .main {{
                background-color: {st.session_state.custom_bg};
                color: {st.session_state.custom_text};
            }}
            .stApp {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.8rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
            }}
            .chat-message.user {{
                background-color: #e6f7ff; /* Or customize */
                border-left: 5px solid {st.session_state.custom_primary};
                color: {st.session_state.custom_text};
            }}
            .chat-message.bot {{
                background-color: #f6ffed; /* Or customize */
                border-left: 5px solid #52c41a; /* Or customize */
                color: {st.session_state.custom_text};
            }}
            .chat-message .message-content {{
                margin-top: 0.5rem;
            }}
            .stButton>button {{
                background-color: {st.session_state.custom_primary};
                color: white;
                border-radius: 0.5rem;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }}
            .sidebar .sidebar-content {{
                background-color: #f0f2f5; /* Or customize */
            }}
            .header-container {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 1rem;
                background-color: {st.session_state.custom_primary};
                border-radius: 0.8rem;
                margin-bottom: 2rem;
            }}
            .header-text {{
                color: white;
                margin: 0;
            }}
            .options-container {{
                background-color: #f0f8ff; /* Or customize */
                border-radius: 0.8rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid #d9e6f2; /* Or customize */
            }}
        </style>
        """, unsafe_allow_html=True)


@st.cache_resource
def get_vectorstore():
    """Load and cache the FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    """Create a prompt template for the LLM."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    """Initialize the LLM with HuggingFace."""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN,
                      "max_length": "512"}
    )
    return llm


def text_to_audio(text):
    """Convert text response to audio."""
    tts = gTTS(text)
    audio_file_path = "response.mp3"
    tts.save(audio_file_path)

    with open(audio_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    # Clean up the file after reading
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    return audio_bytes


def display_chat_message(role, content):
    """Display a chat message with better styling."""
    if role == "user":
        st.markdown(
            f'<div class="chat-message user"><div class="message-content">👤 <strong>You:</strong><br>{content}</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="chat-message bot"><div class="message-content">🏥 <strong>MediBot:</strong><br>{content}</div></div>',
            unsafe_allow_html=True)


def main():
    apply_theme()  # Apply initial theme

    # Header
    st.markdown(
        '<div class="header-container"><h1 class="header-text">🏥 MediBot - Your Healthcare Assistant</h1></div>',
        unsafe_allow_html=True)

    # Add only Disclaimer and About in sidebar
    with st.sidebar:
        st.markdown("### Theme Selection")
        theme_choice = st.radio("Choose Theme",
                                 ["Light", "Dark", "Custom"],
                                 index=0)

        if theme_choice:
            if theme_choice.lower() != st.session_state.theme:
                st.session_state.theme = theme_choice.lower()
                st.rerun()  # Apply new theme

        if theme_choice == "Custom":
            st.markdown("### Custom Theme Options")
            st.session_state.custom_primary = st.color_picker("Primary Color", st.session_state.custom_primary)
            st.session_state.custom_bg = st.color_picker("Background Color", st.session_state.custom_bg)
            st.session_state.custom_text = st.color_picker("Text Color", st.session_state.custom_text)

        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.info("""
        This chatbot provides general medical information and is not a substitute for professional medical advice. 

        Always consult your doctor for any health concerns. Do not delay seeking medical advice because of something you read here.
        """)

        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        MediBot is powered by:
        - Mistral-7B LLM
        - FAISS Vector Store
        - Custom medical knowledge base
        - It is an RAG(Retrieval-Augmented Generation) based model
        """)

    # Create main page layout
    # Top row for options
    st.markdown('<div class="options-container">', unsafe_allow_html=True)
    options_col1, options_col2, options_col3 = st.columns(3)

    with options_col1:
        # Voice option
        enable_voice = st.toggle("Enable Voice Response", value=True)

    with options_col2:
        # Model selection (for future expansion)
        model_option = st.selectbox(
            "Select Model",
            ["Mistral-7B", "Future Model 1", "Future Model 2"],
            index=0,
            disabled=True
        )

    with options_col3:
        # Chat history management
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Main chat area
    chat_container = st.container()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(message['role'], message['content'])

    # User input
    user_input = st.chat_input("Ask your health question here...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        # Display user message immediately
        with chat_container:
            display_chat_message('user', user_input)

        # Show loader while processing
        with st.spinner("MediBot is thinking..."):
            try:
                # Get environment variables
                HF_TOKEN = os.environ.get("HF_TOKEN")

                # Define prompt template
                CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer the user's health question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Don't provide anything out of the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

                # Load vector store
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                else:
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )

                    # Get response
                    response = qa_chain.invoke({'query': user_input})
                    result = response["result"]
                    source_documents = response["source_documents"]

                    # Add bot response to history
                    st.session_state.messages.append({'role': 'assistant', 'content': result})

                    # Display bot response
                    with chat_container:
                        display_chat_message('assistant', result)

                    # Convert to audio if enabled
                    if enable_voice:
                        audio_bytes = text_to_audio(result)
                        st.audio(audio_bytes, format='audio/mp3')

            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()