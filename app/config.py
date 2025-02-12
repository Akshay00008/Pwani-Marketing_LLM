# config.py
import streamlit as st
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set higher log level for langchain to suppress embedding logs
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Configure Streamlit page
def configure_streamlit_page():
    st.set_page_config(
        page_title="Pwani Oil Marketing Generator",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# Load API Keys
def load_api_keys():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not google_api_key and not openai_api_key:
        st.error("🔑 No API Keys found. Please set either Gemini_API_KEY or OPENAI_API_KEY in your .env file")
        st.stop()

    return google_api_key, openai_api_key

# Custom CSS for better styling
def load_css():
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )