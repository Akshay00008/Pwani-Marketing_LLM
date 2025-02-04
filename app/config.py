import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def load_api_keys():
    """Loads API keys from environment variables or user input."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Add API key input fields if not found in environment variables
    if not google_api_key:
        google_api_key = st.text_input("Enter your Google API Key:", type="password")
        if not google_api_key:
            st.warning("Google API Key is missing. Functionalities using Google models will be disabled.")
    
    if not openai_api_key:
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if not openai_api_key:
            st.warning("OpenAI API Key is missing. Functionalities using OpenAI models will be disabled.")
    
    return google_api_key, openai_api_key

def configure_streamlit_page():
    """Configures Streamlit page settings."""
    st.set_page_config(
        page_title="Pwani Oil Marketing Content Generator",
        page_icon="🌟",
        layout="wide",
        menu_items={
            'Get Help': 'https://www.example.com', # Replace with your help URL
            'Report a bug': "https://www.example.com", # Replace with your bug report URL
            'About': """
             # Pwani Oil Marketing Content Generator
             This app generates marketing content for Pwani Oil brands using AI.
             """
        }
    )

def load_css():
    """Load custom CSS styles from the style.css file."""
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Style file not found. Using default styling.")
# You might have other configuration settings here, like file paths, etc.