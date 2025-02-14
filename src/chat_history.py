import streamlit as st
import logging
from typing import Optional, Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from llm import get_llm
from rag import RAGSystem
from langchain_community.document_loaders import TextLoader

logger = logging.getLogger(__name__)

def clear_chat_history() -> None:
    """Clears all chat messages and resets conversation memory.
    
    This function removes all messages from the session state and clears
    the conversation memory if available.
    """
    try:
        if "messages" in st.session_state:
            logger.info("Clearing chat messages from session state")
            st.session_state.messages = []
            
        if "conversation" in st.session_state:
            if hasattr(st.session_state.conversation.memory, "clear"):
                logger.info("Clearing conversation memory")
                st.session_state.conversation.memory.clear()
                
        st.success("Chat history cleared successfully!")
        logger.info("Chat history cleared successfully")
        
    except Exception as e:
        error_msg = f"Error clearing chat history: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)

def delete_message(index: int) -> None:
    """Deletes a specific message from the chat history.
    
    Args:
        index: The index of the message to delete.
    """
    try:
        if "messages" not in st.session_state:
            logger.warning("No messages found in session state")
            return
            
        if not 0 <= index < len(st.session_state.messages):
            logger.warning(f"Invalid message index: {index}")
            return
            
        # Remove the message
        message = st.session_state.messages.pop(index)
        logger.info(f"Deleted message at index {index}: {message}")
        
        # Clean up conversation memory if needed
        if "conversation" in st.session_state:
            if hasattr(st.session_state.conversation.memory, "chat_memory"):
                if len(st.session_state.conversation.memory.chat_memory.messages) > index:
                    st.session_state.conversation.memory.chat_memory.messages.pop(index)
                    logger.info(f"Removed corresponding memory entry at index {index}")
                    
        st.success("Message deleted successfully!")
        
    except Exception as e:
        error_msg = f"Error deleting message: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)

def initialize_rag_system(openai_api_key: str) -> Optional[RAGSystem]:
    """Initialize RAG system with enhanced error handling and caching.
    
    Args:
        openai_api_key: The OpenAI API key for LLM initialization.
        
    Returns:
        Optional[RAGSystem]: The initialized RAG system or None if initialization fails.
    """
    try:
        if 'rag_system' not in st.session_state:
            logger.info("Initializing new RAG system")
            llm = get_llm(openai_api_key, "gpt-4", temperature=0)
            
            if not llm:
                raise ValueError("Failed to initialize LLM")
                
            st.session_state.rag_system = RAGSystem(llm)
            logger.info("RAG system instance created successfully")
            
            # Load knowledge base if vector store not initialized
            if not hasattr(st.session_state.rag_system, 'vector_store') or \
               st.session_state.rag_system.vector_store is None:
                   
                st.info("🔄 Loading knowledge base...")
                try:
                    loader = TextLoader("/Users/vishalroy/Downloads/ContentGenApp/cleaned_cleaned_output.txt")
                    documents = loader.load()
                    logger.info(f"Loaded {len(documents)} documents from knowledge base")
                    
                    if st.session_state.rag_system.ingest_documents(documents):
                        st.success("✨ RAG system initialized successfully")
                        logger.info("Documents ingested successfully into RAG system")
                    else:
                        error_msg = "Failed to ingest documents into RAG system"
                        logger.warning(error_msg)
                        st.warning(f"{error_msg} - will proceed without context")
                        
                except Exception as doc_error:
                    error_msg = f"Error loading documents: {str(doc_error)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    raise
            else:
                st.info("✨ Using existing RAG knowledge base")
                logger.info("Using cached RAG knowledge base")
                
        return st.session_state.rag_system
        
    except Exception as e:
        error_msg = f"RAG system initialization failed: {str(e)}"
        logger.error(error_msg)
        st.error(f"{error_msg} - will proceed without context")
        
        if 'rag_system' in st.session_state:
            del st.session_state.rag_system
        return None