import streamlit as st
from typing import List, Dict, Optional
from llm import get_llm
from rag import RAGSystem

class ChatBot:
    def __init__(self, llm, rag_system: Optional[RAGSystem] = None):
        self.llm = llm
        self.rag_system = rag_system
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        st.session_state.chat_history.append({"role": role, "content": content})

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history."""
        return st.session_state.chat_history

    def clear_chat_history(self):
        """Clear the chat history."""
        st.session_state.chat_history = []

    def get_response(self, user_message: str) -> str:
        """Get a response from the chatbot with enhanced context handling."""
        try:
            # Add user message to history
            self.add_message("user", user_message)

            # Get context from RAG system if available
            context = ""
            if self.rag_system:
                try:
                    rag_response = self.rag_system.query(user_message)
                    if isinstance(rag_response, dict):
                        context = rag_response.get("answer", "")
                        if rag_response.get("web_results"):
                            context += f"\n\nAdditional Information:\n{rag_response['web_results']}"
                    else:
                        context = str(rag_response)
                except Exception as e:
                    st.warning(f"Could not retrieve context: {str(e)}")

            # Create prompt with enhanced context and chat history
            prompt = f"""You are a knowledgeable marketing assistant for Pwani Oil products. 
            Your role is to provide helpful, accurate, and relevant responses about Pwani Oil's products, 
            marketing strategies, and related information.

            Guidelines:
            - Be concise but informative
            - Focus on marketing and product-related information
            - Use professional and engaging language
            - Provide specific examples when relevant
            - If unsure, acknowledge limitations

            Context Information:
            {context}

            Previous Conversation:
            {str(self.get_chat_history()[-5:])}

            User Question: {user_message}
            Assistant: """

            # Get response from LLM with error handling
            try:
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    response = response.content
                response = str(response).strip()

                # Validate response
                if not response:
                    raise ValueError("Empty response received from LLM")

                # Add assistant response to history
                self.add_message("assistant", response)
                return response

            except Exception as llm_error:
                error_msg = f"Error generating response: {str(llm_error)}"
                st.error(error_msg)
                fallback_response = "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
                self.add_message("assistant", fallback_response)
                return fallback_response

        except Exception as e:
            error_msg = f"Error in chat processing: {str(e)}"
            st.error(error_msg)
            fallback_response = "I encountered an error. Please try again or contact support if the issue persists."
            self.add_message("assistant", fallback_response)
            return fallback_response

def create_chat_interface():
    """Create and render the chat interface."""
    st.subheader("💬 Chat with Marketing Assistant")

    # Initialize chatbot if not already done
    if 'chatbot' not in st.session_state:
        google_api_key, openai_api_key = st.session_state.get('api_keys', (None, None))
        llm = get_llm(openai_api_key or google_api_key, "gpt-4", temperature=0.7)
        rag_system = st.session_state.get('rag_system')
        st.session_state.chatbot = ChatBot(llm, rag_system)

    # Display chat messages
    for message in st.session_state.chatbot.get_chat_history():
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about marketing content..."):
        response = st.session_state.chatbot.get_response(prompt)
        with st.chat_message("assistant"):
            st.write(response)

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chatbot.clear_chat_history()
        st.experimental_rerun()