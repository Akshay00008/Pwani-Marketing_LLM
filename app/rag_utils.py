# rag_utils.py
from typing import Optional
from rag import RAGSystem
from llm import get_llm
from agent_state import AgentState  # Relative import
import os
# Function to initialize RAG system from state (important for LangGraph context)
def state_to_rag_system(state: AgentState) -> Optional[RAGSystem]:
    """Initializes RAG system using LLM from the agent state."""
    llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0) # Or use model from state if needed
    if llm:
        return RAGSystem(llm)
    return None