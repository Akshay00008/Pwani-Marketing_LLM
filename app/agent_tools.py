# agent_tools.py
import os
from typing import Optional, TYPE_CHECKING
from llm import get_llm, search_tool
from rag import RAGSystem
from prompt import create_prompt_template
from utils import generate_content_with_retries
from app.agent_state import AgentState

# Define tool descriptions for LLM-based decision making
TOOL_DESCRIPTIONS = {
    "rag_tool": "Retrieves relevant information from a knowledge base to provide context for content generation. Use this tool when you need to answer questions about the brand, product, or marketing context based on internal documents.",
    "web_search_tool": "Searches the internet for up-to-date information, market trends, competitor analysis, or general knowledge relevant to the campaign. Use this tool when you need to gather current information from the web.",
    "generate_content_tool": "Generates the main marketing content (social media post, email, marketing copy) based on all provided information and context. This tool is essential for creating the final output.",
    "generate_image_tool": "Generates a product image to accompany the marketing content. Use this tool if an image is requested to make the content more visually appealing."
}


# Define tools as functions for LangGraph - RAG Tool
def rag_tool_function(state: 'AgentState'):  # Keep forward reference in type hint
    """Uses the RAG system to retrieve relevant context."""
    if state.use_rag and state.rag_query:
        rag_system = state_to_rag_system(state)
        if rag_system:
            context_query = f"""
            Brand: {state.brand}
            Product: {state.sku if state.sku else 'N/A'}
            Category: {state.product_category}
            Query: {state.rag_query}
            """
            rag_context = rag_system.query(context_query)
            return {"rag_context": rag_context}
    return {"rag_context": "RAG was not used or no query provided."}


# Define tools as functions for LangGraph - Web Search Tool
def web_search_tool_function(state: 'AgentState'): # Keep forward reference in type hint
    """Uses the web search tool to get up-to-date information."""
    if state.use_search_engine and state.search_engine_query:
        search_results = search_tool.run(state.search_engine_query)
        return {"search_results": search_results}
    return {"search_results": "Web search was not used or no query provided."}


# Define tools as functions for LangGraph - Content Generation Tool
def generate_content_tool_function(state: 'AgentState'): # Keep forward reference in type hint
    """Generates marketing content based on the current state."""
    llm = get_llm(os.getenv('OPENAI_API_KEY'), state.model_name_select, state.temperature_slider, state.top_p_slider)
    if not llm:
        return {"generated_content": "Failed to initialize LLM."}

    prompt = create_prompt_template(
        instruction="Generate marketing campaign content",
        output_format=state.output_format,
        use_search_engine=state.use_search_engine,
        search_engine_prompt_template=state.search_engine_query
    )

    input_vars = state.dict()
    workflow_result = generate_content_with_retries(llm, prompt, input_vars, state.output_format, state.use_search_engine, state.search_engine_query, state.use_rag, state_to_rag_system(state))
    if workflow_result:
        return {"generated_content": workflow_result}
    else:
        return {"generated_content": "Content generation failed."}


# Define tools as functions for LangGraph - Image Generation Tool
def generate_image_tool_function(state: 'AgentState'): # Keep forward reference in type hint
    """Generates a product image if requested."""
    if state.generate_image_checkbox and state.brand and state.image_style:
        from image import generate_product_image
        description = ""
        if isinstance(state.generated_content, dict) and 'MarketingContent' in state.generated_content:
            content_data = state.generated_content['MarketingContent']
            description = f"{content_data.get('headline', '')}. {content_data.get('body', '')}"
        elif isinstance(state.generated_content, dict) and 'SocialMediaContent' in state.generated_content:
            content_data = state.generated_content['SocialMediaContent']
            description = f"{content_data.get('post_text', '')}"
        elif isinstance(state.generated_content, dict) and 'EmailContent' in state.generated_content:
            content_data = state.generated_content['EmailContent']
            description = f"{content_data.get('subject_line', '')}. {content_data.get('body', '')}"
        elif isinstance(state.generated_content, str):
            description = state.generated_content[:500]

        image_url = generate_product_image(
            state.brand,
            description,
            state.image_style,
            os.getenv('OPENAI_API_KEY')
        )
        if image_url:
            return {"image_url": image_url}
        else:
            return {"image_url": "Image generation failed."}
    return {"image_url": None}

# Function to initialize RAG system from state
def state_to_rag_system(state: 'AgentState') -> Optional[RAGSystem]: # Keep forward reference in type hint
    """Initializes RAG system using LLM from the agent state."""
    llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0)
    if llm:
        return RAGSystem(llm)
    return None