import os
import logging
from langgraph.graph import StateGraph, END
from llm import get_llm
from app.agent_state import AgentState
from agent_tools import (
    rag_tool_function,
    web_search_tool_function,
    generate_content_tool_function,
    generate_image_tool_function
)

def llm_tool_choice_function(state: AgentState):
    """Decides which tool to use next using an LLM."""
    from agent_tools import TOOL_DESCRIPTIONS

    # Dynamically build available tools based on state flags
    available_tools = {}

    if state.use_rag:
        available_tools["RAG Tool"] = "rag_tool"
    if state.use_search_engine:
        available_tools["Web Search Tool"] = "web_search_tool"

    # Always include content generation
    available_tools["Generate Content Tool"] = "generate_content_tool"

    if state.generate_image_checkbox:
        available_tools["Generate Image Tool"] = "generate_image_tool"

    available_tools["Final Output"] = "content_output"

    tool_descriptions_str = "\n".join(
        [f"{name}: {TOOL_DESCRIPTIONS[tool_id]}"
         for name, tool_id in available_tools.items()if tool_id != 'content_output']
    )

    prompt_text = f"""
    You are an AI agent designed to generate marketing content. You have access to several tools:
    {tool_descriptions_str}

    Your goal is to generate high-quality marketing content based on the user's request and the current campaign details.
    You need to decide which tool to use next to best achieve this goal.

    Current Campaign Details:
    Campaign Name: {state.campaign_name}
    Brand: {state.brand}
    Product Category: {state.product_category}
    Output Format: {state.output_format}
    Specific Instructions: {state.specific_instructions}
    Use RAG: {'Yes' if state.use_rag else 'No'}
    Use Web Search: {'Yes' if state.use_search_engine else 'No'}
    Generate Image: {'Yes' if state.generate_image_checkbox else 'No'}

    Current State and Results:
    RAG Context: {state.rag_context if state.rag_context else 'Not yet retrieved'}
    Web Search Results: {state.search_results if state.search_results else 'Not yet performed'}
    Generated Content: { 'Yes' if state.generated_content else 'No'}
    Generated Image: {'Yes' if state.image_url else 'No'}

    Consider the user's request, the available tools, and the current state of the content generation process.
    Which tool should you use next?  If you have all the necessary information and have generated the content and image (if requested), choose "Final Output".

    Choose from: [{', '.join(available_tools.keys())}]
    Respond with just the name of the tool to use next, or "Final Output" if you are ready to output the content.
    """

    llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0) # Use GPT-4 for decision-making
    if not llm:
        return {"next": "content_output"} # Default to output if LLM fails for decision - Return dict

    response = llm.invoke(prompt_text)
    tool_choice = response.content.strip()

    # Map tool name to node key
    tool_node_map = {
        "RAG Tool": "rag_tool",
        "Web Search Tool": "web_search_tool",
        "Generate Content Tool": "generate_content_tool",
        "Generate Image Tool": "generate_image_tool",
        "Final Output": "content_output"
    }

    # Validate and map the tool choice to the graph node
    # chosen_node = tool_node_map.get(tool_choice, "content_output") # Default to content_output if invalid choice

    # logging.info(f"LLM Tool Choice: {tool_choice} -> Node: {chosen_node}")
    # return {"next": chosen_node} # Ensure dictionary is returned
    chosen_node = tool_node_map.get(tool_choice, "content_output")
    logging.info(f"LLM Tool Choice: {tool_choice} -> Node: {chosen_node}")
    return_value = {"next": chosen_node} # Store in a variable for printing
    print(f"DEBUG: llm_tool_choice_function return type: {type(return_value)}, value: {return_value}") # ADD THIS PRINT
    return return_value
# Output Node - Just passes state through for final output
def content_output(state):
    """Output node to finalize content generation."""
    return {"final_state": state} # Just return the state

# LangGraph Workflow Definition
def create_agent_workflow():
    """Creates the LangGraph agent workflow."""
    builder = StateGraph(AgentState)

    # Add nodes (same as before)
    builder.add_node("rag_tool", rag_tool_function)
    builder.add_node("web_search_tool", web_search_tool_function)
    builder.add_node("generate_content_tool", generate_content_tool_function)
    builder.add_node("generate_image_tool", generate_image_tool_function)
    builder.add_node("content_output", content_output)
    builder.add_node("tool_choice", llm_tool_choice_function)

    # Set up edges correctly
    builder.set_entry_point("tool_choice")  # Start with decision node

    # Corrected add_conditional_edges - Mapping node keys to node keys
    builder.add_conditional_edges(
        "tool_choice",
        {
            "rag_tool": "rag_tool",
            "web_search_tool": "web_search_tool",
            "generate_content_tool": "generate_content_tool",
            "generate_image_tool": "generate_image_tool",
            "content_output": "content_output"
        }
    )

    # Add edges back to decision node - Use node *keys* as strings
    builder.add_edge("rag_tool", "tool_choice")
    builder.add_edge("web_search_tool", "tool_choice")
    builder.add_edge("generate_content_tool", "tool_choice")
    builder.add_edge("generate_image_tool", "tool_choice")
    builder.add_edge("content_output", "tool_choice") # Corrected edge to tool_choice - important!

    # End condition
    builder.add_edge("content_output", END)

    return builder.compile()