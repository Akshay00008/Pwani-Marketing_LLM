# import streamlit as st
# import time
# import logging  # Add this import
# import re  # Add this import
# import os  # Add this import
# from config import configure_streamlit_page, load_api_keys, load_css
# from data import BRAND_OPTIONS, SocialMediaContent, EmailContent, MarketingContent
# from prompt import create_prompt_template
# from llm import get_llm, search_tool  # Import search_tool
# from utils import validate_inputs, save_content_to_file, load_campaign_template, validate_date_range
# from image import generate_product_image, save_generated_image
# import json
# from datetime import datetime
# from langchain_community.document_loaders import TextLoader
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode, FunctionNode
# from typing import Dict, Optional, Tuple, Union, List
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# from langchain_core.tools import Tool
# from langchain_core.runnables import RunnableConfig
# from langchain.pydantic_v1 import BaseModel, Field
# from langchain.agents.format_scratchpad import format_to_openai_function_messages
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain_core.agents import AgentContext
# from langchain.agents import AgentExecutor
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory
# from rag import RAGSystem # Import RAGSystem
# from langchain.output_parsers import PydanticOutputParser  # Add this import


# # Define the Agent State -  Crucial for LangGraph
# class AgentState(BaseModel):
#     campaign_name: Optional[str] = Field(None)
#     promotion_link: Optional[str] = Field(None)
#     previous_campaign_reference: Optional[str] = Field(None)
#     sku: Optional[str] = Field(None)
#     product_category: Optional[str] = Field(None)
#     campaign_date_range: Optional[str] = Field(None)
#     age_range: Optional[str] = Field(None)
#     gender: Optional[str] = Field(None)
#     income_level: Optional[str] = Field(None)
#     region: Optional[str] = Field(None)
#     urban_rural: Optional[str] = Field(None)
#     specific_instructions: Optional[str] = Field(None)
#     brand: Optional[str] = Field(None)
#     tone_style: Optional[str] = Field(None)
#     output_format: Optional[str] = Field(None)
#     rag_query: Optional[str] = Field(None)
#     use_rag: bool = Field(False)
#     use_search_engine: bool = Field(False)
#     search_engine_query: Optional[str] = Field(None)
#     rag_context: Optional[str] = Field(None)
#     search_results: Optional[str] = Field(None)
#     generated_content: Optional[Union[str, SocialMediaContent, EmailContent, MarketingContent, dict]] = Field(None)
#     image_url: Optional[str] = Field(None)
#     image_style: Optional[str] = Field(None)
#     generate_image_checkbox: bool = Field(False) # Add generate_image_checkbox to state
#     model_name_select: Optional[str] = Field(None) # Add model_name_select to state
#     temperature_slider: Optional[float] = Field(None) # Add temperature_slider to state
#     top_p_slider: Optional[float] = Field(None) # Add top_p_slider to state
#     intermediate_steps: List[Tuple] = Field(default_factory=list)
#     messages: List[BaseMessage] = Field(default_factory=list) # Conversation history for agent


# # Define tool descriptions for LLM-based decision making
# TOOL_DESCRIPTIONS = {
#     "rag_tool": "Retrieves relevant information from a knowledge base to provide context for content generation. Use this tool when you need to answer questions about the brand, product, or marketing context based on internal documents.",
#     "web_search_tool": "Searches the internet for up-to-date information, market trends, competitor analysis, or general knowledge relevant to the campaign. Use this tool when you need to gather current information from the web.",
#     "generate_content_tool": "Generates the main marketing content (social media post, email, marketing copy) based on all provided information and context. This tool is essential for creating the final output.",
#     "generate_image_tool": "Generates a product image to accompany the marketing content. Use this tool if an image is requested to make the content more visually appealing."
# }


# # Define tools as functions for LangGraph - RAG Tool
# def rag_tool_function(state: AgentState):
#     """Uses the RAG system to retrieve relevant context."""
#     if state.use_rag and state.rag_query:
#         rag_system = state_to_rag_system(state)  # Get RAG system from state
#         if rag_system:
#             context_query = f"""
#             Brand: {state.brand}
#             Product: {state.sku if state.sku else 'N/A'}
#             Category: {state.product_category}
#             Query: {state.rag_query}
#             """
#             rag_context = rag_system.query(context_query)  # Corrected method call
#             return {"rag_context": rag_context}
#     return {"rag_context": "RAG was not used or no query provided."}


# # Define tools as functions for LangGraph - Web Search Tool
# def web_search_tool_function(state: AgentState):
#     """Uses the web search tool to get up-to-date information."""
#     if state.use_search_engine and state.search_engine_query:
#         search_results = search_tool.run(state.search_engine_query)
#         return {"search_results": search_results}
#     return {"search_results": "Web search was not used or no query provided."}


# # Define tools as functions for LangGraph - Content Generation Tool
# def generate_content_tool_function(state: AgentState):
#     """Generates marketing content based on the current state."""
#     llm = get_llm(os.getenv('OPENAI_API_KEY'), state.model_name_select, state.temperature_slider, state.top_p_slider) # Get LLM from state
#     if not llm:
#         return {"generated_content": "Failed to initialize LLM."}

#     prompt = create_prompt_template(
#         instruction="Generate marketing campaign content",
#         output_format=state.output_format,
#         use_search_engine=state.use_search_engine,
#         search_engine_prompt_template=state.search_engine_query
#     )

#     input_vars = state.dict() # Use the entire state as input variables
#     workflow_result = generate_content_with_retries(llm, prompt, input_vars, state.output_format, state.use_search_engine, state.search_engine_query, state.use_rag, state_to_rag_system(state)) # Pass rag_system

#     if workflow_result:
#         return {"generated_content": workflow_result}
#     else:
#         return {"generated_content": "Content generation failed."}


# # Define tools as functions for LangGraph - Image Generation Tool
# def generate_image_tool_function(state: AgentState):
#     """Generates a product image if requested."""
#     if state.generate_image_checkbox and state.brand and state.image_style:
#         description = ""
#         if isinstance(state.generated_content, MarketingContent):
#             description = f"{state.generated_content.headline}. {state.generated_content.body}"
#         elif isinstance(state.generated_content, str):
#             description = state.generated_content[:500]  # Take first 500 chars if string

#         image_url = generate_product_image(
#             state.brand,
#             description,
#             state.image_style,  # Corrected variable name
#             os.getenv('OPENAI_API_KEY')
#         )
#         if image_url:
#             return {"image_url": image_url}
#         else:
#             return {"image_url": "Image generation failed."}
#     return {"image_url": None}  # No image generated if not requested


# # Define the LLM-based agent decision node
# def llm_tool_choice_function(state: AgentState):
#     """Decides which tool to use next using an LLM."""
#     available_tools = {
#         "RAG Tool": "rag_tool",
#         "Web Search Tool": "web_search_tool",
#         "Generate Content Tool": "generate_content_tool",
#         "Generate Image Tool": "generate_image_tool",
#         "Final Output": "content_output"
#     }

#     tool_descriptions_str = "\n".join([f"{name}: {TOOL_DESCRIPTIONS[tool_id]}" for name, tool_id in available_tools.items() if tool_id != 'content_output']) # Exclude final output from tool list

#     prompt_text = f"""
#     You are an AI agent designed to generate marketing content. You have access to several tools:
#     {tool_descriptions_str}

#     Your goal is to generate high-quality marketing content based on the user's request and the current campaign details.
#     You need to decide which tool to use next to best achieve this goal.

#     Current Campaign Details:
#     Campaign Name: {state.campaign_name}
#     Brand: {state.brand}
#     Product Category: {state.product_category}
#     Output Format: {state.output_format}
#     Specific Instructions: {state.specific_instructions}
#     Use RAG: {'Yes' if state.use_rag else 'No'}
#     Use Web Search: {'Yes' if state.use_search_engine else 'No'}
#     Generate Image: {'Yes' if state.generate_image_checkbox else 'No'}

#     Current State and Results:
#     RAG Context: {state.rag_context if state.rag_context else 'Not yet retrieved'}
#     Web Search Results: {state.search_results if state.search_results else 'Not yet performed'}
#     Generated Content: { 'Yes' if state.generated_content else 'No'}
#     Generated Image: {'Yes' if state.image_url else 'No'}

#     Consider the user's request, the available tools, and the current state of the content generation process.
#     Which tool should you use next?  If you have all the necessary information and have generated the content and image (if requested), choose "Final Output".

#     Choose from: [{', '.join(available_tools.keys())}]
#     Respond with just the name of the tool to use next, or "Final Output" if you are ready to output the content.
#     """

#     llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0) # Use GPT-4 for decision-making
#     if not llm:
#         return "content_output" # Default to output if LLM fails for decision

#     response = llm.invoke(prompt_text)
#     tool_choice = response.content.strip()

#     # Map tool name to node key
#     tool_node_map = {
#         "RAG Tool": "rag_tool",
#         "Web Search Tool": "web_search_tool",
#         "Generate Content Tool": "generate_content_tool",
#         "Generate Image Tool": "generate_image_tool",
#         "Final Output": "content_output"
#     }

#     # Validate and map the tool choice to the graph node
#     chosen_node = tool_node_map.get(tool_choice, "content_output") # Default to content_output if invalid choice

#     logging.info(f"LLM Tool Choice: {tool_choice} -> Node: {chosen_node}")
#     return chosen_node


# # Output Node - Just passes state through for final output
# def content_output(state):
#     """Output node to finalize content generation."""
#     return {"final_state": state} # Just return the state


# # Function to initialize RAG system from state (important for LangGraph context)
# def state_to_rag_system(state: AgentState) -> Optional[RAGSystem]:
#     """Initializes RAG system using LLM from the agent state."""
#     llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0) # Or use model from state if needed
#     if llm:
#         return RAGSystem(llm)
#     return None


# # LangGraph Workflow Definition
# def create_agent_workflow():
#     """Creates the LangGraph agent workflow."""

#     builder = StateGraph(AgentState)

#     # Define nodes as FunctionNodes or ToolNodes
#     builder.add_node("rag_tool", FunctionNode(rag_tool_function))
#     builder.add_node("web_search_tool", FunctionNode(web_search_tool_function))
#     builder.add_node("generate_content_tool", FunctionNode(generate_content_tool_function))
#     builder.add_node("generate_image_tool", FunctionNode(generate_image_tool_function))
#     builder.add_node("content_output", FunctionNode(content_output)) # Final output node
#     builder.add_node("tool_choice", FunctionNode(llm_tool_choice_function)) # LLM-based decision node


#     # Define edges - state transitions based on node outputs
#     builder.add_edge("start", "tool_choice") # Start node to decision node
#     builder.add_conditional_edges(
#         "tool_choice",
#         {
#             "rag_tool": "rag_tool",
#             "web_search_tool": "web_search_tool",
#             "generate_content_tool": "generate_content_tool",
#             "generate_image_tool": "generate_image_tool",
#             "content_output": "content_output" # If no tool needed, go to output
#         },
#         # You can add a fallback edge here if needed, e.g., to a "handle_error" node
#     )
#     builder.add_edge("rag_tool", "tool_choice") # After RAG, decide next step
#     builder.add_edge("web_search_tool", "tool_choice") # After web search, decide next step
#     builder.add_edge("generate_content_tool", "tool_choice") # After content generation, decide next step
#     builder.add_edge("generate_image_tool", "tool_choice") # After image generation, decide next step - back to decision for potential further actions
#     builder.set_entry_point("start") # Entry point of the graph
#     builder.add_edge("content_output", END) # End node


#     graph = builder.compile()
#     return graph


# # --- Streamlit UI Functions ---
# # In the initialize_rag_system function
# def initialize_rag_system_streamlit(openai_api_key):
#     """Initialize RAG system in Streamlit session state, loading documents only once."""
#     if 'rag_system_initialized' not in st.session_state:  # Flag to track initialization
#         try:
#             llm = get_llm(openai_api_key, "gpt-4", temperature=0)
#             rag_system = RAGSystem(llm)

#             if not rag_system.vector_store:  # Check vector_store directly
#                 st.info("🔄 Loading knowledge base...")
#                 loader = TextLoader("/Users/vishalroy/Downloads/ContentGenApp/cleaned_output_simple.txt")
#                 documents = loader.load()
#                 if rag_system.ingest_documents(documents):
#                     st.success("✨ RAG system initialized successfully")
#                 else:
#                     st.warning("RAG system initialization failed.")
#                     return None  # Indicate failure
#             else:
#                 st.info("✨ Using existing RAG knowledge base")

#             st.session_state.rag_system_instance = rag_system  # Store RAG instance
#             st.session_state.rag_system_initialized = True  # Set initialization flag
#             return rag_system  # Return the initialized instance

#         except Exception as e:
#             st.warning(f"RAG system initialization skipped: {str(e)} - will proceed without context")
#             st.session_state.rag_system_initialized = True  # Set flag to avoid retrying
#             return None  # Indicate failure


# def apply_template_defaults(template_type):
#     """Apply default values from a template if selected."""
#     if template_type != "Custom Campaign":
#         template_data = load_campaign_template(template_type)
#         for key, value in template_data.items():
#             if key in st.session_state:
#                 st.session_state[key] = value


# def generate_content_with_retries(llm, prompt, input_vars, output_format, use_search_engine=False, search_engine_query=None, use_rag=False, rag_system: Optional[RAGSystem] = None): # rag_system is now optional and can be passed in
#     max_retries = 3
#     retry_count = 0
#     parser = None

#     if output_format in ["Social Media", "Email", "Marketing"]:
#         parser_map = {
#             "Social Media": PydanticOutputParser(pydantic_object=SocialMediaContent),
#             "Email": PydanticOutputParser(pydantic_object=EmailContent),
#             "Marketing": PydanticOutputParser(pydantic_object=MarketingContent),
#         }
#         parser = parser_map[output_format]

#     while retry_count < max_retries:
#         try:
#             # Simplified RAG handling - Use the passed-in rag_system if available
#             if use_rag and rag_system:
#                 rag_query = input_vars.get("rag_query", input_vars.get("query", input_vars.get("topic", ""))) # Correctly get rag_query from input_vars
#                 if rag_query:
#                     input_vars["rag_context"] = rag_system.query_knowledge_base(rag_query) or "No relevant context found" # Use query_knowledge_base
#                 else:
#                     input_vars["rag_context"] = ""
#             else:
#                 input_vars["rag_context"] = ""

#             # Existing search engine logic
#             if use_search_engine and search_engine_query:
#                 logging.info(f"Performing web search with query: {search_engine_query}")
#                 search_results = search_tool.run(search_engine_query)
#                 logging.info("Search Results:")
#                 logging.info("-" * 50)
#                 logging.info(search_results)
#                 logging.info("-" * 50)
#                 input_vars["search_results"] = search_results
#             else:
#                 logging.info("No web search performed")
#                 input_vars["search_results"] = "No search terms were provided"

#             formatted_prompt = prompt.format(**input_vars)
#             formatted_prompt += "\nIMPORTANT: Return ONLY a valid JSON object with no additional text or formatting."

#             response = llm.invoke(formatted_prompt)
#             response_text = response.content

#             if parser:
#                 try:
#                     # Improved JSON cleaning
#                     response_text = re.sub(r'[\n\r\t]', ' ', response_text)
#                     response_text = re.sub(r'\s+', ' ', response_text)
#                     json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

#                     if not json_match:
#                         raise ValueError("No valid JSON object found in response")

#                     json_str = json_match.group()

#                     # Handle escaped characters before parsing
#                     json_str = json_str.replace('\\"', '"').replace("\\'", "'")

#                     # Handle apostrophes before JSON parsing
#                     def escape_apostrophes(match):
#                         text = match.group(1)
#                         # Escape any apostrophes within the quoted text
#                         text = text.replace("'", "\\'")
#                         return f'"{text}"'

#                     # Replace content within double quotes, handling apostrophes
#                     json_str = re.sub(r'"([^"]*)"', escape_apostrophes, json_str)

#                     # Normalize property names - Fix the regex pattern
#                     json_str = re.sub(
#                         r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str
#                     )

#                     # Remove any remaining unescaped apostrophes
#                     json_str = json_str.replace("'", "\\'")

#                     # Clean up any double-escaped quotes
#                     json_str = json_str.replace('\\"', '"')

#                     # Ensure proper spacing
#                     json_str = re.sub(r",\s*([^\s])", r", \1", json_str)
#                     json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

#                     try:
#                         parsed_json = json.loads(json_str)
#                     except json.JSONDecodeError as je:
#                         # Add detailed logging for debugging
#                         print(f"JSON decode error position {je.pos}: {je.msg}")
#                         print(
#                             f"Character at position: {json_str[je.pos-5:je.pos+5]}"
#                         )
#                         print(f"Full JSON string: {json_str}")
#                         raise

#                     return parser.parse(json.dumps(parsed_json))

#                 except (json.JSONDecodeError, ValueError) as e:
#                     logging.error(f"JSON parsing error: {str(e)}")
#                     logging.error(f"Raw response: {response_text}")
#                     if retry_count < max_retries - 1:
#                         retry_count += 1
#                         time.sleep(1)
#                         continue
#                     raise

#             return response_text

#         except Exception as e:
#             logging.error(f"Generation error: {str(e)}")
#             if retry_count < max_retries - 1:
#                 retry_count += 1
#                 time.sleep(1)
#                 continue
#             raise

#     return None


# def main():
#     configure_streamlit_page()
#     load_css()

#     # Load API Keys once
#     google_api_key, openai_api_key = load_api_keys()

#     # Initialize RAG system in Streamlit session state
#     rag_system_instance = initialize_rag_system_streamlit(openai_api_key) # Get RAG instance

#     # Initialize LangGraph workflow
#     agent_workflow = create_agent_workflow()

#     # Sidebar setup
#     with st.sidebar:
#         st.header("📊 Campaign Tools")
#         template_type = st.selectbox(
#             "Select Campaign Type",
#             ["Custom Campaign", "Product Launch", "Seasonal Sale", "Brand Awareness"],
#             on_change=apply_template_defaults,  # Apply template on change
#             args=[st.session_state.get("template_type", "Custom Campaign")] # Pass current value to set default in function
#         )
#         st.session_state["template_type"] = template_type

#         # Campaign History
#         st.subheader("Recent Campaigns")
#         if "campaign_history" not in st.session_state:
#             st.session_state.campaign_history = []
#         for campaign in st.session_state.campaign_history[-5:]:
#             st.text(f"📄 {campaign}")

#     # Main content
#     st.title("🌟 Pwani Oil Marketing Content Generator (Agentic)") # Updated title
#     st.caption("Generate professional marketing content powered by AI")

#     # Tabs for different content types
#     tab1, tab2, tab3 = st.tabs(
#         ["Campaign Details", "Target Market", "Advanced Settings"]
#     )

#     with tab1:
#         col1, col2 = st.columns(2)
#         with col1:
#             campaign_name = st.text_input(
#                 "Campaign Name",
#                 key="campaign_name",
#                 help="Enter a unique name for your campaign",
#             )

#             # Replace the existing brand selection with new dropdown and description
#             selected_brand = st.selectbox(
#                 "Brand",
#                 options=list(BRAND_OPTIONS.keys()),
#                 help="Select the brand for the campaign"
#             )

#             if selected_brand:
#                 st.info(f"📝 **Brand Description:** {BRAND_OPTIONS[selected_brand]}")

#             promotion_link = st.text_input(
#                 "Promotion Link",
#                 key="promotion_link",
#                 help="Enter the landing page URL",
#             )
#             previous_campaign_reference = st.text_input(
#                 "Previous Campaign Reference", key="previous_campaign_reference"
#             )
#         # In tab1, under col2
#             with col2:
#                 sku = st.text_input("SKU", key="sku", help="Product SKU number")
#                 product_category = st.selectbox(
#                     "Product Category",
#                     ["Cooking Oil", "Personal Care", "Home Care"],
#                     key="product_category"
#                 )
#                 campaign_date_range = st.text_input(
#                     "Campaign Date Range (YYYY-MM-DD to YYYY-MM-DD)",
#                     key="campaign_date_range",
#                 )
#                 tone_style = st.selectbox(
#                     "Tone & Style",
#                     [
#                         "Professional",
#                         "Casual",
#                         "Friendly",
#                         "Humorous",
#                         "Formal",
#                         "Inspirational",
#                         "Educational",
#                         "Persuasive",
#                         "Emotional"
#                     ],
#                     key="tone_style_tab1",
#                     help="Select the tone and style for your content"
#                 )

#     with tab2:
#         col1, col2 = st.columns(2)
#         with col1:
#             age_range = (
#                 st.select_slider(
#                     "Age Range", options=list(range(18, 76, 1)), value=(25, 45),
#                      key="age_range_slider" #Unique Key
#                 )
#                 if st.checkbox("Add Age Range", key="use_age_range_tab2")
#                 else None
#             )
#             gender = (
#                 st.multiselect(
#                     "Gender", ["Male", "Female", "Other"], default=["Female"],
#                     key="gender_multiselect" #Unique Key
#                 )
#                 if st.checkbox("Add Gender", key="use_gender_tab2")
#                 else None
#             )
#         with col2:
#             income_level = (
#                 st.select_slider(
#                     "Income Level",
#                     options=["Low", "Middle Low", "Middle", "Middle High", "High"],
#                     value="Middle",
#                     key="income_level_slider" #Unique Key
#                 )
#                 if st.checkbox("Add Income Level", key="use_income_level_tab2")
#                 else None
#             )
#             region = (
#                 st.multiselect(
#                     "Region",
#                     ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"],
#                     default=["Nairobi", "Mombasa"],
#                      key="region_multiselect" #Unique Key
#                 )
#                 if st.checkbox("Add Region", key="use_region_multiselect")
#                 else None
#             )
#             urban_rural = (
#                 st.multiselect(
#                     "Area Type", ["Urban", "Suburban", "Rural"], default=["Urban"],
#                     key="urban_rural_multiselect" #Unique Key
#                 )
#                 if st.checkbox("Add Area Type", key="use_urban_rural_tab2")
#                 else None
#             )

#     with tab3:
#         col1, col2 = st.columns(2)
#         with col1:
#             model_name = st.selectbox(
#                 "Model",
#                 ["gpt-4", "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
#                 help="Select the AI model to use",
#                 key="model_name_select"
#             )
#             output_format = st.selectbox(
#                 "Output Format",
#                 ["Social Media", "Email", "Marketing", "Text"],
#                 help="Choose the type of content to generate",
#                 key="output_format_select"
#             )

#             # Add RAG option
#             use_rag = st.checkbox("Use RAG System", value=True, help="Use Retrieval Augmented Generation for better context", key="use_rag_checkbox")
#             if use_rag:
#                 rag_query = st.text_input("RAG Query", help="Enter specific query for knowledge retrieval", key="rag_query_input")

#             # Add image generation options
#             generate_image = st.checkbox("Generate Product Image", value=False, key="generate_image_checkbox")
#             if generate_image:
#                 image_style = st.selectbox(
#                     "Image Style",
#                     ["Realistic", "Artistic", "Modern", "Classic"],
#                     help="Select the style for the generated image",
#                     key="image_style_select"
#                 )
#             # Add search engine option
#             use_search_engine = st.checkbox("Use Web Search", value=False, help="Incorporate live web search results into the content", key="use_search_engine_checkbox")
#             search_engine_query = None  # Initialize with None
#             if use_search_engine:
#                 search_engine_query = st.text_input("Search Query", help="Enter the search query for the web search engine", key="search_query_input")

#         with col2:
#             temperature = st.slider(
#                 "Creativity Level",
#                 0.0,
#                 1.0,
#                 0.7,
#                 help="Higher values = more creative output",
#                 key="temperature_slider"
#             )
#             top_p = st.slider(
#                 "Diversity Level",
#                 0.0,
#                 1.0,
#                 0.9,
#                 help="Higher values = more diverse output",
#                 key="top_p_slider"
#             )


#     # Content requirements
#     st.subheader("Content Requirements")
#     specific_instructions = st.text_area(
#         "Specific Instructions",
#         help="Enter any specific requirements or guidelines for the content",
#         key="specific_instructions_input"
#     )

#     # Generate button with loading state
#     if st.button("🚀 Generate Content", type="primary"):
#         agent_state_input = AgentState( # Initialize AgentState object
#             campaign_name=campaign_name,
#             promotion_link=promotion_link,
#             previous_campaign_reference=previous_campaign_reference,
#             sku=sku,
#             product_category=product_category,
#             campaign_date_range=campaign_date_range,
#             age_range=f"{age_range[0]}-{age_range[1]}" if age_range else None,
#             gender=", ".join(gender) if gender else None,
#             income_level=income_level if income_level else None,
#             region=", ".join(region) if region else None,
#             urban_rural=", ".join(urban_rural) if urban_rural else None,
#             specific_instructions=specific_instructions,
#             brand=selected_brand,
#             tone_style=tone_style,
#             output_format=output_format,
#             rag_query=rag_query,
#             use_rag=use_rag,
#             use_search_engine=use_search_engine,
#             search_engine_query=search_engine_query,
#             generate_image_checkbox=generate_image, # Pass image generation flag
#             image_style_select=image_style, # Pass image style
#             model_name_select=model_name, # Pass model name
#             temperature_slider=temperature, # Pass temperature
#             top_p_slider=top_p # Pass top_p
#         )

#         # Validate inputs
#         is_valid, error_message = validate_inputs(agent_state_input.dict()) # Validate state as dict
#         if not is_valid:
#             st.error(error_message)
#             st.stop()

#         # Validate date range
#         if not validate_date_range(campaign_date_range):
#             st.error("Invalid date range. End date must be after start date.")
#             st.stop()

#         # Run LangGraph agent workflow
#         with st.spinner("🎨 Generating your marketing content..."):
#             progress_bar = st.progress(0)
#             for i in range(100):
#                 time.sleep(0.01)
#                 progress_bar.progress(i + 1)

#             try:
#                 # Invoke LangGraph workflow - Pass initial state
#                 agent_result = agent_workflow.invoke(agent_state_input) # Pass AgentState object

#                 final_state = agent_result['final_state'] # Access final state from output
#                 generated_content = final_state.generated_content
#                 image_url = final_state.image_url

#                 if not generated_content:
#                     st.error("No content was generated. Please try again.")
#                     st.stop()

#                 # Display generated content
#                 st.success("✨ Content generated successfully!")
#                 st.subheader("Generated Content")
#                 st.markdown("---")

#                 # Display content based on type - Use final_state.generated_content
#                 if isinstance(generated_content, str):
#                     st.markdown(generated_content)
#                 elif isinstance(generated_content, MarketingContent):
#                     st.subheader(generated_content.headline)
#                     st.write(generated_content.body)
#                     st.markdown(f"**Call to Action:** {generated_content.call_to_action}")
#                     st.markdown("**Key Benefits:**")
#                     for benefit in generated_content.key_benefits:
#                         st.markdown(f"- {benefit}")
#                 elif isinstance(generated_content, SocialMediaContent):
#                     st.markdown(f"**Platform:** {generated_content.platform}")
#                     st.markdown(f"**Post Text:** {generated_content.post_text}")
#                     st.markdown(f"**Hashtags:** {', '.join(generated_content.hashtags)}")
#                     st.markdown(f"**Call to Action:** {generated_content.call_to_action}")

#                 elif isinstance(generated_content, EmailContent):
#                     st.markdown(f"**Subject Line:** {generated_content.subject_line}")
#                     st.markdown(f"**Preview Text:** {generated_content.preview_text}")
#                     st.markdown(f"**Body:** {generated_content.body}")
#                     st.markdown(f"**Call to Action:** {generated_content.call_to_action}")
#                 elif isinstance(generated_content, dict):
#                     st.markdown(json.dumps(generated_content, indent=2))
#                 else:
#                     st.markdown(generated_content)

#                 # Generate image if requested and display - Use final_state.image_url
#                 if generate_image and image_url:
#                     st.subheader("Generated Image")
#                     st.image(image_url, caption=f"{selected_brand} Product Image")
#                     if st.button("💾 Save Image"):
#                         saved_image_path = save_generated_image(image_url, selected_brand)
#                         if saved_image_path:
#                             st.success(f"Image saved to: {saved_image_path}")
#                 elif generate_image and not image_url:
#                     st.error("Failed to generate image. Please try again.")


#                 # Save content options
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     save_format = st.selectbox("Save Format", ["txt", "json"])
#                 with col2:
#                     if st.button("💾 Save Content"):
#                         saved_file = save_content_to_file(
#                             generated_content, campaign_name, save_format
#                         )
#                         if saved_file:
#                             st.success(f"Content saved to: {saved_file}")
#                             st.session_state.campaign_history.append(
#                                 f"{campaign_name} ({datetime.now().strftime('%Y-%m-%d')})"
#                             )

#             except Exception as e:
#                 st.error(f"Error generating content: {str(e)}")
#                 st.error(e)


# if __name__ == "__main__":
#     main()

from streamlit_ui import main

if __name__ == "__main__":
    main()