import streamlit as st
import time
from config import configure_streamlit_page, load_api_keys, load_css
from data import BRAND_OPTIONS
from prompt import create_prompt_template
from llm import get_llm
from workflow import create_langraph_workflow
from utils import generate_product_image, save_generated_image ,validate_inputs, save_content_to_file, load_campaign_template, validate_date_range
from image import generate_product_image, save_generated_image
from data import SocialMediaContent, EmailContent, MarketingContent
import json
from datetime import datetime
from rag import RAGSystem
from langchain_community.document_loaders import TextLoader
from typing import List, Dict, Optional
import logging
from pathlib import Path
import os
import random

# --- UI Enhancements and Helper Functions ---

def display_loading_message():
    """Displays a random loading message."""
    messages = [
        "✨ Brewing your perfect marketing potion...",
        "🧠 Engaging the AI marketing guru...",
        "🚀 Launching the content rocket...",
        "💡 Gathering creative sparks...",
        "📝 Crafting your masterpiece...",
    ]
    return random.choice(messages)

def display_help_message(section: str):
    """Displays help messages for different sections, more structured."""
    messages = {
        "campaign_overview": """
            **Campaign Overview:**  This section defines the core elements of your marketing campaign. 
            - **Campaign Name:** Give your campaign a memorable and descriptive name.
            - **Brand, Product Category, SKU:**  Specify which product this campaign is for.
            - **Campaign Date Range:** When will the campaign run?  (Format: YYYY-MM-DD to YYYY-MM-DD).
            - **Tone & Style:**  Choose the overall feel of your campaign (e.g., Professional, Friendly).
        """,
        "target_market": """
            **Target Market:**  Define your ideal customer for this campaign.  Be as specific as possible.
            - Use the checkboxes to enable/disable specific demographic filters.
            - **Age Range:** Use the slider to select the age bracket.
            - **Gender:** Choose one or more genders.
            - **Income Level:** Select the income bracket.
            - **Region:** Select the geographical areas to target.
            - **Area Type:**  Specify if your target audience is primarily in urban, suburban, or rural areas.
        """,
        "advanced_settings": """
            **Advanced Settings:** Customize the AI model and content generation parameters.
            - **Model:** Select the AI model (GPT-4 or Gemini Pro) for generating content.  GPT-4 is generally more powerful.
            - **Output Format:** Choose the type of content you want to generate (e.g., Social media post, email).
            - **Use RAG System:**  Enable Retrieval-Augmented Generation (RAG) to incorporate information from a knowledge base (highly recommended for accuracy).
            - **Generate Product Image:** Check this box to generate an image related to your campaign.
            - **Image Style:** Choose a style for the generated image.
            - **Use Web Search:**  Enable web search to automatically gather the latest market data based on your brand and product.
            - **Search Query:** Provide keywords related to your campaign for web search.
            - **Creativity/Diversity:** Adjust these sliders to control the AI's creativity and the variety of the output.
        """,
        "content_requirements": """
            **Content Requirements:** Provide any specific instructions or details that the AI should consider.
            - Be as clear and detailed as possible to guide the AI.  
            - Example: "Focus on the health benefits of the product," or "Mention the ongoing 20% discount offer."
        """,
        "campaign_details": """
            **Campaign Details:** Provide additional information for your campaign.
            - **Promotion Link:** Include the URL of any promotional offer or landing page.
            - **Previous Campaign Reference:**  If this campaign builds upon a previous one, provide a reference (name or ID).
            - **Success Metrics:**  Select the key performance indicators (KPIs) you'll use to measure success.
            - **Marketing Channels:**  Choose the channels you'll use to distribute your campaign (e.g., Social Media, Email, TV).
        """,
        "chat": """
            **Chat with Marketing Assistant:**  Interact with a specialized AI assistant to get quick answers and refine your campaign strategy.
            - Ask questions about Pwani Oil products, marketing strategies, and best practices.
            - The chat assistant uses a knowledge base and can access real-time information.
            - Use the "Clear Chat" button to start a new conversation.
        """,
    }
    if st.button("ℹ️ Help", key=f"help_button_{section}"):
        st.info(messages.get(section, "Help information for this section is not available."))

def enhance_input_field(input_type, label, key, **kwargs):
    """Creates a styled input field with a modern look."""
    with st.container():
        # Use a slightly lighter background and a subtle border
        st.markdown(f"""
            <style>
            div[data-baseweb="{input_type}"][key="{key}"] {{
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 5px;
            }}
            </style>
        """, unsafe_allow_html=True)
        return input_type(label, key=key, **kwargs)


class ChatBot:
    """
    ChatBot class for handling conversational interactions, including RAG integration.
    """
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


def create_chat_interface(llm, rag_system):
    """Create and render the chat interface."""
    st.subheader("💬 Chat with Marketing Assistant")
    display_help_message("chat")

    # Initialize chatbot if not already done
    if 'chatbot' not in st.session_state:
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


def initialize_rag_system(openai_api_key):
    """Initialize RAG system, loading documents only once."""

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if 'rag_system' not in st.session_state:
        try:
            # Validate API key
            if not openai_api_key:
                raise ValueError("OpenAI API key is not provided")

            # Initialize LLM with retry mechanism
            max_retries = 3
            retry_count = 0
            llm = None

            while retry_count < max_retries:
                try:
                    llm = get_llm(openai_api_key, "gpt-4", temperature=0)
                    break
                except Exception as llm_error:
                    retry_count += 1
                    logger.warning(f"LLM initialization attempt {retry_count} failed: {str(llm_error)}")
                    if retry_count == max_retries:
                        raise

            if not llm:
                raise RuntimeError("Failed to initialize LLM after multiple attempts")

            st.session_state.rag_system = RAGSystem(llm)
            st.session_state.llm = llm # Store LLM for chatbot

            # Check and initialize vector store with proper cleanup
            try:
                if not hasattr(st.session_state.rag_system, 'vector_store') or st.session_state.rag_system.vector_store is None:
                    st.info("🔄 Loading knowledge base...")

                    # Validate knowledge base file
                    knowledge_base_path = Path("/Users/vishalroy/Downloads/ContentGenApp/cleaned_cleaned_output.txt")
                    if not knowledge_base_path.exists():
                        raise FileNotFoundError(f"Knowledge base file not found at {knowledge_base_path}")

                    try:
                        loader = TextLoader(str(knowledge_base_path))
                        documents = loader.load()

                        if not documents:
                            raise ValueError("No documents loaded from knowledge base")

                        # Use proper ingest method and handle cleanup
                        if st.session_state.rag_system.ingest(documents):
                            logger.info("RAG system initialized successfully")
                            st.success("✨ RAG system initialized successfully")
                        else:
                            logger.warning("Document ingestion failed")
                            st.warning("RAG system initialization skipped - will proceed without context")
                            # Cleanup on failure
                            if hasattr(st.session_state.rag_system, 'vector_store'):
                                st.session_state.rag_system.vector_store = None
                    except Exception as doc_error:
                        logger.error(f"Error loading documents: {str(doc_error)}")
                        # Ensure cleanup on error
                        if hasattr(st.session_state.rag_system, 'vector_store'):
                            st.session_state.rag_system.vector_store = None
                        raise RuntimeError(f"Failed to load knowledge base: {str(doc_error)}")
                else:
                    logger.info("Using existing RAG knowledge base")
                    st.info("✨ Using existing RAG knowledge base")
            except Exception as vs_error:
                logger.error(f"Vector store initialization failed: {str(vs_error)}")
                if hasattr(st.session_state.rag_system, 'vector_store'):
                    st.session_state.rag_system.vector_store = None
                raise

        except Exception as e:
            error_msg = f"RAG system initialization failed: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

            # Attempt to recover or provide fallback functionality
            if 'rag_system' in st.session_state:
                del st.session_state.rag_system

            # Notify user and provide guidance
            st.warning(
                "The system will proceed without RAG capabilities. "
                "This may affect content generation quality. "
                "Please try refreshing the page or contact support if the issue persists."
            )


def apply_template_defaults(template_type):
    """Apply default values from a template if selected."""
    if template_type != "Custom Campaign":
        template_data = load_campaign_template(template_type)
        for key, value in template_data.items():
            if key in st.session_state:
                st.session_state[key] = value

def generate_content_workflow(input_vars, model_name, temperature, top_p, output_format, use_rag, rag_query, use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key):
    """Handles the content generation workflow with enhanced error handling and caching."""
    logger = logging.getLogger(__name__)

    # Create cache key based on input parameters
    cache_key = f"{model_name}_{temperature}_{top_p}_{output_format}_{hash(frozenset(input_vars.items()))}"

    # Check if result is in cache
    if hasattr(st.session_state, 'content_cache') and cache_key in st.session_state.content_cache:
        logger.info("Using cached content result")
        return st.session_state.content_cache[cache_key]

    try:
        # Validate input parameters
        if not model_name or not output_format:
            raise ValueError("Model name and output format are required")

        # Select appropriate API key and initialize LLM with retries
        api_key = google_api_key if not model_name.startswith("gpt") else openai_api_key
        if not api_key:
            raise ValueError(f"API key not found for model {model_name}")

        max_retries = 3
        retry_count = 0
        llm = None

        while retry_count < max_retries:
            try:
                llm = get_llm(api_key, model_name, temperature, top_p)
                if llm:
                    break
                retry_count += 1
                logger.warning(f"LLM initialization attempt {retry_count} failed")
            except Exception as llm_error:
                retry_count += 1
                logger.error(f"LLM initialization error: {str(llm_error)}")
                if retry_count == max_retries:
                    raise RuntimeError(f"Failed to initialize LLM after {max_retries} attempts")

        # Create prompt template with validation
        try:
            prompt = create_prompt_template(
                instruction="Generate marketing campaign content",
                output_format=output_format,
                use_search_engine=use_search_engine,
                search_engine_prompt_template=search_engine_query
            )
        except Exception as prompt_error:
            logger.error(f"Failed to create prompt template: {str(prompt_error)}")
            raise

        # Handle RAG context if enabled
        if use_rag and rag_query:
            try:
                if not hasattr(st.session_state, 'rag_system'):
                    logger.warning("RAG system not initialized, skipping context retrieval")
                else:
                    context_query = f"""
                    Brand: {selected_brand}
                    Product: {input_vars.get('sku', 'N/A')}
                    Category: {input_vars.get('product_category', 'N/A')}
                    Query: {rag_query}
                    """
                    rag_context = st.session_state.rag_system.query(context_query)
                    if rag_context:
                        input_vars["rag_context"] = rag_context
                        logger.info("Successfully retrieved RAG context")
                    else:
                        logger.warning("No RAG context retrieved")
            except Exception as rag_error:
                logger.error(f"RAG context retrieval failed: {str(rag_error)}")
                input_vars["rag_context"] = ""

        # Create and invoke workflow with optimized error handling
        try:
            workflow = create_langraph_workflow(
                llm,
                prompt,
                input_vars,
                output_format,
                use_search_engine,
                search_engine_query if search_engine_query else None
            )

            result = workflow.invoke(input_vars)

            if "error" in result:
                logger.error(f"Workflow error: {result['error']}")
                return {"error": result["error"]}

            generated_content = result["output"]
            logger.info("Content generated successfully")

            # Cache the successful result
            if not hasattr(st.session_state, 'content_cache'):
                st.session_state.content_cache = {}
            st.session_state.content_cache[cache_key] = {"content": generated_content}

            return {"content": generated_content}

        except Exception as workflow_error:
            logger.error(f"Workflow execution failed: {str(workflow_error)}")
            raise

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Content generation failed: {error_msg}")

        # Provide user-friendly error messages
        if "rate limit" in error_msg.lower():
            return {"error": "Service is currently busy. Please try again in a few moments."}
        elif "timeout" in error_msg.lower():
            return {"error": "Request took too long to process. Please try again."}
        elif "api key" in error_msg.lower():
            return {"error": "Authentication error. Please check your API configuration."}
        else:
            return {"error": f"An unexpected error occurred: {error_msg}"}


def display_generated_content(generated_content, selected_brand, generate_image, image_style, openai_api_key, campaign_name):
    """Displays the generated content and handles image generation/saving."""
    st.success("✨ Content generated successfully!")
    st.subheader("Generated Content")
    st.markdown("---")

    if isinstance(generated_content, str):
        st.markdown(generated_content)
    elif isinstance(generated_content, MarketingContent):
        st.subheader(generated_content.headline)
        st.write(generated_content.body)
        st.markdown(f"**Call to Action:** {generated_content.call_to_action}")
        st.markdown("**Key Benefits:**")
        for benefit in generated_content.key_benefits:
            st.markdown(f"- {benefit}")
    elif isinstance(generated_content, SocialMediaContent):
        st.markdown(f"**Platform:** {generated_content.platform}")
        st.markdown(f"**Post Text:** {generated_content.post_text}")
        st.markdown(f"**Hashtags:** {', '.join(generated_content.hashtags)}")
        st.markdown(f"**Call to Action:** {generated_content.call_to_action}")
    elif isinstance(generated_content, EmailContent):
        st.markdown(f"**Subject Line:** {generated_content.subject_line}")
        st.markdown(f"**Preview Text:** {generated_content.preview_text}")
        st.markdown(f"**Body:** {generated_content.body}")
        st.markdown(f"**Call to Action:** {generated_content.call_to_action}")
    elif isinstance(generated_content, dict):
        st.markdown(json.dumps(generated_content, indent=2))
    else:
        st.markdown(generated_content)

    # Generate image if requested
    if generate_image:
        st.subheader("Generated Image")
        with st.spinner("🎨 Generating product image..."):
            description = ""
            if isinstance(generated_content, MarketingContent):
                description = f"{generated_content.headline}. {generated_content.body}"
            elif isinstance(generated_content, str):
                description = generated_content[:500]

            image_url = generate_product_image(
                selected_brand,
                description,
                image_style,
                openai_api_key
            )

            if image_url:
                st.image(image_url, caption=f"{selected_brand} Product Image")
                if st.button("💾 Save Image"):
                    saved_image_path = save_generated_image(image_url, selected_brand)
                    if saved_image_path:
                        st.success(f"Image saved to: {saved_image_path}")
            else:
                st.error("Failed to generate image. Please try again.")

    return True  # Indicate content display success


def create_content_generation_page():
    st.title("🌟 Pwani Oil Marketing Content Generator")
    st.markdown("#### *Generate professional marketing content powered by AI*")

    # Essential campaign details in a clean layout
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            campaign_name = st.text_input(
                "Campaign Name*",
                key="campaign_name",
                help="Enter a unique name for your campaign"
            )

            selected_brand = st.selectbox(
                "Brand*",
                key="selected_brand",
                options=list(BRAND_OPTIONS.keys()),
                help="Select the brand for this campaign"
            )

            product_category = st.selectbox(
                "Product Category*",
                key="product_category",
                options=["Cooking Oil", "Cooking Fat", "Bathing Soap", "Home Care", "Lotion", "Margarine", "Medicine Soap"],
                help="Select the product category"
            )

        with col2:
            sku = st.selectbox(
                "SKU*",
                key="sku",
                options=BRAND_OPTIONS.get(selected_brand, ["No SKUs available"]),
                help="Select the specific product SKU"
            )

            campaign_objective = st.selectbox(
                "Campaign Objective*",
                key="campaign_objective",
                options=["Brand Awareness", "Lead Generation", "Sales Conversion", "Product Launch"],
                help="Select the primary goal of your campaign"
            )

    # Content generation settings
    with st.container():
        st.subheader("Content Settings")
        col3, col4 = st.columns(2)
        with col3:
            output_format = st.selectbox(
                "Content Type*",
                ["Social Media Post", "Email Campaign", "Marketing Copy"],
                help="Choose the type of content you want to generate"
            )

            use_rag = st.checkbox(
                "Use Knowledge Base",
                value=True,
                help="Enable to use our product knowledge base for better content"
            )

        with col4:
            model_name = st.selectbox(
                "AI Model*",
                ["gpt-4", "gemini-pro"],
                help="Select the AI model for content generation"
            )

            generate_image = st.checkbox(
                "Generate Product Image",
                help="Generate an AI image for your campaign"
            )

    # Generate button
    if st.button("✨ Generate Content", type="primary", use_container_width=True):
        if not campaign_name or not selected_brand or not sku:
            st.error("Please fill in all required fields marked with *")
            return

        with st.spinner(display_loading_message()):
            input_vars = {
                "campaign_name": campaign_name,
                "campaign_objective": campaign_objective,
                "selected_brand": selected_brand,
                "product_category": product_category,
                "sku": sku
            }

            result = generate_content_workflow(
                input_vars,
                model_name,
                0.7,  # Default temperature
                0.9,  # Default top_p
                output_format,
                use_rag,
                f"Information about {selected_brand} {sku}",
                False,  # Disabled web search
                "",
                selected_brand,
                st.session_state.get("openai_api_key"),
                st.session_state.get("google_api_key")
            )

            if "error" in result:
                st.error(result["error"])
            else:
                display_generated_content(
                    result["content"],
                    selected_brand,
                    generate_image,
                    "modern",  # Default image style
                    st.session_state.get("openai_api_key"),
                    campaign_name
                )


def main():
    configure_streamlit_page()
    load_css()

    # Load API Keys once
    google_api_key, openai_api_key = load_api_keys()

    # Initialize RAG system and LLM for chatbot *before* creating the UI
    initialize_rag_system(openai_api_key)

    # Sidebar setup -  Enhanced Layout
    with st.sidebar:
        st.header("📊 Campaign Tools")

        # Campaign History - More Compact and Informative
        st.subheader("Recent Campaigns")
        if "campaign_history" not in st.session_state:
            st.session_state.campaign_history = []

        if st.session_state.campaign_history:
            for campaign in st.session_state.campaign_history[-5:]:
                name, date = campaign.split(" (")
                date = date[:-1]  # Remove the closing parenthesis
                st.markdown(f"- **{name}** ({date})") # Bolding the campaign name
        else:
            st.write("No campaigns created yet.")


        # Template Selection - Now in the sidebar, more prominent.
        template_type = st.selectbox(
            "Select Campaign Template",
            ["Custom Campaign", "Product Launch", "Seasonal Sale", "Brand Awareness"],
            on_change=apply_template_defaults,
            args=[st.session_state.get("template_type", "Custom Campaign")]
        )
        st.session_state["template_type"] = template_type


    # Main content
     # Title and Caption - More visually appealing.
    st.title("🌟 Pwani Oil Marketing Content Generator")
    st.markdown("#### *Generate professional marketing content powered by AI*")
    st.caption("Fill in the details, choose your settings, and let the AI do the magic!")


    # Place the chat interface *outside* of the tabs
    create_chat_interface(st.session_state.llm, st.session_state.rag_system)

    # Tabs for different content types -  Using expanders for better organization
    tab1, tab2, tab3 = st.tabs(
        ["Campaign Details", "Target Market", "Advanced Settings"]
    )


    with tab1:
        with st.expander("Campaign Overview", expanded=True):
            display_help_message("campaign_overview")
            col1, col2 = st.columns(2)
            with col1:
                campaign_name = enhance_input_field(
                    st.text_input,
                    "Campaign Name",
                    "campaign_name",
                    help="Enter a unique name for your campaign",
                )

                selected_brand = enhance_input_field(
                    st.selectbox,
                    "Brand",
                    "selected_brand",
                    options=list(BRAND_OPTIONS.keys()),
                    help="Select the brand for this campaign",
                )

                product_category = enhance_input_field(
                    st.selectbox,
                    "Product Category",
                    "product_category",
                    options=["Cooking Oil", "Cooking Fat", "Bathing Soap", "Home Care", "Lotion", "Margarine", "Medicine Soap"],
                    help="Select the product category",
                )

                sku = enhance_input_field(
                    st.selectbox,
                    "SKU",
                    "sku",
                    options=["500L", "250L", "1L", "10L", "20L", "2L", "3L", "5L", "10KG", "500G", "1KG", "2KG", "17KG", "4KG", "100G", "700G", "800G", "600G", "80G", "125G", "175G", "200G", "225G", "20G"],
                    help="Select the product SKU number"
                )

            with col2:
                campaign_date_range = enhance_input_field(
                    st.text_input,
                    "Campaign Date Range (YYYY-MM-DD to YYYY-MM-DD)",
                    "campaign_date_range",
                    help="Enter the campaign date range",
                )



                tone_style = enhance_input_field(
                    st.selectbox,
                    "Tone & Style",
                    "tone_style",
                    options=["Professional", "Casual", "Friendly", "Formal", "Humorous"],
                    help="Select the tone for your campaign",
                )

        with st.expander("Campaign Details", expanded=False):
            display_help_message("campaign_details")
            col3, col4 = st.columns(2)
            with col3:
                promotion_link = enhance_input_field(
                    st.text_input,
                    "Promotion Link",
                    "promotion_link",
                    help="Enter the promotional URL (optional)",
                )

                previous_campaign_reference = enhance_input_field(
                    st.text_input,
                    "Previous Campaign Reference",
                    "previous_campaign_reference",
                    help="Reference to any previous related campaign (optional)",
                )

            with col4:
                success_metrics = enhance_input_field(
                    st.multiselect,
                    "Success Metrics",
                    "success_metrics",
                    options=["Sales Revenue", "Website Traffic", "Social Media Engagement", "Lead Generation", "Brand Mentions", "Customer Feedback"],
                    default=["Sales Revenue", "Social Media Engagement"],
                    help="Select metrics to measure campaign success"
                )

                campaign_channels = enhance_input_field(
                    st.multiselect,
                    "Marketing Channels",
                    "campaign_channels",
                    options=["Social Media", "Email", "SMS", "Radio", "TV", "Print Media", "Outdoor Advertising", "Digital Display"],
                    default=["Social Media", "Email"],
                    help="Select channels for campaign distribution"
                )

    with tab2:
        with st.expander("Target Market", expanded=True):
            display_help_message("target_market")
            col1, col2 = st.columns(2)
            with col1:
                age_range = (
                    enhance_input_field(
                        st.select_slider,
                        "Age Range",
                        "age_range_slider",  # Unique Key
                        options=list(range(18, 76, 1)),
                        value=(25, 45)
                    )
                    if st.checkbox("Add Age Range", key="use_age_range_tab2")
                    else None
                )
                gender = (
                    enhance_input_field(
                        st.multiselect,
                        "Gender",
                        "gender_multiselect",  # Unique Key
                        options=["Male", "Female", "Other"],
                        default=["Female"]
                    )
                    if st.checkbox("Add Gender", key="use_gender_tab2")
                    else None
                )
            with col2:
                income_level = (
                    enhance_input_field(
                        st.select_slider,
                        "Income Level",
                        "income_level_slider",  # Unique Key
                        options=["Low", "Middle Low", "Middle", "Middle High", "High"],
                        value="Middle"
                    )
                    if st.checkbox("Add Income Level", key="use_income_level_tab2")
                    else None
                )
                region = (
                    enhance_input_field(
                        st.multiselect,
                        "Region",
                        "region_multiselect",  # Unique Key
                        options=["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"],
                        default=["Nairobi", "Mombasa"]
                    )
                    if st.checkbox("Add Region", key="use_region_tab2")
                    else None
                )
                urban_rural = (
                    enhance_input_field(
                        st.multiselect,
                        "Area Type",
                        "urban_rural_multiselect",  # Unique Key
                        options=["Urban", "Suburban", "Rural"],
                        default=["Urban"]
                    )
                    if st.checkbox("Add Area Type", key="use_urban_rural_tab2")
                    else None
                )

    with tab3:
        with st.expander("Advanced Settings", expanded=True):
           display_help_message("advanced_settings")
           col1, col2 = st.columns(2)
           with col1:
                model_name = enhance_input_field(
                    st.selectbox,
                    "Model",
                    "model_name_select",
                    options=["gpt-4", "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp","gemini-2.0-flash-thinking-exp-01-21"],
                    help="Select the AI model to use",
                )
                output_format = enhance_input_field(
                    st.selectbox,
                    "Output Format",
                    "output_format_select",
                    options=["Social Media", "Email", "Marketing", "Text", "Tagline"],
                    help="Choose the type of content to generate",
                )

                # Add RAG option
                use_rag = enhance_input_field(
                    st.checkbox,
                    "Use RAG System",
                    "use_rag_checkbox",
                    value=True,
                    help="Use Retrieval Augmented Generation for better context",
                )

                # Add image generation options
                generate_image = enhance_input_field(
                    st.checkbox,
                    "Generate Product Image",
                    "generate_image_checkbox",
                    value=False
                )
                image_style = enhance_input_field(
                    st.selectbox,
                    "Image Style",
                    "image_style_select",
                    options=["Realistic", "Artistic", "Modern", "Classic"],
                    help="Select the style for the generated image",
                ) if generate_image else None  # Conditional selectbox

                # Add search engine option with automatic query generation
                use_search_engine = enhance_input_field(
                    st.checkbox,
                    "Use Web Search",
                    "use_search_engine_checkbox",
                    value=False,
                    help="Automatically fetch real-time market data based on campaign details",
                )
                search_engine_query = None  # Initialize search_engine_query
                if use_search_engine:
                    search_engine_query = enhance_input_field(
                        st.text_input,
                        "Search Query",
                        "search_query_input",
                        help="Enter the search query for the web search engine",
                    )

           with col2:
                temperature = enhance_input_field(
                    st.slider,
                    "Creativity Level",
                    "temperature_slider",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Higher values = more creative output",
                )
                top_p = enhance_input_field(
                    st.slider,
                    "Diversity Level",
                    "top_p_slider",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    help="Higher values = more diverse output",
                )



    # Content requirements
    with st.expander("Content Requirements", expanded=True):
        display_help_message("content_requirements")
        specific_instructions = enhance_input_field(
            st.text_area,
            "Specific Instructions",
            "specific_instructions_input",
            help="Enter any specific requirements or guidelines for the content",
        )

    # State to manage content generation and satisfaction
    if "generated_content_result" not in st.session_state:
        st.session_state.generated_content_result = None
    if "content_satisfied" not in st.session_state:
        st.session_state.content_satisfied = False

    # Generate button with loading state
    if st.button("🚀 Generate Content", type="primary") or (st.session_state.generated_content_result and not st.session_state.content_satisfied): # Regenerate if not satisfied
        st.session_state.content_satisfied = False # Reset satisfaction on new generation
        input_vars = {
                        "campaign_name": campaign_name,
            "promotion_link": promotion_link,
            "previous_campaign_reference": previous_campaign_reference,
            "sku": sku,
            "product_category": product_category,
            "campaign_date_range": campaign_date_range,
            "age_range": f"{age_range[0]}-{age_range[1]}" if age_range else None,
            "gender": ", ".join(gender) if gender else None,
            "income_level": income_level if income_level else None,
            "region": ", ".join(region) if region else None,
            "urban_rural": ", ".join(urban_rural) if urban_rural else None,
            "specific_instructions": specific_instructions,
            "brand": selected_brand,
            "tone_style": tone_style,
            "search_results": None,
            "template_type": template_type,
            "output_format": output_format
        }

        # Validate inputs
        is_valid, error_message = validate_inputs(input_vars)
        if not is_valid:
            st.error(error_message)
            st.stop()

        # Validate date range
        if not validate_date_range(campaign_date_range):
            st.error("Invalid date range. End date must be after start date.")
            st.stop()

        # Generate content with progress bar and enhanced loading message
        with st.spinner(display_loading_message()):  # Use the dynamic loading message
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate work
                progress_bar.progress(i + 1)


            generation_result = generate_content_workflow(
                input_vars, model_name, temperature, top_p, output_format, use_rag, specific_instructions, use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key
            )

            if "error" in generation_result:
                st.error(f"Failed to generate content: {generation_result['error']}")
                st.stop()

            st.session_state.generated_content_result = generation_result["content"]

        display_generated_content(st.session_state.generated_content_result, selected_brand, generate_image, image_style, openai_api_key, campaign_name)


    # Enhanced Feedback and Regeneration Section
    if st.session_state.generated_content_result:
        col1, col2 = st.columns([1, 3])  # Adjust column widths
        with col1:
            if st.button("👍 Satisfied"):
                st.session_state.content_satisfied = True
                st.success("Content marked as satisfactory!")

        with col2:
            if not st.session_state.content_satisfied and st.button("🔄 Regenerate", type="secondary"):
                st.info("Click '🚀 Generate Content' button above to regenerate, or adjust parameters.")

        # Save and Download Section - Only visible after successful generation and satisfaction
        if st.session_state.content_satisfied:
            st.subheader("Save & Download")
            col1, col2 = st.columns(2)
            with col1:
                save_format = st.selectbox("Save Format", ["txt", "json"], key="save_format_select")
            with col2:
                if st.button("💾 Save Content", key="save_content_button"):
                    saved_file = save_content_to_file(
                        st.session_state.generated_content_result, campaign_name, save_format
                    )
                    if saved_file:
                        st.success(f"Content saved to: {saved_file}")
                        st.session_state.campaign_history.append(
                            f"{campaign_name} ({datetime.now().strftime('%Y-%m-%d')})"
                        )
                        # Provide a download link immediately after saving
                        with open(saved_file, "rb") as file:
                            st.download_button(
                                label="⬇️ Download Now",
                                data=file,
                                file_name=os.path.basename(saved_file),
                                mime=f"text/{save_format}",  # Dynamic MIME type
                            )

if __name__ == "__main__":
    main()