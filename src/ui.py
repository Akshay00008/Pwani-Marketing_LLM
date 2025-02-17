import streamlit as st
import time
from config import configure_streamlit_page, load_api_keys, load_css
from data import BRAND_OPTIONS
from prompt import create_prompt_template
from llm import get_llm
from workflow import create_langraph_workflow
from utils import validate_inputs, save_content_to_file, load_campaign_template, validate_date_range
from image import generate_product_image, save_generated_image
from data import SocialMediaContent, EmailContent, MarketingContent
import json
from datetime import datetime
from rag import RAGSystem
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from chat_history import clear_chat_history, delete_message

def initialize_rag_system(openai_api_key):
    """Initialize RAG system, loading documents only once."""
    if 'rag_system' not in st.session_state:
        try:
            llm = get_llm(openai_api_key, "gpt-4", temperature=0)
            st.session_state.rag_system = RAGSystem(llm)

            # Check if the vector store exists.  If not, load documents.
            if not hasattr(st.session_state.rag_system, 'vector_store') or st.session_state.rag_system.vector_store is None:
                st.info("🔄 Loading knowledge base...")
                loader = TextLoader("/Users/vishalroy/Downloads/ContentGenApp/cleaned_cleaned_output.txt")  # Replace with your actual path
                documents = loader.load()
                if st.session_state.rag_system.ingest_documents(documents):
                    st.success("✨ RAG system initialized successfully")
                else:
                    # This shouldn't normally happen, but keep it for safety.
                    st.warning("RAG system initialization skipped - will proceed without context")
            else:
                st.info("✨ Using existing RAG knowledge base")

        except Exception as e:
            # It's better to proceed without RAG than to completely fail.
            st.warning(f"RAG system initialization skipped: {str(e)} - will proceed without context")


def apply_template_defaults(template_type):
    """Apply default values from a template if selected."""
    if template_type != "Custom Campaign":
        template_data = load_campaign_template(template_type)
        # Iterate through the template data and apply defaults.
        for key, value in template_data.items():
            if key in st.session_state and st.session_state.get(key) is not None:
                st.session_state[key] = value



def generate_content_workflow(input_vars, model_name, temperature, top_p, output_format, use_rag, rag_query, use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key):
    """Handles the content generation workflow."""

    # Get the correct LLM based on model name (handles OpenAI and Google models).
    llm = get_llm(google_api_key if not model_name.startswith("gpt") else openai_api_key, model_name, temperature, top_p)
    if not llm:
        return {"error": "Failed to initialize LLM."}

    # Create the prompt template.
    prompt = create_prompt_template(
        instruction="Generate marketing campaign content",
        output_format=output_format,
        use_search_engine=use_search_engine,
        search_engine_prompt_template=search_engine_query
    )

    try:
        # Use RAG if enabled and a query is provided.
        if use_rag and rag_query:
            context_query = f"""
            Brand: {selected_brand}
            Product: {input_vars.get('sku', 'N/A')}
            Category: {input_vars.get('product_category', 'N/A')}
            Query: {rag_query}
            """
            rag_context = st.session_state.rag_system.query(context_query)
            if rag_context:
                input_vars["rag_context"] = rag_context

        # Create the Langraph workflow.
        workflow = create_langraph_workflow(
            llm,
            prompt,
            input_vars,
            output_format,
            use_search_engine,
            search_engine_query if search_engine_query else None  # Only pass if not None
        )

        # Invoke the workflow and get the result.
        result = workflow.invoke(input_vars)
        if "error" in result:
            return {"error": result["error"]}
        generated_content = result["output"]
        return {"content": generated_content}

    except Exception as e:
        return {"error": str(e)}



def display_generated_content(generated_content, selected_brand, generate_image, image_style, openai_api_key, campaign_name):
    """Displays generated content with campaign name context."""

    if isinstance(generated_content, str):
        st.text(generated_content)
    elif isinstance(generated_content, MarketingContent):
        content = f"Campaign: {campaign_name}\n\n{generated_content.headline}\n\n{generated_content.body}\n\nCall to Action: {generated_content.call_to_action}\n\nKey Benefits:\n"
        content += '\n'.join([f"- {benefit}" for benefit in generated_content.key_benefits])
        st.text(content)
    elif isinstance(generated_content, SocialMediaContent):
        content = f"Campaign: {campaign_name}\nPlatform: {generated_content.platform}\nPost Text: {generated_content.post_text}\nHashtags: {', '.join(generated_content.hashtags)}\nCall to Action: {generated_content.call_to_action}"
        st.text(content)
    elif isinstance(generated_content, EmailContent):
        content = f"Campaign: {campaign_name}\nSubject Line: {generated_content.subject_line}\nPreview Text: {generated_content.preview_text}\nBody: {generated_content.body}\nCall to Action: {generated_content.call_to_action}"
        st.text(content)
    elif isinstance(generated_content, dict):
        generated_content['campaign_name'] = campaign_name  # Ensure campaign_name is included
        st.text(json.dumps(generated_content, indent=2))  # Use json.dumps for pretty printing
    else:
        st.text(str(generated_content))  # Fallback for unexpected types

    # Generate image if requested
    if generate_image:
        with st.spinner("🎨 Generating product image..."):
            description = ""
            if isinstance(generated_content, MarketingContent):
                description = f"Campaign: {campaign_name}. {generated_content.headline}. {generated_content.body}"
            elif isinstance(generated_content, str):
                # Limit length to avoid overly long descriptions.
                description = f"Campaign: {campaign_name}. {generated_content[:500]}"
            # ... (rest of your image generation code) ...

            image_url = generate_product_image(
                selected_brand,
                description,
                image_style,
                openai_api_key
            )

            if image_url:
                st.image(image_url, caption=f"{selected_brand} - {campaign_name} Product Image")
                if st.button("💾 Save Image"):
                    saved_image_path = save_generated_image(image_url, f"{selected_brand}_{campaign_name}")
                    if saved_image_path:
                        st.success(f"Image saved to: {saved_image_path}")
            else:
                st.error("Failed to generate image. Please try again.")

    return True  # Explicit return is good practice.

def initialize_chatbot(model_name, temperature, openai_api_key, google_api_key):
    """Initializes the chatbot conversation chain."""
    llm = get_llm(google_api_key if not model_name.startswith("gpt") else openai_api_key, model_name, temperature)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory()
        )


def handle_chat_input(user_input, model_name, temperature, openai_api_key, google_api_key, input_vars):
    """Handles user input, generating responses or creating content based on context."""

    if 'conversation' not in st.session_state:
        initialize_chatbot(model_name, temperature, openai_api_key, google_api_key)

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Check if the user is asking for content generation
        if "generate" in user_input.lower() or "create" in user_input.lower():
            with st.spinner("🎨 Generating your marketing content..."):
                generation_result = generate_content_workflow(
                    input_vars, model_name, temperature, st.session_state.get('top_p', 0.9),
                    st.session_state.get('output_format', 'Text'),
                    st.session_state.get('use_rag', False),
                    user_input if st.session_state.get('use_rag', False) else None,
                    st.session_state.get('use_search_engine', False),
                    st.session_state.get('search_engine_query', None),
                    st.session_state.get('selected_brand'), openai_api_key, google_api_key
                )

                if "error" in generation_result:
                    full_response = f"Failed to generate content: {generation_result['error']}"
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})  # Add error to chat
                else:
                    st.session_state.generated_content_result = generation_result["content"]
                    # Directly display *only* the generated content.
                    display_generated_content(
                        st.session_state.generated_content_result,
                        st.session_state.get('selected_brand', 'Default Brand'),
                        st.session_state.get('generate_image', False),
                        st.session_state.get('image_style', 'Realistic'),
                        openai_api_key,
                        st.session_state.get('campaign_name', 'Unnamed Campaign')
                    )
                    # Don't add anything else to the chatbot.  The content display is handled above.
                    st.session_state.messages.append({"role": "assistant", "content": "Content generated."})

        else:  # General Q&A and Content Refinement
            with st.spinner("Thinking..."):
                # Build rich context including campaign details, input variables, and current content
                context = f"User Input: {user_input}\n\n"

                # Add campaign and input variables context
                campaign_context = f"Campaign Name: {input_vars.get('campaign_name')}\n"
                campaign_context += f"Brand: {input_vars.get('brand')}\n"
                campaign_context += f"Product Category: {input_vars.get('product_category')}\n"
                campaign_context += f"SKU: {input_vars.get('sku')}\n"
                campaign_context += f"Tone & Style: {input_vars.get('tone_style')}\n"
                context += f"Campaign Context: {campaign_context}\n"

                # Add current content for refinement
                if 'generated_content_result' in st.session_state:
                    context += f"Current Content: {str(st.session_state.generated_content_result)}\n"

                # Add specific instructions for content refinement
                context += "\nInstructions: If the user is asking to refine or modify the content, please provide specific suggestions "
                context += "and explain the reasoning behind them. Focus on maintaining brand voice and campaign objectives."

                response = st.session_state.conversation.predict(input=context)

                # Check if response contains actionable refinements
                if "REFINED_CONTENT:" in response:
                    refined_content = response.split("REFINED_CONTENT:")[1].strip()
                    st.session_state.generated_content_result = refined_content
                    #st.success("✨ Content updated based on your feedback!")  # Removed success message
                    display_generated_content(
                        refined_content,
                        st.session_state.get('selected_brand', 'Default Brand'),
                        st.session_state.get('generate_image', False),
                        st.session_state.get('image_style', 'Realistic'),
                        openai_api_key,
                        st.session_state.get('campaign_name', 'Unnamed Campaign')
                    )
                else: #if no refined content still continue the normal flow
                     full_response = response
                     message_placeholder.markdown(full_response)
                     st.session_state.messages.append({"role": "assistant", "content": full_response})





def main():
    configure_streamlit_page()
    load_css()

    # Load API Keys once
    google_api_key, openai_api_key = load_api_keys()

    # Initialize RAG system
    initialize_rag_system(openai_api_key)

    # Initialize default values for target market fields
    if 'target_market_defaults' not in st.session_state:
        st.session_state.target_market_defaults = {
            'age_range': (25, 45),
            'gender': ["Male", "Female"],
            'income_level': "Middle", 
            'region': ["Nairobi", "Mombasa", "Kisumu"],
            'urban_rural': ["Urban", "Suburban"]
        }

    # Improve chat interface styling
    st.markdown("""
        <style>
        .stChatFloatingInputContainer {
            position: fixed;
            bottom: 20px;
            width: calc(100% - 80px);
            z-index: 999;
        }
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 60px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize tab states
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0  # Start with the first tab (index 0)

    # Sidebar setup
    with st.sidebar:
        st.header("📊 Campaign Tools")
        template_type = st.selectbox(
            "Select Campaign Type",
            ["Custom Campaign", "Product Launch", "Seasonal Sale", "Brand Awareness"],
            on_change=apply_template_defaults,  # Apply template on change
            args=[st.session_state.get("template_type", "Custom Campaign")]  # Pass current value to set default
        )
        st.session_state["template_type"] = template_type

        # Campaign History
        st.subheader("Recent Campaigns")
        if "campaign_history" not in st.session_state:
            st.session_state.campaign_history = []
        for campaign in st.session_state.campaign_history[-5:]:  # Show last 5
            st.text(f"📄 {campaign}")


    # Main content
    st.title("🌟 Pwani Oil Marketing Content Generator")
    st.caption("Generate professional marketing content powered by AI")

     # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Tab Navigation Logic ---
    def next_tab():
        st.session_state.current_tab = (st.session_state.current_tab + 1) % 3  # Cycle through tabs (0, 1, 2)

    def prev_tab():
        st.session_state.current_tab = (st.session_state.current_tab - 1) % 3

    # --- Tab Content ---
    # Use st.container to group elements within each tab.  This is crucial for tab switching.
    if st.session_state.current_tab == 0:
        with st.container():  # Tab 1: Campaign Details
            st.header("Campaign Details")  # Add a header for each tab
            col1, col2 = st.columns(2)
            with col1:
                campaign_name = st.text_input(
                    "Campaign Name",
                    key="campaign_name",
                    help="Enter a unique name for your campaign",
                )

                # Replace the existing brand selection with new dropdown and description
                selected_brand = st.selectbox(
                    "Brand",
                    options=list(BRAND_OPTIONS.keys()),
                    help="Select the brand for the campaign"
                )

                if selected_brand:
                    st.info(f"📝 **Brand Description:** {BRAND_OPTIONS[selected_brand]}")

                promotion_link = st.text_input(
                    "Promotion Link",
                    key="promotion_link",
                    help="Enter the landing page URL",
                )
                previous_campaign_reference = st.text_input(
                    "Previous Campaign Reference", key="previous_campaign_reference"
                )
            # In tab1, under col2
            with col2:
                sku = st.selectbox(
                    "SKU",
                    ["500L", "250L", "1L", "10L", "20L", "2L", "3L", "5L", "10KG", "500G", "1KG", "2KG", "17KG", "4KG", "100G", "700G", "800G", "600G", "80G", "125G", "175G", "200G", "225G", "20G"],
                    key="sku",
                    help="Select the product SKU number"
                )
                product_category = st.selectbox(
                    "Product Category",
                    ["Cooking Oil", "Cooking Fat", "Bathing Soap", "Home Care", "Lotion", "Margarine", "Medicine Soap"],
                    key="product_category"
                )
                campaign_date_range = st.text_input(
                    "Campaign Date Range (YYYY-MM-DD to YYYY-MM-DD)",
                    key="campaign_date_range",
                )
                tone_style = st.selectbox(
                    "Tone & Style",
                    [
                        "Professional",
                        "Casual",
                        "Friendly",
                        "Humorous",
                        "Formal",
                        "Inspirational",
                        "Educational",
                        "Persuasive",
                        "Emotional"
                    ],
                    key="tone_style_tab1",  # Unique key
                    help="Select the tone and style for your content"
                )

            col1, _, col2 = st.columns([1, 2, 1])
            with col2:
                if st.button("Next →", key="tab1_next"):
                    next_tab()



    elif st.session_state.current_tab == 1:
        with st.container():  # Tab 2: Target Market
            st.header("Target Market")
            col1, col2 = st.columns(2)
            with col1:
                age_range = st.select_slider(
                    "Age Range",
                    options=list(range(18, 76, 1)),
                    value=st.session_state.target_market_defaults['age_range'],
                    key="age_range_slider"
                )
                gender = st.multiselect(
                    "Gender",
                    ["Male", "Female", "Other"],
                    default=st.session_state.target_market_defaults['gender'],
                    key="gender_multiselect"
                )
            with col2:
                income_level = st.select_slider(
                    "Income Level",
                    options=["Low", "Middle Low", "Middle", "Middle High", "High"],
                    value=st.session_state.target_market_defaults['income_level'],
                    key="income_level_slider"
                )
                region = st.multiselect(
                    "Region",
                    ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"],
                    default=st.session_state.target_market_defaults['region'],
                    key="region_multiselect"
                )
                urban_rural = st.multiselect(
                    "Area Type",
                    ["Urban", "Suburban", "Rural"],
                    default=st.session_state.target_market_defaults['urban_rural'],
                    key="urban_rural_multiselect"
                )
            col1, _, col2 = st.columns([1, 2, 1])
            with col1:
                if st.button("← Previous", key="tab2_prev"):
                    prev_tab()
            with col2:
                if st.button("Next →", key="tab2_next"):
                    next_tab()


    elif st.session_state.current_tab == 2:
        with st.container():  # Tab 3: Advanced Settings
            st.header("Advanced Settings")  # Add a header
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.selectbox(
                    "Model",
                    ["gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp","gemini-2.0-flash-thinking-exp-01-21"],
                    help="Select the AI model to use",
                    key="model_name_select"
                )
                output_format = st.selectbox(
                    "Output Format",
                    ["Social Media", "Email", "Marketing", "Text"],
                    help="Choose the type of content to generate",
                    key="output_format_select"
                )

                # Add RAG option
                use_rag = st.checkbox("Use RAG System", value=True, help="Use Retrieval Augmented Generation for better context", key="use_rag_checkbox")

                # Add image generation options
                generate_image = st.checkbox("Generate Product Image", value=False, key="generate_image_checkbox")
                image_style = st.selectbox(
                    "Image Style",
                    ["Realistic", "Artistic", "Modern", "Classic"],
                    help="Select the style for the generated image",
                    key="image_style_select"
                ) if generate_image else None # Conditional selectbox

                # Add search engine option
                use_search_engine = st.checkbox("Use Web Search", value=False, help="Incorporate live web search results into the content", key="use_search_engine_checkbox")
                search_engine_query = st.text_input("Search Query", help="Enter the search query for the web search engine", key="search_query_input") if use_search_engine else None

                # Content requirements
                st.subheader("Content Requirements")
                specific_instructions = st.text_area(
                    "Specific Instructions",
                    help="Enter any specific requirements or guidelines for the content",
                    key="specific_instructions_input"
                )

            with col2:
                temperature = st.slider(
                    "Creativity Level",
                    0.0,
                    1.0,
                    0.7,
                    help="Higher values = more creative output",
                    key="temperature_slider"
                )
                top_p = st.slider(
                    "Diversity Level",
                    0.0,
                    1.0,
                    0.9,
                    help="Higher values = more diverse output",
                    key="top_p_slider"
                )

            col1, _, col2 = st.columns([1, 2, 1])
            with col1:
                if st.button("← Previous", key="tab3_prev"):
                    prev_tab()


    # --- Common elements (outside tab containers) ---

    # State to manage content generation and satisfaction
    if "generated_content_result" not in st.session_state:
        st.session_state.generated_content_result = None
    if "content_satisfied" not in st.session_state:
        st.session_state.content_satisfied = False

    # Gather input variables.  This needs to happen *after* all the input widgets.
    input_vars = {
        "campaign_name": st.session_state.get("campaign_name", ""),  # Use .get for safety
        "promotion_link": st.session_state.get("promotion_link", ""),
        "previous_campaign_reference": st.session_state.get("previous_campaign_reference", ""),
        "sku": st.session_state.get("sku", ""),
        "product_category": st.session_state.get("product_category", ""),
        "campaign_date_range": st.session_state.get("campaign_date_range", ""),
        "age_range": f"{st.session_state.get('age_range_slider', [25,45])[0]}-{st.session_state.get('age_range_slider',[25,45])[1]}" if st.session_state.get("use_age_range_tab2") else None,
        "gender": ", ".join(st.session_state.get("gender_multiselect", [])) if st.session_state.get("use_gender_tab2") else None,
        "income_level": st.session_state.get("income_level_slider", "Middle") if st.session_state.get("use_income_level_tab2") else None,
        "region": ", ".join(st.session_state.get("region_multiselect", [])) if st.session_state.get("use_region_tab2") else None,
        "urban_rural": ", ".join(st.session_state.get("urban_rural_multiselect", [])) if st.session_state.get("use_urban_rural_tab2") else None,
        "specific_instructions": st.session_state.get("specific_instructions_input", ""),
        "brand": st.session_state.get("selected_brand", ""),
        "tone_style": st.session_state.get("tone_style_tab1", ""),
        "search_results": None,  # Placeholder for search results
        "template_type": st.session_state.get("template_type", "Custom Campaign"),
        "output_format": st.session_state.get("output_format_select", "Text")
    }

     # Move validation logic here, outside of any tab
    if st.session_state.current_tab == 0:  # Only validate on tab 0
        is_valid, error_message = validate_inputs(input_vars)
        if not is_valid:
            st.error(error_message)
            # Prevent moving to the next tab by not calling next_tab()
            st.stop()  # Stop execution so the rest of the code doesn't run

        if not validate_date_range(st.session_state.get("campaign_date_range", "")):
            st.error("Invalid date range. End date must be after start date.")
            # Prevent moving to the next tab
            st.stop()


    # Generate content button (can be outside the tabs)
    if st.button("🎨 Generate Content", type="primary"):
        with st.spinner("🎨 Generating your marketing content..."):
            generation_result = generate_content_workflow(
                input_vars,
                st.session_state.get('model_name_select', 'gpt-4'), # Default to gpt-4
                st.session_state.get('temperature_slider', 0.7),
                st.session_state.get('top_p_slider', 0.9),
                st.session_state.get('output_format_select', 'Text'),
                st.session_state.get('use_rag', True),
                None,  # No RAG query on initial generation
                st.session_state.get('use_search_engine', False),
                st.session_state.get('search_query_input', None),
                st.session_state.get('selected_brand'), openai_api_key, google_api_key
            )
            if "error" in generation_result:
                st.error(f"Failed to generate content: {generation_result['error']}")
            else:
                st.session_state.generated_content_result = generation_result["content"]
                display_generated_content(
                    st.session_state.generated_content_result,
                    st.session_state.get('selected_brand', 'Default Brand'),  # Provide defaults
                    st.session_state.get('generate_image', False),
                    st.session_state.get('image_style_select', 'Realistic'),
                    openai_api_key,
                    st.session_state.get('campaign_name', 'Unnamed Campaign')
                )


    # Chat interface and other elements (place these outside the tab containers)
    if st.session_state.generated_content_result:
        st.subheader("💬 Refine Your Content")
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            # Add Clear All button
            if st.button("🗑️ Clear All Messages", key="clear_all"):
                clear_chat_history()
                st.rerun()

            # Display chat messages with delete buttons
            for idx, message in enumerate(st.session_state.messages):
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                with col2:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        delete_message(idx)
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            # Fixed position chat input
            if user_input := st.chat_input("Ask questions or provide instructions to refine the content..."):
                handle_chat_input(user_input, st.session_state.get("model_name_select", "gpt-4"),
                                st.session_state.get("temperature_slider", 0.7),
                                openai_api_key, google_api_key, input_vars)


        # Content actions - Removed satisfaction and regeneration buttons
        # Save content options
        st.subheader("Save Options")
        col1, col2 = st.columns(2)
        with col1:
            save_format = st.selectbox("Save Format", ["txt", "json"], key="save_format_selectbox")
        with col2:
            if st.button("💾 Save Content", key="save_content_button"):
                saved_file = save_content_to_file(
                    st.session_state.generated_content_result, st.session_state.get('campaign_name', 'Unnamed_Campaign'), save_format
                )
                if saved_file:
                    st.success(f"Content saved to: {saved_file}")
                    st.session_state.campaign_history.append(
                        f"{st.session_state.get('campaign_name', 'Unnamed_Campaign')} ({datetime.now().strftime('%Y-%m-%d')})"
                    )


if __name__ == "__main__":
    main()
