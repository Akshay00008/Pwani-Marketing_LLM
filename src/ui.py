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


# In the initialize_rag_system function
def initialize_rag_system(openai_api_key):
    """Initialize RAG system, loading documents only once."""
    if 'rag_system' not in st.session_state:
        try:
            llm = get_llm(openai_api_key, "gpt-4", temperature=0)
            st.session_state.rag_system = RAGSystem(llm)

            if not hasattr(st.session_state.rag_system, 'vector_store') or st.session_state.rag_system.vector_store is None:
                # st.info("🔄 Loading knowledge base...")
                loader = TextLoader("cleaned_cleaned_output.txt")
                documents = loader.load()
                # if st.session_state.rag_system.ingest_documents(documents):
                #     st.success("✨ RAG system initialized successfully")
                # else:
                #     st.warning("RAG system initialization skipped - will proceed without context")
            else:
                st.info("✨ Using existing RAG knowledge base")

        except Exception as e:
            st.warning(f"RAG system initialization skipped: {str(e)} - will proceed without context")


def apply_template_defaults(template_type):
    """Apply default values from a template if selected."""
    if template_type != "Custom Campaign":
        template_data = load_campaign_template(template_type)
        for key, value in template_data.items():
            if key in st.session_state:
                st.session_state[key] = value

def generate_content_workflow(input_vars, model_name, temperature, top_p, output_format, use_rag, rag_query, use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key):
    """Handles the content generation workflow."""
    llm = get_llm(google_api_key if not model_name.startswith("gpt") else openai_api_key, model_name, temperature, top_p)
    if not llm:
        return {"error": "Failed to initialize LLM."}

    prompt = create_prompt_template(
        instruction="Generate marketing campaign content",
        output_format=output_format,
        use_search_engine=use_search_engine,
        search_engine_prompt_template=search_engine_query
    )

    try:
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
            return {"error": result["error"]}
        generated_content = result["output"]
        return {"content": generated_content}

    except Exception as e:
        return {"error": str(e)}


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

def initialize_chatbot(model_name, temperature, openai_api_key, google_api_key):
    """Initializes the chatbot conversation chain."""
    llm = get_llm(google_api_key if not model_name.startswith("gpt") else openai_api_key, model_name, temperature)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory()
        )

def handle_chat_input(user_input, model_name, temperature, openai_api_key, google_api_key):
    """Handles user input in the chatbot, generating and displaying responses."""
    if 'conversation' not in st.session_state:
        initialize_chatbot(model_name, temperature, openai_api_key, google_api_key)

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from conversation chain
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder for the full response
        full_response = ""
        with st.spinner("Thinking..."):
             # Directly use the predict method, as it interacts with memory correctly
            response = st.session_state.conversation.predict(input=user_input)
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)  # Final update without cursor

    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    configure_streamlit_page()
    load_css()

    # Load API Keys once
    google_api_key, openai_api_key = load_api_keys()

    # Initialize RAG system
    initialize_rag_system(openai_api_key)


    # Sidebar setup
    with st.sidebar:
        st.header("📊 Campaign Tools")
        template_type = st.selectbox(
            "Select Campaign Type",
            ["Custom Campaign", "Product Launch", "Seasonal Sale", "Brand Awareness"],
            on_change=apply_template_defaults,  # Apply template on change
            args=[st.session_state.get("template_type", "Custom Campaign")] # Pass current value to set default in function
        )
        st.session_state["template_type"] = template_type

        # Campaign History
        st.subheader("Recent Campaigns")
        if "campaign_history" not in st.session_state:
            st.session_state.campaign_history = []
        for campaign in st.session_state.campaign_history[-5:]:
            st.text(f"📄 {campaign}")

    # Main content
    st.title("🌟 Pwani Oil Marketing Content Generator")
    st.caption("Generate professional marketing content powered by AI")

     # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Tabs for different content types
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Campaign Details", "Target Market", "Advanced Settings", "Chatbot"]
    )

    with tab1:
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
                    key="tone_style_tab1",
                    help="Select the tone and style for your content"
                )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            age_range = (
                st.select_slider(
                    "Age Range", options=list(range(18, 76, 1)), value=(25, 45),
                     key="age_range_slider" #Unique Key
                )
                if st.checkbox("Add Age Range", key="use_age_range_tab2")
                else None
            )
            gender = (
                st.multiselect(
                    "Gender", ["Male", "Female", "Other"], default=["Female"],
                    key="gender_multiselect" #Unique Key
                )
                if st.checkbox("Add Gender", key="use_gender_tab2")
                else None
            )
        with col2:
            income_level = (
                st.select_slider(
                    "Income Level",
                    options=["Low", "Middle Low", "Middle", "Middle High", "High"],
                    value="Middle",
                    key="income_level_slider" #Unique Key
                )
                if st.checkbox("Add Income Level", key="use_income_level_tab2")
                else None
            )
            region = (
                st.multiselect(
                    "Region",
                    ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"],
                    default=["Nairobi", "Mombasa"],
                     key="region_multiselect" #Unique Key
                )
                if st.checkbox("Add Region", key="use_region_tab2")
                else None
            )
            urban_rural = (
                st.multiselect(
                    "Area Type", ["Urban", "Suburban", "Rural"], default=["Urban"],
                    key="urban_rural_multiselect" #Unique Key
                )
                if st.checkbox("Add Area Type", key="use_urban_rural_tab2")
                else None
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox(
                "Model",
                ["gpt-4", "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp","gemini-2.0-flash-thinking-exp-01-21"],
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


    # Content requirements
    st.subheader("Content Requirements")
    specific_instructions = st.text_area(
        "Specific Instructions",
        help="Enter any specific requirements or guidelines for the content",
        key="specific_instructions_input"
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

        # Generate content with progress bar
        with st.spinner("🎨 Generating your marketing content..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            generation_result = generate_content_workflow(
                input_vars, model_name, temperature, top_p, output_format, use_rag, specific_instructions, use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key
            )

            if "error" in generation_result:
                st.error(f"Failed to generate content: {generation_result['error']}")
                st.stop()

            st.session_state.generated_content_result = generation_result["content"]

        display_generated_content(st.session_state.generated_content_result, selected_brand, generate_image, image_style, openai_api_key, campaign_name)


    if st.session_state.generated_content_result:
        col1, col2 = st.columns([1, 3]) # Adjust column widths as needed
        with col1:
            if st.button("👍 Satisfied"):
                st.session_state.content_satisfied = True
                st.success("Content marked as satisfactory!")

        with col2:
            if not st.session_state.content_satisfied and st.button("🔄 Regenerate", type="secondary"):
                st.info("Click '🚀 Generate Content' button above to regenerate with current settings or adjust parameters.")


        if st.session_state.content_satisfied:
            # Save content options only when satisfied
            st.subheader("Save Options")
            col1, col2 = st.columns(2)
            with col1:
                save_format = st.selectbox("Save Format", ["txt", "json"], key="save_format_selectbox") # Add key
            with col2:
                if st.button("💾 Save Content", key="save_content_button"): # Add key
                    saved_file = save_content_to_file(
                        st.session_state.generated_content_result, campaign_name, save_format
                    )
                    if saved_file:
                        st.success(f"Content saved to: {saved_file}")
                        st.session_state.campaign_history.append(
                            f"{campaign_name} ({datetime.now().strftime('%Y-%m-%d')})"
                        )

    with tab4:
        st.subheader("🤖 Chat with your Marketing Assistant")

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if user_input := st.chat_input("Ask a question about your campaign or content..."):
            handle_chat_input(user_input, model_name, temperature, openai_api_key, google_api_key)


if __name__ == "__main__":
    main()