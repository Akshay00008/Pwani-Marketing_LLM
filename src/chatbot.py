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


def initialize_rag_system(openai_api_key):
    """Initialize RAG system, loading documents only once."""
    if 'rag_system' not in st.session_state:
        try:
            llm = get_llm(openai_api_key, "gpt-4", temperature=0)
            st.session_state.rag_system = RAGSystem(llm)

            if not hasattr(st.session_state.rag_system, 'vector_store') or st.session_state.rag_system.vector_store is None:
                st.info("🔄 Loading knowledge base...")
                loader = TextLoader("cleaned_cleaned_output.txt")#C:\Users\hp\OneDrive - Algo8.ai\Marketing_Content\ContentGenApp\cleaned_cleaned_output.txt
                documents = loader.load()
                if st.session_state.rag_system.ingest_documents(documents):
                    st.success("✨ RAG system initialized successfully")
                else:
                    st.warning("RAG system initialization skipped - will proceed without context")
            else:
                st.info("✨ Using existing RAG knowledge base")

        except Exception as e:
            st.warning(f"RAG system initialization skipped: {str(e)} - will proceed without context")


def apply_template_defaults(template_type):
    """Apply default values from a template."""
    if template_type != "Custom Campaign":
        template_data = load_campaign_template(template_type)
        if template_data:  # Check if template_data is not None
            for key, value in template_data.items():
                if key in st.session_state:
                    st.session_state[key] = value
        else:
            st.warning(f"Template '{template_type}' not found.")

def generate_initial_context(input_vars, model_name, temperature, top_p, use_rag, rag_query, use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key):
    """Generates an initial context message *formatted as a chatbot message*."""
    llm = get_llm(google_api_key if not model_name.startswith("gpt") else openai_api_key, model_name, temperature, top_p)
    if not llm:
        return "Error: Failed to initialize LLM."

    # Create a more conversational prompt.  This is the key change.
    prompt_template = create_prompt_template(
        instruction="""You are a helpful marketing assistant chatbot.  The user has provided details about a new marketing campaign.  
        Summarize these details in a friendly and conversational way, as if you are introducing the campaign to the user. 
        Mention the key aspects, but keep it concise. Be enthusiastic!""",
        output_format="Text",
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

        workflow = create_langraph_workflow(llm, prompt_template, input_vars, "Text", use_search_engine, search_engine_query)
        result = workflow.invoke(input_vars)

        if "error" in result:
            return f"Error generating context: {result['error']}"

        generated_context = result.get("output", "No context generated.")
        if isinstance(generated_context, dict) and "text" in generated_context:
            generated_context = generated_context["text"]
        elif not isinstance(generated_context, str):
            generated_context = str(generated_context)

        return generated_context  # Return the conversational summary

    except Exception as e:
        return f"Error generating context: {str(e)}"


def initialize_chatbot(model_name, temperature, openai_api_key, google_api_key, initial_context=""):
    """Initializes the chatbot with a conversational greeting and context."""
    llm = get_llm(google_api_key if not model_name.startswith("gpt") else openai_api_key, model_name, temperature)
    if 'conversation' not in st.session_state:
        memory = ConversationBufferMemory()
        if initial_context:
            # More natural priming of the memory
            memory.save_context({"input": "Hello!"}, {"output": initial_context})
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory
        )


def handle_chat_input(user_input, model_name, temperature, openai_api_key, google_api_key):
    """Handles user input, generating conversational responses."""
    if 'conversation' not in st.session_state:
        initialize_chatbot(model_name, temperature, openai_api_key, google_api_key)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.predict(input=user_input)
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": full_response})



def main():
    configure_streamlit_page()
    load_css()

    google_api_key, openai_api_key = load_api_keys()
    initialize_rag_system(openai_api_key)

    st.title("🌟 Pwani Oil Marketing Assistant Chatbot")
    st.caption("Provide campaign details, then chat with the AI.")

    with st.sidebar:
        st.header("⚙️ Chatbot Settings")
        model_name = st.selectbox(
            "Model",
            ["gpt-4", "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp-01-21"],
            key="model_name_select"
        )
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, key="temperature_slider")
        use_rag = st.checkbox("Use RAG", value=True, help="Use Retrieval Augmented Generation.")
        use_search_engine = st.checkbox("Use Web Search", value=False, help="Incorporate web search.")
        search_engine_query = st.text_input("Search Query", key="search_query_input") if use_search_engine else None

    with st.expander("📝 Campaign Details", expanded=True):
        template_type = st.selectbox(
            "Campaign Type",
            ["Custom Campaign", "Product Launch", "Seasonal Sale", "Brand Awareness"],
            key="template_type",
            on_change=apply_template_defaults,
            args=["Custom Campaign"]
        )

        col1, col2 = st.columns(2)
        with col1:
            campaign_name = st.text_input("Campaign Name", key="campaign_name")
            selected_brand = st.selectbox("Brand", options=list(BRAND_OPTIONS.keys()))
            if selected_brand:
                st.info(f"📝 Brand Description: {BRAND_OPTIONS[selected_brand]}")
            promotion_link = st.text_input("Promotion Link", key="promotion_link")
            previous_campaign_reference = st.text_input("Previous Campaign Ref", key="previous_campaign_reference")
        with col2:
            sku = st.selectbox("SKU", ["500L", "250L", "1L", "10L", "20L", "2L", "3L", "5L", "10KG", "500G", "1KG", "2KG", "17KG", "4KG", "100G", "700G", "800G", "600G", "80G", "125G", "175G", "200G", "225G", "20G"], key="sku")
            product_category = st.selectbox("Product Category", ["Cooking Oil", "Cooking Fat", "Bathing Soap", "Home Care", "Lotion", "Margarine", "Medicine Soap"], key="product_category")
            campaign_date_range = st.text_input("Date Range (YYYY-MM-DD to YYYY-MM-DD)", key="campaign_date_range")
            tone_style = st.selectbox("Tone & Style", ["Professional", "Casual", "Friendly", "Humorous", "Formal", "Inspirational", "Educational", "Persuasive", "Emotional"], key="tone_style")

        col1, col2 = st.columns(2)
        with col1:
            age_range = st.select_slider("Age Range", options=list(range(18, 76, 1)), value=(25, 45), key="age_range_slider") if st.checkbox("Add Age Range", key="use_age_range") else None
            gender = st.multiselect("Gender", ["Male", "Female", "Other"], default=["Female"], key="gender_multiselect") if st.checkbox("Add Gender", key="use_gender") else None
        with col2:
            income_level = st.select_slider("Income Level", options=["Low", "Middle Low", "Middle", "Middle High", "High"], value="Middle", key="income_level_slider") if st.checkbox("Add Income Level", key="use_income_level") else None
            region = st.multiselect("Region", ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"], default=["Nairobi", "Mombasa"], key="region_multiselect") if st.checkbox("Add Region", key="use_region") else None
            urban_rural = st.multiselect("Area Type", ["Urban", "Suburban", "Rural"], default=["Urban"], key="urban_rural_multiselect") if st.checkbox("Add Area Type", key="use_urban_rural") else None

        specific_instructions = st.text_area("Specific Instructions", key="specific_instructions_input")

        if st.button("🚀 Initialize Chatbot", type="primary"):
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
                "template_type": template_type,  # Now using template_type
                "output_format": "Text"
            }

            is_valid, error_message = validate_inputs(input_vars)
            if not is_valid:
                st.error(error_message)
                st.stop()
            if not validate_date_range(campaign_date_range):
                st.error("Invalid date range.")
                st.stop()

            with st.spinner("Generating initial context..."):
                initial_context = generate_initial_context(
                    input_vars, model_name, temperature, 0.9, use_rag, specific_instructions, use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key
                )

            initialize_chatbot(model_name, temperature, openai_api_key, google_api_key, initial_context)

            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": initial_context})

            st.success("Chatbot initialized! You can now start chatting.")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask me anything about the campaign..."):
            handle_chat_input(user_input, model_name, temperature, openai_api_key, google_api_key)


if __name__ == "__main__":
    main()