# streamlit_ui.py
import streamlit as st
import time
import logging
import re
import os
import json
from datetime import datetime
from typing import Optional, List
from langchain_community.document_loaders import TextLoader
from langchain.output_parsers import PydanticOutputParser

from config import configure_streamlit_page, load_api_keys, load_css
from data import BRAND_OPTIONS, SocialMediaContent, EmailContent, MarketingContent
from utils import validate_inputs, save_content_to_file, load_campaign_template, validate_date_range # No generate_content_with_retries here anymore
from llm import get_llm, search_tool
from rag import RAGSystem # Import RAGSystem
from agent_workflow import create_agent_workflow # Relative import
from agent_state import AgentState # Relative import


# --- Streamlit UI Functions ---
# In the initialize_rag_system function
def initialize_rag_system_streamlit(openai_api_key):
    """Initialize RAG system in Streamlit session state, loading documents only once."""
    if 'rag_system_initialized' not in st.session_state:  # Flag to track initialization
        try:
            llm = get_llm(openai_api_key, "gpt-4", temperature=0)
            rag_system = RAGSystem(llm)

            if not rag_system.vector_store:  # Check vector_store directly
                st.info("🔄 Loading knowledge base...")
                loader = TextLoader("/Users/vishalroy/Downloads/ContentGenApp/cleaned_output_simple.txt")
                documents = loader.load()
                if rag_system.ingest_documents(documents):
                    st.success("✨ RAG system initialized successfully")
                else:
                    st.warning("RAG system initialization failed.")
                    return None  # Indicate failure
            else:
                st.info("✨ Using existing RAG knowledge base")

            st.session_state.rag_system_instance = rag_system  # Store RAG instance
            st.session_state.rag_system_initialized = True  # Set initialization flag
            return rag_system  # Return the initialized instance

        except Exception as e:
            st.warning(f"RAG system initialization skipped: {str(e)} - will proceed without context")
            st.session_state.rag_system_initialized = True  # Set flag to avoid retrying
            return None  # Indicate failure


def apply_template_defaults(template_type):
    """Apply default values from a template if selected."""
    if template_type != "Custom Campaign":
        template_data = load_campaign_template(template_type)
        for key, value in template_data.items():
            if key in st.session_state:
                st.session_state[key] = value


# REMOVE the function definition of generate_content_with_retries from here, it's now in utils.py


def main():
    configure_streamlit_page()
    load_css()

    # Load API Keys once
    google_api_key, openai_api_key = load_api_keys()

    # Initialize RAG system in Streamlit session state
    rag_system_instance = initialize_rag_system_streamlit(openai_api_key) # Get RAG instance

    # Initialize LangGraph workflow
    if 'agent_workflow' not in st.session_state:
        st.session_state.agent_workflow = create_agent_workflow()
    agent_workflow = st.session_state.agent_workflow
    
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
    st.title("🌟 Pwani Oil Marketing Content Generator (Agentic)") # Updated title
    st.caption("Generate professional marketing content powered by AI")

    # Tabs for different content types
    tab1, tab2, tab3 = st.tabs(
        ["Campaign Details", "Target Market", "Advanced Settings"]
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
                    ["Cooking Oil", "Personal Care", "Home Care"],
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
                if st.checkbox("Add Region", key="use_region_multiselect")
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
                ["gpt-4", "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
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
            image_style = None  # Initialize image_style with None
            if generate_image:
                image_style = st.selectbox(
                    "Image Style",
                    ["Realistic", "Artistic", "Modern", "Classic"],
                    help="Select the style for the generated image",
                    key="image_style_select"
                )
            # Add search engine option
            use_search_engine = st.checkbox("Use Web Search", value=False, help="Incorporate live web search results into the content", key="use_search_engine_checkbox")
            search_engine_query = None  # Initialize with None
            if use_search_engine:
                search_engine_query = st.text_input("Search Query", help="Enter the search query for the web search engine", key="search_query_input")

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

    # Generate button with loading state
    if st.button("🚀 Generate Content", type="primary"):
        agent_state_input = AgentState( # Initialize AgentState object
            campaign_name=campaign_name,
            promotion_link=promotion_link,
            previous_campaign_reference=previous_campaign_reference,
            sku=sku,
            product_category=product_category,
            campaign_date_range=campaign_date_range,
            age_range=f"{age_range[0]}-{age_range[1]}" if age_range else None,
            gender=", ".join(gender) if gender else None,
            income_level=income_level if income_level else None,
            region=", ".join(region) if region else None,
            urban_rural=", ".join(urban_rural) if urban_rural else None,
            specific_instructions=specific_instructions,
            brand=selected_brand,
            tone_style=tone_style,
            output_format=output_format,
            rag_query=rag_query,
            use_rag=use_rag,
            use_search_engine=use_search_engine,
            search_engine_query=search_engine_query,
            generate_image_checkbox=generate_image, # Pass image generation flag
            image_style_select=image_style, # Pass image style
            model_name_select=model_name, # Pass model name
            temperature_slider=temperature, # Pass temperature
            top_p_slider=top_p # Pass top_p
        )

        # Validate inputs
        is_valid, error_message = validate_inputs(agent_state_input.dict()) # Validate state as dict
        if not is_valid:
            st.error(error_message)
            st.stop()

        # Validate date range
        if not validate_date_range(campaign_date_range):
            st.error("Invalid date range. End date must be after start date.")
            st.stop()

        # Run LangGraph agent workflow
        with st.spinner("🎨 Generating your marketing content..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            try:
                # Invoke LangGraph workflow - Pass initial state
                agent_result = st.session_state.agent_workflow.invoke(agent_state_input)
                
                final_state = agent_result['final_state'] # Access final state from output
                generated_content = final_state.generated_content
                image_url = final_state.image_url

                if not generated_content:
                    st.error("No content was generated. Please try again.")
                    st.stop()

                # Display generated content
                st.success("✨ Content generated successfully!")
                st.subheader("Generated Content")
                st.markdown("---")

                # Display content based on type - Use final_state.generated_content
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

                # Generate image if requested and display - Use final_state.image_url
                if generate_image and image_url:
                    from image import save_generated_image # Local import to avoid circular dependency
                    st.subheader("Generated Image")
                    st.image(image_url, caption=f"{selected_brand} Product Image")
                    if st.button("💾 Save Image"):
                        saved_image_path = save_generated_image(image_url, selected_brand)
                        if saved_image_path:
                            st.success(f"Image saved to: {saved_image_path}")
                elif generate_image and not image_url:
                    st.error("Failed to generate image. Please try again.")


                # Save content options
                col1, col2 = st.columns(2)
                with col1:
                    save_format = st.selectbox("Save Format", ["txt", "json"])
                with col2:
                    if st.button("💾 Save Content"):
                        saved_file = save_content_to_file(
                            generated_content, campaign_name, save_format
                        )
                        if saved_file:
                            st.success(f"Content saved to: {saved_file}")
                            st.session_state.campaign_history.append(
                                f"{campaign_name} ({datetime.now().strftime('%Y-%m-%d')})"
                            )

            except Exception as e:
                st.error(f"Error generating content: {str(e)}")
                st.error(e)


if __name__ == "__main__":
    main()