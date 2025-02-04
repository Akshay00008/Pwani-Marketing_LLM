import streamlit as st
import os
from dotenv import load_dotenv
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
from langgraph.graph import StateGraph, END
import re
from typing import List, Optional
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Pwani Oil Marketing Generator",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Define multiple content types for different marketing needs
class SocialMediaContent(BaseModel):
    platform: str = Field(
        description="Social media platform (Facebook, Instagram, Twitter)"
    )
    post_text: str = Field(description="Main post content")
    hashtags: List[str] = Field(description="Relevant hashtags")
    call_to_action: str = Field(description="Call to action text")


class EmailContent(BaseModel):
    subject_line: str = Field(description="Email subject line")
    preview_text: str = Field(description="Email preview text")
    body: str = Field(description="Main email body")
    call_to_action: str = Field(description="Call to action button text")


class MarketingContent(BaseModel):
    headline: str = Field(description="The main headline for the marketing content")
    body: str = Field(description="The main body of the marketing message")
    call_to_action: str = Field(description="A clear call to action")
    target_audience: str = Field(description="Description of the target audience")
    key_benefits: List[str] = Field(description="Key benefits of the product")


# Custom CSS for better styling
def load_css():
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


# Validate input fields
def validate_inputs(inputs: dict) -> tuple[bool, str]:
    if not inputs["campaign_name"].strip():
        return False, "Campaign name is required"

    if not inputs["sku"].strip():
        return False, "SKU is required"

    # Validate date range format
    date_pattern = r"^\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}$"
    if not re.match(date_pattern, inputs["campaign_date_range"]):
        return False, "Date range must be in format YYYY-MM-DD to YYYY-MM-DD"

    # Validate URL format
    if inputs["promotion_link"].strip():
        url_pattern = r"^[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z]{2,}(/\S*)?$"
        if not re.match(url_pattern, inputs["promotion_link"]):
            return False, "Invalid promotion link format"

    return True, ""


def get_llm(
    api_key: str, model_name: str, temperature: float = 0.7, top_p: float = 0.9
):
    """Initialize LLM with error handling"""
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None


def save_content_to_file(
    content, campaign_name: str, format: str = "txt"
) -> Optional[str]:
    """Save generated content with error handling and multiple format support"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{campaign_name.replace(' ', '_')}_{timestamp}.{format}"

    try:
        with open(file_name, "w", encoding="utf-8") as f:
            if format == "json":
                json.dump(
                    content.dict() if hasattr(content, "dict") else content,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            else:
                f.write(str(content))
        return file_name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


# Add these functions to the main code


def create_prompt_template(
    instruction="Generate marketing content", output_format="text"
):
    """Creates a prompt template based on the content type"""
    base_template = """
    {instruction} for Pwani Oil Limited based on the following details:

    Company Overview:
    "You are a Marketing Content Generator LLM designed to generate tailored marketing content for Pwani Oil Products Limited. The content will be based on the product categories and the list of brands and their definitions provided below. The content generation will also consider the campaign details, target market details, and specific features/instructions provided by the user.
    Brands and Their Definitions:
    Diria – A premium cooking oil brand known for its high quality and versatility in cooking.
    Frymate – A trusted cooking oil brand ideal for frying, delivering great taste and performance.
    Mpishi Poa – A cooking oil brand that offers superior quality at an affordable price, perfect for everyday use.
    Pwani SBF – Specially formulated for high-performance frying with longer-lasting oil quality.
    Super Chef – A popular cooking oil brand used by professional chefs for exceptional frying results.
    Criso – A cooking oil brand that ensures purity and health with every meal.
    Fresh Fri – A leading cooking oil brand that provides freshness and quality, enhancing every meal.
    Fresh Zait – A premium cooking oil made from high-quality ingredients, perfect for healthier cooking.
    Popco – A cooking oil brand known for its excellent performance and affordable pricing.
    Salit – A cooking oil brand that offers great taste and quality, trusted by many Kenyan households.
    Tiku – A reliable cooking oil brand that provides purity and exceptional cooking results.
    Twiga – A cooking oil brand known for its great value and high quality for everyday cooking needs.
    Onja – A trusted brand offering quality oils with a focus on taste and performance.
    Ndume – A cooking oil brand that combines quality and affordability for everyday use.
    Whitewash – A home care brand offering effective cleaning solutions with outstanding performance.
    4U – A home care product line that delivers excellent results in cleaning with a focus on customer satisfaction.
    Belleza – A personal care brand offering luxurious skincare products for a refined experience.
    Fresco – A personal care brand known for offering effective and gentle beauty products.
    Sawa – A trusted personal care brand offering a variety of soaps for hygiene and skincare.
    Afrisense – A personal care brand providing a wide range of deodorants and fragrances.
    Detrex – A personal care brand specializing in hygiene products with a focus on quality.
    Diva – A personal care brand offering beauty and grooming products for a sophisticated lifestyle.
    Ushindi – A brand providing quality hygiene products that cater to everyday needs and ensure freshness.
    Objective:
    Based on the above brands and their definitions, generate marketing content for the following categories:
    Product Description: Generate product descriptions highlighting features, benefits, and unique selling points.
    Campaign Slogans: Create catchy slogans for advertising campaigns.
    Social Media Content: Write engaging posts that attract customer attention and interaction.
    Email Marketing Copy: Generate email content for promotions, product launches, or updates.
    Promotional Offers: Suggest promotional content with discounts, special offers, or incentives.
    The generated content should be creative, informative, and in line with the brand image of Pwani Oil Products Limited, resonating with the target audience.

    Campaign Details:
    Campaign Name: {campaign_name}
    Promotion Reference Link: {promotion_link}
    Previous Campaign Reference: {previous_campaign_reference}
    Brand: Pwani Oil
    SKU: {sku}
    Campaign Category: {campaign_category}
    Campaign Type: {campaign_type}
    Campaign Date Range: {campaign_date_range}

    Target Market Details:
    Age Range: {age_range}
    Gender: {gender}
    Income Level: {income_level}
    Region: {region}
    Urban/Rural: {urban_rural}

    Specific Features/Instructions:
    Generate content that aligns with the campaign details, target market preferences, and specific instructions, while maintaining consistency with Pwani Oil’s core values of quality, sustainability, and innovation."
    {specific_instructions}
    """

    if output_format == "Social Media":
        parser = PydanticOutputParser(pydantic_object=SocialMediaContent)
        template = base_template + "\n{format_instructions}"
        return PromptTemplate(
            template=template,
            input_variables=[
                "campaign_name",
                "promotion_link",
                "previous_campaign_reference",
                "sku",
                "campaign_category",
                "campaign_type",
                "campaign_date_range",
                "age_range",
                "gender",
                "income_level",
                "region",
                "urban_rural",
                "specific_instructions",
            ],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "instruction": instruction,
            },
        )
    elif output_format == "Email":
        parser = PydanticOutputParser(pydantic_object=EmailContent)
        template = base_template + "\n{format_instructions}"
        return PromptTemplate(
            template=template,
            input_variables=[
                "campaign_name",
                "promotion_link",
                "previous_campaign_reference",
                "sku",
                "campaign_category",
                "campaign_type",
                "campaign_date_range",
                "age_range",
                "gender",
                "income_level",
                "region",
                "urban_rural",
                "specific_instructions",
            ],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "instruction": instruction,
            },
        )
    else:
        template = base_template  # Assign the base template here
        return PromptTemplate(
            template=template,
            input_variables=[
                "campaign_name",
                "promotion_link",
                "previous_campaign_reference",
                "sku",
                "campaign_category",
                "campaign_type",
                "campaign_date_range",
                "age_range",
                "gender",
                "income_level",
                "region",
                "urban_rural",
                "specific_instructions",
            ],
            partial_variables={"instruction": instruction},
        )


def create_langraph_workflow(llm, prompt, input_vars, output_format):
    """Creates a LangGraph workflow with error handling and retries"""

    def generate_content(state):
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                if output_format == "Social Media":
                    parser = PydanticOutputParser(pydantic_object=SocialMediaContent)
                    _input = prompt.format(**state)
                    response = llm.invoke(_input)
                    output = parser.parse(response.content)
                elif output_format == "Email":
                    parser = PydanticOutputParser(pydantic_object=EmailContent)
                    _input = prompt.format(**state)
                    response = llm.invoke(_input)
                    output = parser.parse(response.content)
                else:
                    _input = prompt.format(**state)
                    response = llm.invoke(_input)
                    output = response.content

                return {"output": output}
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    st.error(
                        f"Failed to generate content after {max_retries} attempts: {str(e)}"
                    )
                    return {"output": None}
                time.sleep(1)  # Wait before retrying

    workflow = StateGraph(dict)
    workflow.add_node("generate_content", generate_content)
    workflow.set_entry_point("generate_content")
    workflow.add_edge("generate_content", END)

    return workflow.compile()


# Add these utility functions for template management


def load_campaign_template(template_type: str) -> dict:
    """Loads predefined campaign templates"""
    templates = {
        "Product Launch": {
            "campaign_category": "New Product",
            "campaign_type": "Multi-channel",
            "specific_instructions": "Create a product launch campaign highlighting unique features, benefits, and introductory offer. Include social media posts, email announcement, and main marketing message.",
        },
        "Seasonal Sale": {
            "campaign_category": "Promotional",
            "campaign_type": "Social Media",
            "specific_instructions": "Create a seasonal sale campaign with time-limited offers, urgency messaging, and clear price benefits. Focus on social media engagement and shareable content.",
        },
        "Brand Awareness": {
            "campaign_category": "Branding",
            "campaign_type": "Display Ads",
            "specific_instructions": "Create a brand awareness campaign focusing on company values, quality commitment, and community impact. Include emotional storytelling elements.",
        },
    }
    return templates.get(template_type, {})


def validate_date_range(date_range: str) -> bool:
    """Validates the date range format and logic"""
    try:
        start_date, end_date = date_range.split(" to ")
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        return end > start
    except Exception as e:
        print(e)
        return False


def main():
    load_dotenv()
    load_css()

    # Sidebar for campaign templates and history
    with st.sidebar:
        st.header("📊 Campaign Tools")

        # Campaign Templates
        st.subheader("Templates")
        template_type = st.selectbox(
            "Select Campaign Type",
            ["Custom Campaign", "Product Launch", "Seasonal Sale", "Brand Awareness"],
        )

        # Campaign History
        st.subheader("Recent Campaigns")
        if "campaign_history" not in st.session_state:
            st.session_state.campaign_history = []

        for campaign in st.session_state.campaign_history[-5:]:
            st.text(f"📄 {campaign}")

    # Main content
    st.title("🌟 Pwani Oil Marketing Content Generator")
    st.caption("Generate professional marketing content powered by AI")

    # Load API Key with better error handling
    api_key = os.getenv("Gemini_API_KEY")
    if not api_key:
        st.error("🔑 API Key not found. Please set Gemini_API_KEY in your .env file")
        st.stop()

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
            promotion_link = st.text_input(
                "Promotion Link",
                key="promotion_link",
                help="Enter the landing page URL",
            )
            previous_campaign_reference = st.text_input(
                "Previous Campaign Reference", key="previous_campaign_reference"
            )
        with col2:
            sku = st.text_input("SKU", key="sku", help="Product SKU number")
            campaign_category = st.selectbox(
                "Campaign Category",
                ["Household Essentials", "Cooking Oil", "Personal Care", "Other"],
            )
            campaign_type = st.selectbox(
                "Campaign Type",
                ["Social Media", "Email Marketing", "Display Ads", "Multi-channel"],
            )
            campaign_date_range = st.text_input(
                "Campaign Date Range (YYYY-MM-DD to YYYY-MM-DD)",
                key="campaign_date_range",
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            age_range = st.select_slider(
                "Age Range", options=list(range(18, 76, 1)), value=(25, 45)
            )
            gender = st.multiselect(
                "Gender", ["Male", "Female", "Other"], default=["Female"]
            )
        with col2:
            income_level = st.select_slider(
                "Income Level",
                options=["Low", "Middle Low", "Middle", "Middle High", "High"],
                value="Middle",
            )
            region = st.multiselect(
                "Region",
                ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"],
                default=["Nairobi", "Mombasa"],
            )
            urban_rural = st.multiselect(
                "Area Type", ["Urban", "Suburban", "Rural"], default=["Urban"]
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox(
                "Model",
                ["gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
                help="Select the AI model to use",
            )
            output_format = st.selectbox(
                "Output Format",
                ["Social Media", "Email", "General Marketing"],
                help="Choose the type of content to generate",
            )
        with col2:
            temperature = st.slider(
                "Creativity Level",
                0.0,
                1.0,
                0.7,
                help="Higher values = more creative output",
            )
            top_p = st.slider(
                "Diversity Level",
                0.0,
                1.0,
                0.9,
                help="Higher values = more diverse output",
            )

    # Content requirements
    st.subheader("Content Requirements")
    specific_instructions = st.text_area(
        "Specific Instructions",
        help="Enter any specific requirements or guidelines for the content",
    )

    # Generate button with loading state
    if st.button("🚀 Generate Content", type="primary"):
        input_vars = {
            "campaign_name": campaign_name,
            "promotion_link": promotion_link,
            "previous_campaign_reference": previous_campaign_reference,
            "sku": sku,
            "campaign_category": campaign_category,
            "campaign_type": campaign_type,
            "campaign_date_range": campaign_date_range,
            "age_range": f"{age_range[0]}-{age_range[1]}",
            "gender": ", ".join(gender),
            "income_level": income_level,
            "region": ", ".join(region),
            "urban_rural": ", ".join(urban_rural),
            "specific_instructions": specific_instructions,
        }

        # Validate inputs
        is_valid, error_message = validate_inputs(input_vars)
        if not is_valid:
            st.error(error_message)
            st.stop()

        # Generate content with progress bar
        with st.spinner("🎨 Generating your marketing content..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            llm = get_llm(api_key, model_name, temperature, top_p)
            if not llm:
                st.stop()

            try:
                workflow = create_langraph_workflow(
                    llm, create_prompt_template(), input_vars, output_format
                )
                result = workflow.invoke(input_vars)
                generated_content = result["output"]

                if generated_content:
                    st.success("✨ Content generated successfully!")
            # Handle image generation
                # Image generation removed as handle_image_generation is not defined
                    selected_brand=selected_brand,
                    brand_description=selected_brand,
                    openai_api_key=openai_api_key
                )
                    # Display content in a formatted way
                    st.subheader("Generated Content")
                    st.markdown("---")
                    st.markdown(
                        generated_content
                        if isinstance(generated_content, str)
                        else json.dumps(generated_content.dict(), indent=2)
                    )

                    # Save options
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
                                # Update campaign history
                                st.session_state.campaign_history.append(
                                    f"{campaign_name} ({datetime.datetime.now().strftime('%Y-%m-%d')})"
                                )

            except Exception as e:
                st.error(f"Error generating content: {str(e)}")

        # Apply template if selected
    if template_type != "Custom Campaign":
        template_data = load_campaign_template(template_type)
        for key, value in template_data.items():
            if key in st.session_state:
                st.session_state[key] = value

    # Additional validation before content generation
    if not validate_date_range(campaign_date_range):
        st.error("Invalid date range. End date must be after start date.")
        st.stop()


if __name__ == "__main__":
    main()
