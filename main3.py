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
from typing import Dict, Any
import logging
import openai  # Add this import
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    target_audience: str = Field(description="Description of the target audience")


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
    target_audience: str = Field(description="Description of the target audience") 
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
    if inputs["campaign_date_range"] and not re.match(
        date_pattern, inputs["campaign_date_range"]
    ):
        return False, "Date range must be in format YYYY-MM-DD to YYYY-MM-DD"

    # Validate URL format
    if inputs["promotion_link"] and inputs["promotion_link"].strip():
        url_pattern = r"^[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z]{2,}(/\S*)?$"
        if not re.match(url_pattern, inputs["promotion_link"]):
            return False, "Invalid promotion link format"

    return True, ""

def get_llm(
    api_key: str, model_name: str, temperature: float = 0.7, top_p: float = 0.9
):
    """Initialize LLM with error handling"""
    try:
        if model_name.startswith("gpt"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
            )
        else:
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

def create_prompt_template(
 instruction="""
"You are a Marketing Content Generator LLM designed to generate tailored and catchy plus out of the box marketing content for Pwani Oil Products Limited. The content will be based on the product categories and the list of brands and their definitions provided below or you can visit this website url https://pwani.net to get more information.The content generation will also consider the campaign details, target market details, and specific features/instructions provided by the user.
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
""", output_format="text", use_search_engine=False, search_engine_prompt_template=None
):
 # """
 # Creates a prompt template with modular sections and proper optional field handling.
 # Args:
 #     instruction (str): The main instruction for content generation
 #     output_format (str): Desired output format ("Social Media", "Email", "Marketing", or "text")
 # """
 base_template = """
 {instruction} for {brand} based on the following details:
 
 Campaign Details:
 Campaign Name: {campaign_name}
 Brand: {brand}
 SKU: {sku}
 Campaign Category: {campaign_category}
 Campaign Type: {campaign_type}
 """

 # Add optional campaign details
 campaign_details = """
 Additional Campaign Information:
 Promotion Reference Link: {promotion_link}
 Previous Campaign Reference: {previous_campaign_reference}
 Campaign Date Range: {campaign_date_range}
 """
 # Add target market details
 market_details = """
 Target Market Details:
 Age Range: {age_range}
 Gender: {gender}
 Income Level: {income_level}
 Region: {region}
 Urban/Rural: {urban_rural}
 """

 # Add specific instructions section
 instructions_section = """
 Specific Features/Instructions:
 {specific_instructions}
 Generate content that aligns with the campaign details, target market preferences, and specific instructions, while maintaining consistency with Pwani Oil’s core values of quality, sustainability, and innovation."
 """

 # Add content type and tone details section
 content_style_section = """
 Content Specifications:
 Content Type: {content_type}
 Tone and Style: {tone_style}
 """

 # Combine template sections
 full_template = (
     base_template + campaign_details + market_details + content_style_section + instructions_section
 )
 
 # Conditionally add search results
 if use_search_engine and search_engine_prompt_template:
     search_results_section = """
     Web Search Results:
     {search_results}
     """
     full_template += search_results_section
     
 # Handle different output formats
 if output_format in ["Social Media", "Email", "Marketing"]:
     parser_map = {
         "Social Media": PydanticOutputParser(pydantic_object=SocialMediaContent),
         "Email": PydanticOutputParser(pydantic_object=EmailContent),
         "Marketing": PydanticOutputParser(pydantic_object=MarketingContent),
     }
     parser = parser_map[output_format]
     full_template += "\n{format_instructions}"

     return PromptTemplate(
         template=full_template,
         input_variables=[
             "brand",
             "campaign_name",
             "sku",
             "campaign_category",
             "campaign_type",
             "promotion_link",
             "previous_campaign_reference",
             "campaign_date_range",
             "age_range",
             "gender",
             "income_level",
             "region",
             "urban_rural",
             "specific_instructions",
             "content_type",
             "tone_style",
              "search_results",
         ],
         partial_variables={
             "format_instructions": parser.get_format_instructions(),
             "instruction": instruction
             + " Output in strict JSON format. Do not include any introductory text or descriptive labels. Ensure that ALL required fields as defined in the provided JSON schema are present in the output. For URL placeholders use {promotion_link}.",
         },
     )

 if output_format in ["Social Media", "Email", "Marketing"]:
     json_instruction = """
     CRITICAL JSON FORMATTING REQUIREMENTS:
     1. Output must be a single, valid JSON object
     2. All property names must be in double quotes
     3. String values must use double quotes
     4. Apostrophes within text must be escaped (e.g., "Pwani Oil\\'s" not "Pwani Oil's")
     5. No trailing commas
     6. No additional text or formatting outside the JSON object
     7. Must exactly match this schema:
     {format_instructions}
     """
     full_template += json_instruction

 return PromptTemplate(
     template=full_template,
     input_variables=[
         "brand",
         "campaign_name",
         "sku",
         "campaign_category",
         "campaign_type",
         "promotion_link",
         "previous_campaign_reference",
         "campaign_date_range",
         "age_range",
         "gender",
         "income_level",
         "region",
         "urban_rural",
         "specific_instructions",
         "content_type",    # Add content_type
         "tone_style",       # Add tone_style
          "search_results",  # Add search results
     ],
     partial_variables={
         "format_instructions": parser.get_format_instructions()
         if output_format in ["Social Media", "Email", "Marketing"]
         else "",
         "instruction": instruction,
     },
 )

def generate_content_with_retries(llm, prompt, input_vars, output_format, use_search_engine=False, search_engine_query=None):
    max_retries = 3
    retry_count = 0
    parser = None

    if output_format in ["Social Media", "Email", "Marketing"]:
        parser_map = {
            "Social Media": PydanticOutputParser(pydantic_object=SocialMediaContent),
            "Email": PydanticOutputParser(pydantic_object=EmailContent),
            "Marketing": PydanticOutputParser(pydantic_object=MarketingContent),
        }
        parser = parser_map[output_format]

    while retry_count < max_retries:
        try:
            # If using search engine
            if use_search_engine and search_engine_query:
                logging.info(f"Performing web search with query: {search_engine_query}")
                search_results = search_tool.run(search_engine_query)
                logging.info("Search Results:")
                logging.info("-" * 50)
                logging.info(search_results)
                logging.info("-" * 50)
                input_vars["search_results"] = search_results
            else:
                logging.info("No web search performed")
                input_vars["search_results"] = "No search terms were provided"
            
            formatted_prompt = prompt.format(**input_vars)
            formatted_prompt += "\nIMPORTANT: Return ONLY a valid JSON object with no additional text or formatting."
            
            response = llm.invoke(formatted_prompt)
            response_text = response.content

            if parser:
                try:
                    # Clean the response text
                    response_text = re.sub(r"```(?:json)?\s*|\s*```", "", response_text)
                    response_text = " ".join(response_text.split())

                    # Extract JSON object
                    json_match = re.search(r"\{.*\}", response_text)
                    if not json_match:
                        raise ValueError("No JSON object found in response")
                    json_str = json_match.group()

                    # Handle apostrophes before JSON parsing
                    def escape_apostrophes(match):
                        text = match.group(1)
                        # Escape any apostrophes within the quoted text
                        text = text.replace("'", "\\'")
                        return f'"{text}"'

                    # Replace content within double quotes, handling apostrophes
                    json_str = re.sub(r'"([^"]*)"', escape_apostrophes, json_str)

                    # Normalize property names - Fix the regex pattern
                    json_str = re.sub(
                        r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str
                    )
                    
                    # Remove any remaining unescaped apostrophes
                    json_str = json_str.replace("'", "\\'")

                    # Clean up any double-escaped quotes
                    json_str = json_str.replace('\\"', '"')

                    # Ensure proper spacing
                    json_str = re.sub(r",\s*([^\s])", r", \1", json_str)
                    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

                    try:
                        parsed_json = json.loads(json_str)
                    except json.JSONDecodeError as je:
                        # Add detailed logging for debugging
                        logging.error(f"JSON decode error position {je.pos}: {je.msg}")
                        logging.error(
                            f"Character at position: {json_str[je.pos-5:je.pos+5]}"
                        )
                        logging.error(f"Full JSON string: {json_str}")
                        raise

                    return parser.parse(json.dumps(parsed_json))

                except (json.JSONDecodeError, ValueError) as e:
                    logging.error(f"JSON parsing error: {str(e)}")
                    logging.error(f"Raw response: {response_text}")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(1)
                        continue
                    raise

            return response_text

        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            if retry_count < max_retries - 1:
                retry_count += 1
                time.sleep(1)
                continue
            raise

    return None

def create_langraph_workflow(llm, prompt, input_vars, output_format, use_search_engine=False, search_engine_query=None):
    def generate_content(state):
        try:
            output = generate_content_with_retries(llm, prompt, state, output_format, use_search_engine, search_engine_query)  # Removed extra input_vars argument
            return {"output": output}
        except Exception as e:
            logging.error(f"Workflow error: {str(e)}")
            return {"error": str(e)}

    workflow = StateGraph(dict)
    workflow.add_node("generate_content", generate_content)
    workflow.set_entry_point("generate_content")
    workflow.add_edge("generate_content", END)

    return workflow.compile()

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
    if not date_range:
        return True
    try:
        start_date, end_date = date_range.split(" to ")
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        return end > start
    except Exception as e:
        print(e)
        return False

# Define brand options with their descriptions
BRAND_OPTIONS = {
    "Diria": "A premium cooking oil brand known for its high quality and versatility in cooking",
    "Frymate": "A trusted cooking oil brand ideal for frying, delivering great taste and performance",
    "Mpishi Poa": "A cooking oil brand that offers superior quality at an affordable price, perfect for everyday use",
    "Pwani SBF": "Specially formulated for high-performance frying with longer-lasting oil quality",
    "Super Chef": "A popular cooking oil brand used by professional chefs for exceptional frying results",
    "Criso": "A cooking oil brand that ensures purity and health with every meal",
    "Fresh Fri": "A leading cooking oil brand that provides freshness and quality, enhancing every meal",
    "Fresh Zait": "A premium cooking oil made from high-quality ingredients, perfect for healthier cooking",
    "Popco": "A cooking oil brand known for its excellent performance and affordable pricing",
    "Salit": "A cooking oil brand that offers great taste and quality, trusted by many Kenyan households",
    "Tiku": "A reliable cooking oil brand that provides purity and exceptional cooking results",
    "Twiga": "A cooking oil brand known for its great value and high quality for everyday cooking needs",
    "Onja": "A trusted brand offering quality oils with a focus on taste and performance",
    "Ndume": "A cooking oil brand that combines quality and affordability for everyday use",
    "Whitewash": "A home care brand offering effective cleaning solutions with outstanding performance",
    "4U": "A home care product line that delivers excellent results in cleaning with a focus on customer satisfaction",
    "Belleza": "A personal care brand offering luxurious skincare products for a refined experience",
    "Fresco": "A personal care brand known for offering effective and gentle beauty products",
    "Sawa": "A trusted personal care brand offering a variety of soaps for hygiene and skincare",
    "Afrisense": "A personal care brand providing a wide range of deodorants and fragrances",
    "Detrex": "A personal care brand specializing in hygiene products with a focus on quality",
    "Diva": "A personal care brand offering beauty and grooming products for a sophisticated lifestyle",
    "Ushindi": "A brand providing quality hygiene products that cater to everyday needs and ensure freshness"
}

def main():
    load_dotenv()
    load_css()

    # Load API Keys once
    google_api_key = os.getenv("Gemini_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not google_api_key and not openai_api_key:
        st.error("🔑 No API Keys found. Please set either Gemini_API_KEY or OPENAI_API_KEY in your .env file")
        st.stop()

    # Sidebar setup
    with st.sidebar:
        st.header("📊 Campaign Tools")
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
                sku = st.text_input("SKU", key="sku", help="Product SKU number")
                product_category = st.selectbox(
                    "Product Category",
                    ["Cooking Oil", "Personal Care", "Home Care"],
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
                    key="tone_style_tab1",  # Added unique key
                    help="Select the tone and style for your content"
                )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            age_range = (
                st.select_slider(
                    "Age Range", options=list(range(18, 76, 1)), value=(25, 45)
                )
                if st.checkbox("Add Age Range", key="use_age_range")
                else None
            )
            gender = (
                st.multiselect(
                    "Gender", ["Male", "Female", "Other"], default=["Female"]
                )
                if st.checkbox("Add Gender", key="use_gender")
                else None
            )
        with col2:
            income_level = (
                st.select_slider(
                    "Income Level",
                    options=["Low", "Middle Low", "Middle", "Middle High", "High"],
                    value="Middle",
                )
                if st.checkbox("Add Income Level", key="use_income_level")
                else None
            )
            region = (
                st.multiselect(
                    "Region",
                    ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"],
                    default=["Nairobi", "Mombasa"],
                )
                if st.checkbox("Add Region", key="use_region")
                else None
            )
            urban_rural = (
                st.multiselect(
                    "Area Type", ["Urban", "Suburban", "Rural"], default=["Urban"]
                )
                if st.checkbox("Add Area Type", key="use_urban_rural")
                else None
            )

    # Remove the duplicate tone_style selectbox from tab3
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox(
                "Model",
                ["gpt-4", "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
                help="Select the AI model to use",
            )
            output_format = st.selectbox(
                "Output Format",
                ["Social Media", "Email", "Marketing", "Text"],
                help="Choose the type of content to generate",
            )
            # Add image generation options
            generate_image = st.checkbox("Generate Product Image", value=False)
            if generate_image:
                image_style = st.selectbox(
                    "Image Style",
                    ["Realistic", "Artistic", "Modern", "Classic"],
                    help="Select the style for the generated image"
                )
            # Add search engine option
            use_search_engine = st.checkbox("Use Web Search", value=False, help="Incorporate live web search results into the content")
            if use_search_engine:
                search_engine_query = st.text_input("Search Query", help="Enter the search query for the web search engine")

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
            # Removed duplicate tone_style selectbox from here

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
            prompt = create_prompt_template(
                instruction="Generate marketing campaign content",
                output_format=output_format,
                use_search_engine=use_search_engine,
                search_engine_prompt_template=search_engine_query
            )

            try:
                workflow = create_langraph_workflow(
                    llm, prompt, input_vars, output_format, use_search_engine, search_engine_query
                )
                result = workflow.invoke(input_vars)
                if "error" in result:
                    st.error(f"Failed to generate content: {result['error']}")
                    st.stop()
                generated_content = result["output"]

                if not generated_content:
                    st.error("No content was generated. Please try again.")
                    st.stop()

                # Display generated content
                st.success("✨ Content generated successfully!")
                st.subheader("Generated Content")
                st.markdown("---")

                # Display content based on type
                if isinstance(generated_content, str):
                    st.markdown(generated_content)
                elif isinstance(generated_content, MarketingContent):
                    st.subheader(generated_content.headline)
                    st.write(generated_content.body)
                    st.markdown(f"**Target Audience:** {generated_content.target_audience}")
                    st.markdown(f"**Call to Action:** {generated_content.call_to_action}")
                    st.markdown("**Key Benefits:**")
                    for benefit in generated_content.key_benefits:
                        st.markdown(f"- {benefit}")
                elif isinstance(generated_content, SocialMediaContent):
                    st.markdown(f"**Platform:** {generated_content.platform}")
                    st.markdown(f"**Post Text:** {generated_content.post_text}")
                    st.markdown(f"**Hashtags:** {', '.join(generated_content.hashtags)}")
                    st.markdown(f"**Call to Action:** {generated_content.call_to_action}")
                    st.markdown(f"**Target Audience:** {generated_content.target_audience}")
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
    def save_generated_image(image_url: str, brand_name: str) -> str:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{brand_name}_image_{timestamp}.png"
            
            os.makedirs("images", exist_ok=True)
            filepath = os.path.join("images", filename)
            
            img.save(filepath)
            logging.info(f"Image saved successfully: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Unexpected error while saving image: {str(e)}")
            return None

    def generate_product_image(brand_name: str, description: str, style: str, openai_api_key: str) -> str:
        try:
            client = OpenAI(api_key=openai_api_key)
            
            # Extract key marketing points from the description
            marketing_content = description[:1000]  # Limit description length for DALL-E
            
            base_prompt = f"""Create a professional marketing image for {brand_name} brand that captures:

{marketing_content}

The image should be suitable for marketing and advertising purposes."""
            
            style_details = {
                "Realistic": "Create a photorealistic product shot with professional studio lighting, clean white background, and commercial-grade presentation",
                "Artistic": "Design a creative and artistic interpretation with elegant design elements, unique composition, and eye-catching visual appeal",
                "Modern": "Generate a contemporary design with bold colors, clean lines, and minimalist aesthetics that appeals to modern consumers",
                "Classic": "Produce a traditional product photography style with timeless appeal, perfect lighting, and professional composition"
            }
            
            full_prompt = f"{base_prompt}\n\nStyle requirements: {style_details.get(style, '')}.\nEnsure the image is high quality and suitable for commercial marketing use."
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            if response and response.data and len(response.data) > 0:
                return response.data[0].url
            return None
                
        except Exception as e:
            logging.error(f"Image generation error: {str(e)}")
            return None

    main()