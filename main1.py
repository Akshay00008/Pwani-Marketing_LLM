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
# import google.generativeai as genai


load_dotenv()

# Define Pydantic model for Structured Output (example - remove if not using structured output)
class MarketingContent(BaseModel):
    headline: str = Field(description="The main headline for the marketing content.")
    body: str = Field(description="The main body of the marketing message.")
    call_to_action: str = Field(description="A clear call to action.")

# Template for the Pydantic Output Parser
output_parser = PydanticOutputParser(pydantic_object=MarketingContent)

def get_llm(api_key, model_name, temperature=0.7, top_p=0.9):
    """Initializes and configures LangChain's LLM (Gemini)"""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        top_p=top_p
    )

def create_prompt_template(instruction="Generate marketing content", output_format="text"):
        """Creates a LangChain PromptTemplate with structured output."""
        if output_format == "pydantic":
            prompt_template = PromptTemplate(
            template="""
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
            {specific_instructions}
            \n{format_instructions}
            """,
            input_variables=["campaign_name","promotion_link", "previous_campaign_reference", "sku", "campaign_category", "campaign_type", "campaign_date_range",
                             "age_range","gender","income_level","region", "urban_rural", "specific_instructions", "instruction"],
            partial_variables = {"format_instructions": output_parser.get_format_instructions(), "instruction": instruction}
        )
        else:
             prompt_template = PromptTemplate(
            template="""
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
            {specific_instructions}
            """,
            input_variables=["campaign_name","promotion_link", "previous_campaign_reference", "sku", "campaign_category", "campaign_type", "campaign_date_range",
                             "age_range","gender","income_level","region", "urban_rural", "specific_instructions", "instruction"],
            partial_variables = {"instruction": instruction}
             )
        return prompt_template

def generate_content_with_langchain(llm, prompt, input_vars, output_format):
        """Generates content using LangChain LLM."""

        if output_format == 'pydantic':
            try:
                _input = prompt.format(**input_vars)
                response = llm.invoke(_input)
                output = output_parser.parse(response.content)
                return output
            except Exception as e:
                 st.error(f"Error in response parsing: {e}")
                 return None
        else:
            try:
                _input = prompt.format(**input_vars)
                response = llm.invoke(_input)
                return response.content
            except Exception as e:
                 st.error(f"Error in response: {e}")
                 return None


def save_content_to_file(content, campaign_name):
  """Saves the generated content to a text file."""
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  file_name = f"{campaign_name.replace(' ', '_')}_{timestamp}.txt"
  try:
    with open(file_name, "w", encoding="utf-8") as f:
      if isinstance(content, str):
          f.write(content)
      else:
          f.write(json.dumps(content.dict(), indent = 4))
    st.success(f"Content saved to: {file_name}")
  except Exception as e:
    st.error(f"Error saving content: {e}")

def create_langraph_workflow(llm, prompt, input_vars, output_format):
     """Creates a basic LangGraph workflow to generate content"""

     def generate_content(state):
        output = generate_content_with_langchain(llm, prompt, state, output_format)
        return {"output": output}
     workflow = StateGraph(dict)
     workflow.add_node("generate_content", generate_content)
     workflow.set_entry_point("generate_content")
     workflow.add_edge("generate_content", END)
     return workflow.compile()

def main():
    st.title("Pwani Oil Marketing Content Generator (LangChain & LangGraph)")

    # Load API Key
    api_key = os.getenv("Gemini_API_KEY")

    if api_key is None:
      st.error('API Key not found. Make sure you have GOOGLE_API_KEY in .env')
      return

    # Input fields
    with st.expander("Campaign Details", expanded=True):
      col1, col2 = st.columns(2)
      with col1:
        campaign_name = st.text_input("Campaign Name", "Pwani Fresh Start")
        promotion_link = st.text_input("Promotion Reference Link", "pwani-oil.com/fresh-start")
        previous_campaign_reference = st.text_input("Previous Campaign Reference", "PwaniHomeEssentials2023")
      with col2:
        sku = st.text_input("SKU", "PO-SOAP-200")
        campaign_category = st.text_input("Campaign Category", "Household Essentials")
        campaign_type = st.text_input("Campaign Type", "Social Media Awareness and Discount Offer")
        campaign_date_range = st.text_input("Campaign Date Range", "2024-07-01 to 2024-07-31")

    with st.expander("Target Market Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age_range = st.text_input("Age Range", "25-45")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=1)
        with col2:
          income_level = st.text_input("Income Level", "Middle Income")
          region = st.text_input("Region", "Nairobi, Mombasa")
          urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"], index=0)
    specific_instructions = st.text_area("Specific Features/Instructions", "Create 3 short social media posts (Facebook, Instagram, Twitter) focusing on the soap's natural ingredients and its benefits for family hygiene. Include a 15% discount code.")

    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox("Select Gemini Model", ["gemini-pro", "gemini-1.5-pro"], index=0)
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, help="Controls randomness")
            top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05, help="Controls diversity")
            output_format = st.selectbox("Output Format", ["text", "pydantic"], index = 0, help="Select the Output format")

    # Example Prompts
    st.markdown("### Example Prompts")
    if st.checkbox("Use Example Campaign Data"):
        example_campaign = {
           "campaign_name": "Pwani Summer Sale",
            "promotion_link": "pwani-oil.com/summer-sale",
            "previous_campaign_reference": "PwaniBackToSchool2023",
            "sku": "PO-COOKING-2L",
            "campaign_category": "Cooking Oil Promotion",
            "campaign_type": "Discount Offer",
            "campaign_date_range": "2024-08-01 to 2024-08-31",
            "age_range": "20-55",
            "gender": "Any",
            "income_level": "All Income",
            "region": "Kenya Nationwide",
            "urban_rural": "Any",
           "specific_instructions": "Create a short email marketing message with a 20% discount for Pwani Cooking Oil. Highlight its health benefits and suitability for family cooking."
        }
        campaign_name = example_campaign.get("campaign_name")
        promotion_link = example_campaign.get("promotion_link")
        previous_campaign_reference = example_campaign.get("previous_campaign_reference")
        sku = example_campaign.get("sku")
        campaign_category = example_campaign.get("campaign_category")
        campaign_type = example_campaign.get("campaign_type")
        campaign_date_range = example_campaign.get("campaign_date_range")
        age_range = example_campaign.get("age_range")
        gender = example_campaign.get("gender")
        income_level = example_campaign.get("income_level")
        region = example_campaign.get("region")
        urban_rural = example_campaign.get("urban_rural")
        specific_instructions = example_campaign.get("specific_instructions")

        # Pre-fill fields with example data.
        st.session_state["campaign_name"] = campaign_name
        st.session_state["promotion_link"] = promotion_link
        st.session_state["previous_campaign_reference"] = previous_campaign_reference
        st.session_state["sku"] = sku
        st.session_state["campaign_category"] = campaign_category
        st.session_state["campaign_type"] = campaign_type
        st.session_state["campaign_date_range"] = campaign_date_range
        st.session_state["age_range"] = age_range
        st.session_state["gender"] = gender
        st.session_state["income_level"] = income_level
        st.session_state["region"] = region
        st.session_state["urban_rural"] = urban_rural
        st.session_state["specific_instructions"] = specific_instructions

    # Pre-fill session state
    if "campaign_name" not in st.session_state:
       st.session_state["campaign_name"] =  "Pwani Fresh Start"
    if "promotion_link" not in st.session_state:
      st.session_state["promotion_link"] = "pwani-oil.com/fresh-start"
    if "previous_campaign_reference" not in st.session_state:
      st.session_state["previous_campaign_reference"] = "PwaniHomeEssentials2023"
    if "sku" not in st.session_state:
      st.session_state["sku"] = "PO-SOAP-200"
    if "campaign_category" not in st.session_state:
       st.session_state["campaign_category"] = "Household Essentials"
    if "campaign_type" not in st.session_state:
       st.session_state["campaign_type"] = "Social Media Awareness and Discount Offer"
    if "campaign_date_range" not in st.session_state:
       st.session_state["campaign_date_range"] = "2024-07-01 to 2024-07-31"
    if "age_range" not in st.session_state:
       st.session_state["age_range"] =  "25-45"
    if "gender" not in st.session_state:
       st.session_state["gender"] =  "Female"
    if "income_level" not in st.session_state:
        st.session_state["income_level"] = "Middle Income"
    if "region" not in st.session_state:
        st.session_state["region"] = "Nairobi, Mombasa"
    if "urban_rural" not in st.session_state:
       st.session_state["urban_rural"] = "Urban"
    if "specific_instructions" not in st.session_state:
        st.session_state["specific_instructions"] = "Create 3 short social media posts (Facebook, Instagram, Twitter) focusing on the soap's natural ingredients and its benefits for family hygiene. Include a 15% discount code."

    if st.button("Generate Content"):
        # Basic input validation
        if not campaign_name or not sku or not campaign_date_range:
            st.error("Please fill in the required campaign details (Name, SKU, Date Range).")
            return

        input_vars = {
            "campaign_name": st.session_state["campaign_name"],
            "promotion_link": st.session_state["promotion_link"],
            "previous_campaign_reference": st.session_state["previous_campaign_reference"],
            "sku": st.session_state["sku"],
            "campaign_category": st.session_state["campaign_category"],
            "campaign_type": st.session_state["campaign_type"],
            "campaign_date_range": st.session_state["campaign_date_range"],
            "age_range": st.session_state["age_range"],
            "gender": st.session_state["gender"],
            "income_level": st.session_state["income_level"],
            "region": st.session_state["region"],
            "urban_rural": st.session_state["urban_rural"],
            "specific_instructions": st.session_state["specific_instructions"],
        }
        # Initialize LLM and Prompt Template
        llm = get_llm(api_key, model_name, temperature, top_p)
        prompt = create_prompt_template(instruction="Create marketing campaign content", output_format = output_format)


        with st.spinner("Generating Content..."):
             # Create a LangGraph workflow and execute
             workflow = create_langraph_workflow(llm, prompt, input_vars, output_format)
             result = workflow.invoke(input_vars)
             generated_content = result['output']


        if generated_content:
            st.subheader("Generated Content:")
            st.write(generated_content if isinstance(generated_content, str) else generated_content.dict())
            save_button = st.button("Save Content")
            if save_button:
              save_content_to_file(generated_content, campaign_name)
        else:
            st.error("Failed to generate content. Please check your inputs and try again.")

if __name__ == "__main__":
    main()