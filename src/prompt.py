# prompts.py
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from data import SocialMediaContent, EmailContent, MarketingContent
from rag import RAGSystem
from llm import get_llm
import os

# Initialize RAG system with a default LLM
default_llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0)
rag_context = RAGSystem(default_llm)

def create_prompt_template(
    instruction,
    output_format,
    rag_context_str=None,
    use_search_engine=False,
    search_engine_prompt_template=None
):
    # Rest of the function remains the same
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
 Product Category: {product_category}
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
 Output Format: {output_format}
 Tone and Style: {tone_style}
 """

 # Combine template sections (without context initially)
 full_template = (
     base_template + campaign_details + market_details + content_style_section + instructions_section
 )
 
 # Add RAG context if provided
 if rag_context_str:
     full_template += """
Relevant Context from Knowledge Base:
{rag_context_str}
"""

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
     
     json_instruction = """
     CRITICAL JSON FORMATTING REQUIREMENTS:
     1. Output must be a valid JSON object
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
             "product_category",
             "promotion_link",
             "previous_campaign_reference",
             "campaign_date_range",
             "age_range",
             "gender",
             "income_level",
             "region",
             "urban_rural",
             "specific_instructions",
             "output_format",
             "tone_style",
             "search_results",
         ],
         partial_variables={
             "format_instructions": parser.get_format_instructions(),
             "instruction": instruction + " Output in strict JSON format. Do not include any introductory text or descriptive labels.",
         },
     )

 # For non-JSON output formats
 return PromptTemplate(
     template=full_template,
     input_variables=[
         "brand",
         "campaign_name",
         "sku",
         "product_category",
         "promotion_link",
         "previous_campaign_reference",
         "campaign_date_range",
         "age_range",
         "gender",
         "income_level",
         "region",
         "urban_rural",
         "specific_instructions",
         "output_format",    # Changed from content_type
         "tone_style",
         "search_results",
     ],
     partial_variables={
         "instruction": instruction,
     },
 )    

# prompts.py
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from data import SocialMediaContent, EmailContent, MarketingContent
from rag import RAGSystem
from llm import get_llm
import os

# Initialize RAG system with a default LLM
default_llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0)
rag_context = RAGSystem(default_llm)

def create_prompt_template(
    instruction,
    output_format,
    rag_context_str=None,
    use_search_engine=False,
    search_engine_prompt_template=None
):
    # Base Template
    base_template = """
    {instruction} for {brand} based on the following details:

    **Campaign Details:**
    - **Campaign Name:** {campaign_name}
    - **Brand:** {brand}
    - **SKU:** {sku}
    - **Product Category:** {product_category}
    """

    # Optional Campaign Details
    campaign_details = """
    **Additional Campaign Information:**
    - **Promotion Reference Link:** {promotion_link}
    - **Previous Campaign Reference:** {previous_campaign_reference}
    - **Campaign Date Range:** {campaign_date_range}
    """

    # Target Market Details
    market_details = """
    **Target Market Details:**
    - **Age Range:** {age_range}
    - **Gender:** {gender}
    - **Income Level:** {income_level}
    - **Region:** {region}
    - **Urban/Rural:** {urban_rural}
    """

    # Handle missing specific instructions
    instructions_section = """
    **Specific Features/Instructions:**
    {specific_instructions}
    "Generate engaging and persuasive content for Pwani Oil's marketing campaign that aligns with the following details. This campaign is not limited to their cooking oil products but encompasses all of their products, including margarine, baking fats, and other related offerings.

Campaign Details:

Focus on promoting Pwani Oil's diverse range of high-quality, sustainable, and innovative products.

Highlight the health benefits, eco-friendly practices, and advanced production techniques across all product lines.

Include a call-to-action encouraging customers to choose Pwani Oil for their cooking, baking, and everyday needs.

Target Market Preferences:

The audience values health-conscious, environmentally responsible, and premium-quality products.

They prefer clear, relatable messaging that emphasizes trust, authenticity, and long-term benefits.

The tone should be warm, informative, and inspiring, resonating with families, chefs, bakers, and health enthusiasts.

Specific Instructions:

Use simple, accessible language that appeals to a wide audience.

Incorporate storytelling elements to create an emotional connection with the brand.

Highlight the versatility of Pwani Oil's products (e.g., cooking oils for frying, margarine for baking, etc.).

Include relevant statistics, testimonials, or examples to build credibility.

Ensure the content is consistent with Pwani Oil's core values of quality, sustainability, and innovation.

Core Values Alignment:

Quality: Emphasize the superior standards and rigorous testing processes behind all Pwani Oil products.

Sustainability: Showcase the brand's commitment to eco-friendly sourcing, production, and packaging across their entire product range.

Innovation: Highlight how Pwani Oil leverages cutting-edge technology to deliver healthier, more efficient, and versatile solutions for cooking, baking, and more.

Deliver the content in a format suitable for social media posts, blog articles, or email newsletters, ensuring it is visually appealing and easy to share. The goal is to strengthen brand loyalty, attract new customers, and reinforce Pwani Oil's position as a leader in the cooking oil, margarine, and baking fats industry"""

    # Content Style and Tone
    content_style_section = """
    **Content Specifications:**
    - **Output Format:** {output_format}
    - **Tone and Style:** {tone_style}
    """

    # Combine template sections
    full_template = (
        base_template + campaign_details + market_details + content_style_section + instructions_section
    )

    # Add RAG context section with improved formatting and usage guidelines
    if rag_context_str:
        full_template += """
    **Knowledge Base Context:**
    {rag_context_str}

    **Guidelines for Using Knowledge Base Context:**
    1. Ensure accuracy in product details and specifications
    2. Maintain brand voice and messaging consistency
    3. Incorporate relevant historical campaign insights
    4. Reference successful marketing approaches
    5. Align content with company values and guidelines
    """

    # Conditionally add search results
    if use_search_engine and search_engine_prompt_template:
        search_results_section = """
        **Web Search Results:**
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

        json_instruction = """
        **CRITICAL JSON FORMATTING REQUIREMENTS:**
        1. Output must be a valid JSON object
        2. All property names must be in double quotes
        3. String values must use double quotes
        4. Apostrophes within text must be escaped (e.g., "Pwani Oil\'s" not "Pwani Oil's")
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
                "product_category",
                "promotion_link",
                "previous_campaign_reference",
                "campaign_date_range",
                "age_range",
                "gender",
                "income_level",
                "region",
                "urban_rural",
                "specific_instructions",
                "output_format",
                "tone_style",
                "search_results",
                "rag_context_str",  # Add rag_context_str as an input variable
            ],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "instruction": instruction + " Output in strict JSON format. Do not include any introductory text or descriptive labels.",
            },
        )

    # For non-JSON output formats
    return PromptTemplate(
        template=full_template,
        input_variables=[
            "brand",
            "campaign_name",
            "sku",
            "product_category",
            "promotion_link",
            "previous_campaign_reference",
            "campaign_date_range",
            "age_range",
            "gender",
            "income_level",
            "region",
            "urban_rural",
            "specific_instructions",
            "output_format",
            "tone_style",
            "search_results",
            "rag_context_str",  # Add rag_context_str as an input variable
        ],
        partial_variables={
            "instruction": instruction,
        },
    )
