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

    # Add RAG context section with improved formatting
    rag_context_section = """

Knowledge Base Context:
{rag_context}

Please utilize this knowledge base context to:
1. Ensure accuracy in product details and specifications
2. Maintain brand voice and messaging consistency
3. Incorporate relevant historical campaign insights
4. Reference successful marketing approaches
5. Align content with company values and guidelines
"""
    full_template += rag_context_section

    # Enhanced search results integration with clear usage instructions
    if use_search_engine and search_engine_prompt_template:
        search_results_section = """
 Market Research and Competitive Analysis:
 **Web Search Results:**
 {search_results}

 **Instructions for using Web Search Results:**
 1. **Identify Current Market Trends:** Analyze the search results to understand the latest trends, discussions, and topics related to {product_category} and {brand}.
 2. **Competitive Analysis:** Look for information about competitors, their campaigns, and pricing strategies.
 3. **Incorporate Relevant Keywords & Phrases:** Use insights from search results to include relevant keywords and phrases that resonate with the target market and improve content relevance.
 4. **Verify Factual Claims:** Use search results to verify any factual claims or statistics you include in the generated content to ensure accuracy.
 5. **Identify Content Gaps/Opportunities:**  Look for unanswered questions or gaps in online content that your marketing content can address.

 Please integrate these insights into the generated {output_format} content to make it timely, relevant, and competitive in the current market landscape.
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
                "rag_context",  # Add rag_context as an input variable
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