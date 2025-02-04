# agent_state.py
from typing import Optional, Tuple, Union, List
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from data import SocialMediaContent, EmailContent, MarketingContent  # Ensure this import is present


# Define the Agent State -  Crucial for LangGraph
class AgentState(BaseModel):
    """Agent State - Crucial for LangGraph"""
    class Config:
        arbitrary_types_allowed = True  # Add this Config class
    campaign_name: Optional[str] = Field(None)
    promotion_link: Optional[str] = Field(None)
    previous_campaign_reference: Optional[str] = Field(None)
    sku: Optional[str] = Field(None)
    product_category: Optional[str] = Field(None)
    campaign_date_range: Optional[str] = Field(None)
    age_range: Optional[str] = Field(None)
    gender: Optional[str] = Field(None)
    income_level: Optional[str] = Field(None)
    region: Optional[str] = Field(None)
    urban_rural: Optional[str] = Field(None)
    specific_instructions: Optional[str] = Field(None)
    brand: Optional[str] = Field(None)
    tone_style: Optional[str] = Field(None)
    output_format: Optional[str] = Field(None)
    rag_query: Optional[str] = Field(None)
    use_rag: bool = Field(False)
    use_search_engine: bool = Field(False)
    search_engine_query: Optional[str] = Field(None)
    rag_context: Optional[str] = Field(None)
    search_results: Optional[str] = Field(None)
    generated_content: Optional[Union[str, 'SocialMediaContent', 'EmailContent', 'MarketingContent', dict]] = Field(None) # Forward declarations to avoid circular imports (if needed later)
    image_url: Optional[str] = Field(None)
    image_style: Optional[str] = Field(None)
    generate_image_checkbox: bool = Field(False)
    model_name_select: Optional[str] = Field(None)
    temperature_slider: Optional[float] = Field(None)
    top_p_slider: Optional[float] = Field(None)
    intermediate_steps: List[Tuple] = Field(default_factory=list)
    messages: List[BaseMessage] = Field(default_factory=list)