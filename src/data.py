# data.py
from pydantic import BaseModel, Field
from typing import List
# Define multiple content types for different marketing needs
class SocialMediaContent(BaseModel):
    platform: str = Field(
        description="Social media platform (Facebook, Instagram, Twitter)"
    )
    post_text: str = Field(description="Main post content")
    hashtags: List[str] = Field(description="Relevant hashtags")
    call_to_action: str = Field(description="Call to action text")
    key_benefits: List[str] = Field(description="Key benefits of the product")

class EmailContent(BaseModel):
    subject_line: str = Field(description="Email subject line")
    preview_text: str = Field(description="Email preview text")
    body: str = Field(description="Main email body")
    call_to_action: str = Field(description="Call to action button text")
    key_benefits: List[str] = Field(description="Key benefits of the product")

class MarketingContent(BaseModel):
    headline: str = Field(description="The main headline for the marketing content")
    body: str = Field(description="The main body of the marketing message")
    call_to_action: str = Field(description="A clear call to action")
    key_benefits: List[str] = Field(description="Key benefits of the product")


# Define brand options with their descriptions
BRAND_OPTIONS = {
    "Fresh Fri": "A leading cooking oil brand that provides freshness and quality, enhancing every meal",
    "Salit": "A cooking oil brand that offers great taste and quality, trusted by many Kenyan households",
    "Popco": "A cooking oil brand known for its excellent performance and affordable pricing",
    "Diria": "A premium cooking oil brand known for its high quality and versatility in cooking",
    "Fryking": "A premium cooking oil brand designed for professional and home cooking excellence",
    "Mpishi Poa": "A cooking oil brand that offers superior quality at an affordable price, perfect for everyday use",
    "Pwani SBF": "Specially formulated for high-performance frying with longer-lasting oil quality",
    "Onja": "A trusted brand offering quality oils with a focus on taste and performance",
    "Fresco": "A versatile brand known for quality cooking products and personal care items",
    "Criso": "A cooking oil brand that ensures purity and health with every meal",
    "Tiku": "A reliable cooking oil brand that provides purity and exceptional cooking results",
    "Twiga": "A cooking oil brand known for its great value and high quality for everyday cooking needs",
    "Fresh Zait": "A premium cooking oil made from high-quality ingredients, perfect for healthier cooking",
    "Ndume": "A cooking oil brand that combines quality and affordability for everyday use",
    "Detrex": "A personal care brand specializing in hygiene products with a focus on quality",
    "Frymate": "A trusted cooking oil brand ideal for frying, delivering great taste and performance",
    "Sawa": "A trusted personal care brand offering a variety of soaps for hygiene and skincare",
    "Diva": "A premium personal care brand delivering luxury and effectiveness in every product",
    "Ushindi": "A reliable personal care brand known for quality and affordability",
    "Super Chef": "A popular cooking oil brand used by professional chefs for exceptional frying results",
    "White Wash": "A home care brand offering effective cleaning solutions with outstanding performance",
    "Belleza": "A personal care brand offering luxurious skincare products for a refined experience",
    "Afrisense": "A personal care brand providing a wide range of deodorants and fragrances",
    "Diva": "A personal care brand offering beauty and grooming products for a sophisticated lifestyle",
    "Ushindi": "A brand providing quality hygiene products that cater to everyday needs and ensure freshness"
}