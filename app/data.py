from pydantic import BaseModel, Field, field_validator
from typing import List, Dict

_VALID_PLATFORMS = [
    "Facebook",
    "Instagram",
    "Twitter",
    "LinkedIn",
    "TikTok",
    "Pinterest",
    "YouTube",
    "Snapchat",
]

class SocialMediaContent(BaseModel):
    platform: str = Field(
        description="Social media platform (e.g., Facebook, Instagram, Twitter, LinkedIn, TikTok)"
    )
    post_text: str = Field(description="Main post content, optimized for the platform")
    hashtags: List[str] = Field(
        description="Relevant hashtags to increase visibility", default_factory=list
    )
    call_to_action: str = Field(
        description="Call to action text (e.g., 'Learn more', 'Shop now', 'Visit our website')"
    )
    key_benefits: List[str] = Field(
        description="Key benefits of the product or service being promoted"
    )

    @field_validator("platform")
    def platform_check(cls, value):
        if value.lower() not in [p.lower() for p in _VALID_PLATFORMS]:
            raise ValueError(f"Invalid platform. Choose from: {', '.join(_VALID_PLATFORMS)}")
        return value

class EmailContent(BaseModel):
    subject_line: str = Field(description="Compelling email subject line")
    preview_text: str = Field(
        description="Short preview text displayed in the inbox (supports the subject line)"
    )
    body: str = Field(description="Main email body content")
    call_to_action: str = Field(description="Call to action button or link text")
    key_benefits: List[str] = Field(
        description="Key benefits of the product or service being offered"
    )

class MarketingContent(BaseModel):
    headline: str = Field(description="Main headline for the marketing content (attention-grabbing)")
    body: str = Field(description="Main body of the marketing message (persuasive and informative)")
    call_to_action: str = Field(
        description="Clear and compelling call to action (e.g., 'Get a free quote', 'Download now')"
    )
    key_benefits: List[str] = Field(
        description="Key benefits of the product or service"
    )

class Brand(BaseModel):
    name: str = Field(description="Brand name")
    description: str = Field(description="Brief description of the brand and its values")
    category: str = Field(description="Category of products the brand offers (e.g., 'Cooking Oil', 'Personal Care', 'Home Care')")

BRAND_OPTIONS: Dict[str, Brand] = {
    "Fresh Fri": Brand(name="Fresh Fri", description="A leading cooking oil brand that provides freshness and quality, enhancing every meal", category="Cooking Oil"),
    "Salit": Brand(name="Salit", description="A cooking oil brand that offers great taste and quality, trusted by many Kenyan households", category="Cooking Oil"),
    "Popco": Brand(name="Popco", description="A cooking oil brand known for its excellent performance and affordable pricing", category="Cooking Oil"),
    "Diria": Brand(name="Diria", description="A premium cooking oil brand known for its high quality and versatility in cooking", category="Cooking Oil"),
    "Fryking": Brand(name="Fryking", description="A premium cooking oil brand designed for professional and home cooking excellence", category="Cooking Oil"),
    "Mpishi Poa": Brand(name="Mpishi Poa", description="A cooking oil brand that offers superior quality at an affordable price, perfect for everyday use", category="Cooking Oil"),
    "Pwani SBF": Brand(name="Pwani SBF", description="Specially formulated for high-performance frying with longer-lasting oil quality", category="Cooking Oil"),
    "Onja": Brand(name="Onja", description="A trusted brand offering quality oils with a focus on taste and performance", category="Cooking Oil"),
    "Fresco": Brand(name="Fresco", description="A versatile brand known for quality cooking products and personal care items", category="Cooking Oil, Personal Care"),
    "Criso": Brand(name="Criso", description="A cooking oil brand that ensures purity and health with every meal", category="Cooking Oil"),
    "Tiku": Brand(name="Tiku", description="A reliable cooking oil brand that provides purity and exceptional cooking results", category="Cooking Oil"),
    "Twiga": Brand(name="Twiga", description="A cooking oil brand known for its great value and high quality for everyday cooking needs", category="Cooking Oil"),
    "Fresh Zait": Brand(name="Fresh Zait", description="A premium cooking oil made from high-quality ingredients, perfect for healthier cooking", category="Cooking Oil"),
    "Ndume": Brand(name="Ndume", description="A cooking oil brand that combines quality and affordability for everyday use", category="Cooking Oil"),
    "Detrex": Brand(name="Detrex", description="A personal care brand specializing in hygiene products with a focus on quality", category="Personal Care"),
    "Frymate": Brand(name="Frymate", description="A trusted cooking oil brand ideal for frying, delivering great taste and performance", category="Cooking Oil"),
    "Sawa": Brand(name="Sawa", description="A trusted personal care brand offering a variety of soaps for hygiene and skincare", category="Personal Care"),
    "Diva": Brand(name="Diva", description="A premium personal care brand delivering luxury and effectiveness in every product", category="Personal Care"),
    "Ushindi": Brand(name="Ushindi", description="A reliable personal care brand known for quality and affordability", category="Personal Care"),
    "Super Chef": Brand(name="Super Chef", description="A popular cooking oil brand used by professional chefs for exceptional frying results", category="Cooking Oil"),
    "White Wash": Brand(name="White Wash", description="A home care brand offering effective cleaning solutions with outstanding performance", category="Home Care"),
    "Belleza": Brand(name="Belleza", description="A personal care brand offering luxurious skincare products for a refined experience", category="Personal Care"),
    "Afrisense": Brand(name="Afrisense", description="A personal care brand providing a wide range of deodorants and fragrances", category="Personal Care"),
    "Diva": Brand(name="Diva", description="A personal care brand offering beauty and grooming products for a sophisticated lifestyle", category="Personal Care"),
    "Ushindi": Brand(name="Ushindi", description="A brand providing quality hygiene products that cater to everyday needs and ensure freshness", category="Personal Care, Home Care")
}

def get_brand_description(brand_name: str) -> str:
    """Returns the description of a brand, or a default message if not found."""
    brand = BRAND_OPTIONS.get(brand_name)
    return brand.description if brand else f"Description not found for brand: {brand_name}"

def get_all_brands() -> List[str]:
    """Returns a list of all available brand names."""
    return list(BRAND_OPTIONS.keys())

def get_brands_by_category(category: str) -> List[Brand]:
    """Returns a list of Brand objects filtered by category."""
    return [brand for brand in BRAND_OPTIONS.values() if category.lower() in brand.category.lower()]