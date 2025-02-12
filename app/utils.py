# utils.py
import datetime
import re
import json
import streamlit as st
from typing import Optional
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

def load_campaign_template(template_type: str) -> dict:
    """Loads predefined campaign templates"""
    templates = {
        "Product Launch": {
            "product_category": "Cooking Oil",  # Changed from campaign_category
            "specific_instructions": "Create a product launch campaign highlighting unique features, benefits, and introductory offer. Include social media posts, email announcement, and main marketing message.",
        },
        "Seasonal Sale": {
            "product_category": "Personal Care",  # Changed from campaign_category
            "specific_instructions": "Create a seasonal sale campaign with time-limited offers, urgency messaging, and clear price benefits. Focus on social media engagement and shareable content.",
        },
        "Brand Awareness": {
            "product_category": "Home Care",  # Changed from campaign_category
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

def generate_product_image(brand: str, description: str, style: str, api_key: str) -> Optional[str]:
    """Generates a product image using OpenAI's DALL-E model.
    
    Args:
        brand: The brand name for the product
        description: Description or context for image generation
        style: The desired style of the image (e.g., 'modern', 'classic')
        api_key: OpenAI API key
        
    Returns:
        URL of the generated image or None if generation fails
    """
    try:
        import openai
        openai.api_key = api_key
        
        # Create a concise prompt for the image
        prompt = f"Professional product photography of {brand} brand. {description}. Style: {style}, clean background, high-quality commercial product shot."
        
        # Generate image using DALL-E
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        
        return response.data[0].url
    except Exception as e:
        st.error(f"Failed to generate image: {str(e)}")
        return None

def save_generated_image(image_url: str, brand_name: str) -> Optional[str]:
    """Downloads and saves a generated image locally.
    
    Args:
        image_url: URL of the generated image
        brand_name: Brand name for the file naming
        
    Returns:
        Path to the saved image file or None if saving fails
    """
    try:
        import requests
        from pathlib import Path
        
        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{brand_name.replace(' ', '_')}_{timestamp}.png"
        
        # Download and save the image
        response = requests.get(image_url)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
            
        return filename
    except Exception as e:
        st.error(f"Failed to save image: {str(e)}")
        return None