# image_generator.py
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import datetime
import os
import logging

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