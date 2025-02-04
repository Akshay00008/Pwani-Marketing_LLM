import os
import requests
import logging
from datetime import datetime

def generate_product_image(brand_name, description, image_style, openai_api_key, image_model="dall-e-3", image_size="1024x1024", image_quality="standard"):
    """Generates a product image using DALL-E."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)

        prompt = f"Create a {image_style.lower()} product image for {brand_name} brand. Product description: {description}. Focus on visually appealing marketing material. Generate a {image_style.lower()} style image." # More explicit prompt

        response = client.images.generate(
            model=image_model,
            prompt=prompt,
            n=1,
            size=image_size,
            quality=image_quality
        )

        image_url = response.data[0].url
        return image_url

    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return None


def save_generated_image(image_url, brand_name):
    """Saves a generated image from URL to disk."""
    try:
        response = requests.get(image_url, stream=True, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{brand_name.replace(' ', '_')}_image_{timestamp}.png"
        filepath = os.path.join("campaign_outputs", filename)

        os.makedirs("campaign_outputs", exist_ok=True)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Error downloading image: {e}")
        return None
    except Exception as e:
        logging.error(f"Error saving image to file: {e}")
        return None