from typing import Dict, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define SKU mappings for each brand
BRAND_SKU_MAPPING: Dict[str, List[str]] = {
    "Fresh Fri": ["500L", "1L", "2L", "5L", "10L", "20L"],
    "Salit": ["500L", "1L", "2L", "5L", "10L"],
    "Popco": ["500L", "1L", "2L", "5L"],
    "Diria": ["1L", "2L", "5L", "10L"],
    "Fryking": ["5L", "10L", "20L"],
    "Mpishi Poa": ["1L", "2L", "5L"],
    "Pwani SBF": ["10L", "20L"],
    "Onja": ["500L", "1L", "2L"],
    "Fresco": ["100G", "175G", "250G"],
    "Criso": ["1L", "2L", "5L"],
    "Tiku": ["500L", "1L", "2L", "5L"],
    "Twiga": ["1L", "2L", "5L"],
    "Fresh Zait": ["500L", "1L", "2L", "5L"],
    "Ndume": ["1L", "2L", "5L"],
    "Detrex": ["80G", "125G", "175G", "250G"],
    "Frymate": ["1L", "2L", "5L"],
    "Sawa": ["125G", "250G", "80G"],
    "Diva": ["100G", "175G", "250G"],
    "Ushindi": ["125G", "175G", "250G"],
    "Super Chef": ["5L", "10L", "20L"],
    "White Wash": ["500G", "1KG", "2KG"],
    "Belleza": ["100G", "175G", "250G"],
    "Afrisense": ["100G", "175G", "250G"]
}

def validate_brand_name(brand_name: str) -> bool:
    """Validates if the brand name exists in the mapping."""
    return brand_name in BRAND_SKU_MAPPING

def get_brand_skus(brand_name: str) -> List[str]:
    """Returns the list of SKUs for a given brand with validation and logging.
    
    Args:
        brand_name: The name of the brand to get SKUs for.
        
    Returns:
        A list of SKUs associated with the brand.
        Returns an empty list if the brand is not found.
    """
    if not brand_name:
        logger.error("Empty brand name provided")
        return []
        
    if not validate_brand_name(brand_name):
        logger.warning(f"Invalid brand name requested: {brand_name}")
        return []
        
    skus = BRAND_SKU_MAPPING.get(brand_name, [])
    logger.info(f"Retrieved {len(skus)} SKUs for brand: {brand_name}")
    return skus

def get_all_brands() -> List[str]:
    """Returns a sorted list of all available brand names.
    
    Returns:
        A sorted list of all brand names in the mapping.
    """
    brands = sorted(BRAND_SKU_MAPPING.keys())
    logger.info(f"Retrieved {len(brands)} total brands")
    return brands