# utils.py
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

from langchain.output_parsers import PydanticOutputParser

from data import EmailContent, MarketingContent, SocialMediaContent
from llm import get_llm, search_tool  # Import necessary llm functions
from prompt import create_prompt_template  # Import necessary prompt function


def validate_inputs(input_data):
    """Validates inputs to ensure required fields are present."""
    required_fields = [
        "campaign_name",
        "brand",
        "product_category",
        "output_format",
        "tone_style",
        "specific_instructions",
        "campaign_date_range"
    ]
    missing_fields = [field for field in required_fields if not input_data.get(field)]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, None


def validate_date_range(date_range_str):
    """Validates that the date range is in correct format and end date is after start date."""
    if not date_range_str:
        return True  # No date range to validate

    try:
        start_date_str, end_date_str = map(str.strip, date_range_str.split("to"))
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        return start_date <= end_date
    except ValueError:
        return False


def save_content_to_file(content, campaign_name, save_format="txt"):
    """Saves generated content to a file, either in txt or json format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{campaign_name.replace(' ', '_')}_{timestamp}.{save_format}"
    filepath = os.path.join("output_content", filename)  # Save to 'output_content' dir
    os.makedirs("output_content", exist_ok=True)  # Ensure directory exists

    try:
        if save_format == "json":
            if isinstance(content, (SocialMediaContent, EmailContent, MarketingContent)):
                content_dict = content.dict() # Convert Pydantic model to dict
            elif isinstance(content, str):
                content_dict = {"text_content": content}
            else:
                content_dict = content # Assume it's already a dict or serializable
            with open(filepath, "w") as f:
                json.dump(content_dict, f, indent=4)
        else:  # Default to txt format
            text_content = ""
            if isinstance(content, (SocialMediaContent, EmailContent, MarketingContent)):
                text_content = content.json(indent=4) # Serialize Pydantic to JSON string for text file
            elif isinstance(content, str):
                text_content = content
            elif isinstance(content, dict):
                 text_content = json.dumps(content, indent=4)
            else:
                text_content = str(content) # Fallback to string conversion

            with open(filepath, "w") as f:
                f.write(text_content)

        return filepath
    except Exception as e:
        logging.error(f"Error saving content to file: {e}")
        return None


def load_campaign_template(template_name):
    """Loads campaign template from a JSON file."""
    template_file = f"templates/{template_name.replace(' ', '_').lower()}.json"
    try:
        with open(template_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Template file not found: {template_file}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from template file: {template_file}")
        return {}


def generate_content_with_retries(llm, prompt, input_vars, output_format, use_search_engine=False, search_engine_query=None, use_rag=False, rag_system=None): # rag_system is now optional and can be passed in
    max_retries = 3
    retry_count = 0
    parser = None

    if output_format in ["Social Media", "Email", "Marketing"]:
        parser_map = {
            "Social Media": PydanticOutputParser(pydantic_object=SocialMediaContent),
            "Email": PydanticOutputParser(pydantic_object=EmailContent),
            "Marketing": PydanticOutputParser(pydantic_object=MarketingContent),
        }
        parser = parser_map[output_format]

    while retry_count < max_retries:
        try:
            # Simplified RAG handling - Use the passed-in rag_system if available
            if use_rag and rag_system:
                rag_query = input_vars.get("rag_query", input_vars.get("query", input_vars.get("topic", ""))) # Correctly get rag_query from input_vars
                if rag_query:
                    input_vars["rag_context"] = rag_system.query_knowledge_base(rag_query) or "No relevant context found" # Use query_knowledge_base
                else:
                    input_vars["rag_context"] = ""
            else:
                input_vars["rag_context"] = ""

            # Existing search engine logic
            if use_search_engine and search_engine_query:
                logging.info(f"Performing web search with query: {search_engine_query}")
                search_results = search_tool.run(search_engine_query)
                logging.info("Search Results:")
                logging.info("-" * 50)
                logging.info(search_results)
                logging.info("-" * 50)
                input_vars["search_results"] = search_results
            else:
                logging.info("No web search performed")
                input_vars["search_results"] = "No search terms were provided"

            formatted_prompt = prompt.format(**input_vars)
            formatted_prompt += "\nIMPORTANT: Return ONLY a valid JSON object with no additional text or formatting."

            response = llm.invoke(formatted_prompt)
            response_text = response.content

            if parser:
                try:
                    # Improved JSON cleaning
                    response_text = re.sub(r'[\n\r\t]', ' ', response_text)
                    response_text = re.sub(r'\s+', ' ', response_text)
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

                    if not json_match:
                        raise ValueError("No valid JSON object found in response")

                    json_str = json_match.group()

                    # Handle escaped characters before parsing
                    json_str = json_str.replace('\\"', '"').replace("\\'", "'")

                    # Handle apostrophes before JSON parsing
                    def escape_apostrophes(match):
                        text = match.group(1)
                        # Escape any apostrophes within the quoted text
                        text = text.replace("'", "\\'")
                        return f'"{text}"'

                    # Replace content within double quotes, handling apostrophes
                    json_str = re.sub(r'"([^"]*)"', escape_apostrophes, json_str)

                    # Normalize property names - Fix the regex pattern
                    json_str = re.sub(
                        r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str
                    )

                    # Remove any remaining unescaped apostrophes
                    json_str = json_str.replace("'", "\\'")

                    # Clean up any double-escaped quotes
                    json_str = json_str.replace('\\"', '"')

                    # Ensure proper spacing
                    json_str = re.sub(r",\s*([^\s])", r", \1", json_str)
                    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

                    try:
                        parsed_json = json.loads(json_str)
                    except json.JSONDecodeError as je:
                        # Add detailed logging for debugging
                        print(f"JSON decode error position {je.pos}: {je.msg}")
                        print(
                            f"Character at position: {json_str[je.pos-5:je.pos+5]}"
                        )
                        print(f"Full JSON string: {json_str}")
                        raise

                    return parser.parse(json.dumps(parsed_json))

                except (json.JSONDecodeError, ValueError) as e:
                    logging.error(f"JSON parsing error: {str(e)}")
                    logging.error(f"Raw response: {response_text}")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(1)
                        continue
                    raise

            return response_text

        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            if retry_count < max_retries - 1:
                retry_count += 1
                time.sleep(1)
                continue
            raise

    return None