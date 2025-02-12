from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from data import SocialMediaContent, EmailContent, MarketingContent  # Assuming these exist
import json
import re
import logging
import time
from langchain.tools import DuckDuckGoSearchRun
from typing import Optional, Dict, Any, Union
from rag import RAGSystem  # Assuming this is your RAG class
from langchain.prompts import PromptTemplate


# Initialize LLM with error handling and type hints
def get_llm(
    api_key: str, model_name: str, temperature: float = 0.7, top_p: float = 0.9
) -> Union[ChatOpenAI, ChatGoogleGenerativeAI, None]:
    try:
        if model_name.startswith("gpt"):
            return ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                top_p=top_p,
            )
    except Exception as e:
        logging.error(f"Error initializing LLM: {str(e)}")
        return None


def generate_content_with_retries(
    llm: Union[ChatOpenAI, ChatGoogleGenerativeAI],
    prompt: PromptTemplate,  # Use PromptTemplate
    input_vars: Dict[str, Any],
    output_format: str,
    use_search_engine: bool = False,
    search_engine_query: Optional[str] = None,
    use_rag: bool = False,
    rag_system: Optional[RAGSystem] = None,
) -> Union[Dict[str, Any], str, None]:  # More specific return type

    max_retries = 3
    retry_count = 0
    parser = None
    search_tool = DuckDuckGoSearchRun()  # Initialize here

    parser_map = {
        "Social Media": PydanticOutputParser(pydantic_object=SocialMediaContent),
        "Email": PydanticOutputParser(pydantic_object=EmailContent),
        "Marketing": PydanticOutputParser(pydantic_object=MarketingContent),
    }
    parser = parser_map.get(output_format)  # Use .get() for safety

    if parser:
        # Add parsing instructions to input_vars
        input_vars["format_instructions"] = parser.get_format_instructions()

    while retry_count < max_retries:
        try:
            # RAG Handling
            if use_rag and rag_system:
                rag_query = input_vars.get("query", input_vars.get("topic", ""))
                if rag_query:
                    rag_response = rag_system.query(rag_query)
                    input_vars["rag_context"] = rag_response.get("answer", "No relevant context found") # Get the rag response
                else:
                    input_vars["rag_context"] = ""
            else:
                input_vars["rag_context"] = ""

            # Search Engine Handling
            if use_search_engine and search_engine_query:
                logging.info(f"Performing web search with query: {search_engine_query}")
                try:
                    search_results = search_tool.run(search_engine_query)
                    logging.info("Search Results:\n" + "-" * 50 + f"\n{search_results}\n" + "-" * 50) # Better formatting
                    input_vars["search_results"] = search_results
                except Exception as e: # Catch web search errors
                    logging.error(f"Web search failed: {e}")
                    input_vars["search_results"] = "Web search failed."

            else:
                logging.info("No web search performed")
                input_vars["search_results"] = "No search terms were provided"

            # Format the prompt using PromptTemplate
            formatted_prompt = prompt.format(**input_vars)
            # formatted_prompt += "\nIMPORTANT: Return ONLY a valid JSON object with no additional text or formatting." # NO, its better to handle this on the parser

            response = llm.invoke(formatted_prompt)
            response_text = response.content

            if parser:
                try:
                    # --- Robust JSON Cleaning and Parsing ---
                    # 1. Remove common whitespace issues and control characters
                    response_text = re.sub(r'[\n\r\t]+', ' ', response_text).strip()
                    response_text = re.sub(r'\s+', ' ', response_text)

                    # 2. Extract JSON-like structure (more resilient)
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if not json_match:
                        raise ValueError("No valid JSON-like object found in response")
                    json_str = json_match.group(0)

                    # 3. Normalize property names (handle variations)
                    json_str = re.sub(r'\s*([\'"])?([a-zA-Z0-9_]+)([\'"])?\s*:', r'"\2":', json_str)

                    # 4. Handle escaped characters and apostrophes robustly
                    json_str = json_str.replace('\\"', '"').replace("\\'", "'")
                    json_str = re.sub(r'"([^"]*)"', lambda m: '"' + m.group(1).replace('"', '\\"').replace("'", "\\'") + '"', json_str)
                    json_str = json_str.replace("'", "\\'")  # Escape any remaining

                    # 5. Ensure consistent spacing
                    json_str = re.sub(r',\s*', ', ', json_str)  # Consistent commas
                    json_str = re.sub(r'}\s*,', '},', json_str)  # Trailing commas
                    json_str = re.sub(r'{\s*', '{', json_str) # Leading space
                    json_str = re.sub(r'\s*}', '}', json_str) # Trailing spaces

                    # 6. Attempt to parse and handle specific errors
                    try:
                        parsed_json = json.loads(json_str)
                    except json.JSONDecodeError as je:
                        logging.error(f"JSON decode error at position {je.pos}: {je.msg}")
                        logging.error(f"Problematic snippet: {json_str[max(0, je.pos - 20):je.pos + 20]}")  # Show context
                        logging.error(f"Full JSON string: {json_str}")
                        raise

                    return parser.parse(json.dumps(parsed_json)) # Parse the correct json format

                except (json.JSONDecodeError, ValueError) as e:
                    logging.error(f"JSON parsing error: {str(e)}")
                    logging.error(f"Raw response: {response_text}")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(1 + retry_count * 0.5)  # Increasing backoff
                        logging.info(f"Retrying... (Attempt {retry_count}/{max_retries})")
                        continue
                    raise  # Re-raise after max retries

            return response_text  # Return raw text if no parser

        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            if retry_count < max_retries - 1:
                retry_count += 1
                time.sleep(1 + retry_count * 0.5)  # Exponential backoff
                logging.info(f"Retrying... (Attempt {retry_count}/{max_retries})")
                continue
            raise

    logging.error("Max retries reached.  Could not generate content.")  # Log failure
    return None