from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from data import SocialMediaContent, EmailContent, MarketingContent
import json
import re
import logging
import time
from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
from typing import Optional
from rag import RAGSystem

# Initialize LLM with error handling
def get_llm(
    api_key: str, model_name: str, temperature: float = 0.7, top_p: float = 0.9
):
    try:
        if model_name.startswith("gpt"):
            return ChatOpenAI(
                api_key=api_key,  # Add API key here
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

def generate_content_with_retries(llm, prompt, input_vars, output_format, use_search_engine=False, search_engine_query=None, use_rag=False, rag_system: Optional[RAGSystem] = None):
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
            # Simplified RAG handling
            if use_rag and rag_system:
                rag_query = input_vars.get("rag_query", input_vars.get("query", input_vars.get("topic", "")) ) # Correctly get rag_query from input_vars
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
                        logging.error(f"JSON decode error position {je.pos}: {je.msg}")
                        logging.error(
                            f"Character at position: {json_str[je.pos-5:je.pos+5]}"
                        )
                        logging.error(f"Full JSON string: {json_str}")
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