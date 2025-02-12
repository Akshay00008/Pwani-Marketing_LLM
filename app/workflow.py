# workflow.py
from langgraph.graph import StateGraph, END
from llm import generate_content_with_retries
import logging
from rag import RAGSystem

def create_langraph_workflow(llm, prompt, input_vars, output_format, use_search_engine=False, search_engine_query=None):
    # Initialize RAG system once at workflow creation
    use_rag = input_vars.get('use_rag', True)
    logging.info(f"Initializing workflow with RAG enabled: {use_rag}")
    rag_system = RAGSystem(llm) if use_rag else None

    def generate_content(state):
        try:
            # Cache key for RAG context
            cache_key = None
            rag_context_str = None
            if rag_system:
                try:
                    # Create a cache key based on input parameters
                    cache_key = f"{state.get('brand', 'N/A')}_{state.get('sku', 'N/A')}_{state.get('product_category', 'N/A')}"

                    # Check if we have cached results
                    if hasattr(rag_system, '_context_cache') and cache_key in rag_system._context_cache:
                        rag_context_str = rag_system._context_cache[cache_key]
                        logging.info(f"Using cached RAG context for key: {cache_key}")
                    else:
                        # If not in cache, perform the query
                        context_query = f"Brand: {state.get('brand', 'N/A')}\nProduct: {state.get('sku', 'N/A')}\nCategory: {state.get('product_category', 'N/A')}"
                        logging.info(f"Querying RAG system with context:\n{context_query}")
                        rag_context_str = rag_system.query(context_query)

                        # Initialize cache if doesn't exist
                        if not hasattr(rag_system, '_context_cache'):
                            rag_system._context_cache = {}

                        # Cache the results
                        rag_system._context_cache[cache_key] = rag_context_str
                        logging.info(f"Cached RAG context for key: {cache_key}")

                    logging.info(f"RAG context retrieved:\n{rag_context_str}")
                except Exception as rag_error:
                    logging.warning(f"RAG query failed: {str(rag_error)}")

            # Handle web search if enabled with automatic query generation
            web_search_results = None
            if use_search_engine:
                try:
                    # Construct search query from campaign details
                    search_query_parts = [
                        f"{state.get('brand', '')} {state.get('product_category', '')}",
                        f"market price {state.get('sku', '')}",
                        f"trends {state.get('region', '')}" if state.get('region') else "",
                        f"target audience {state.get('age_range', '')} {state.get('gender', '')}" if state.get('age_range') or state.get('gender') else ""
                    ]
                    auto_search_query = " ".join(filter(None, search_query_parts))
                    
                    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
                    search = DuckDuckGoSearchAPIWrapper()
                    web_search_results = search.run(auto_search_query)
                    logging.info(f"Web search results retrieved from DuckDuckGo for query: {auto_search_query}\n{web_search_results}")
                    state['search_results'] = web_search_results # Pass to state as 'search_results'
                except Exception as search_error:
                    logging.warning(f"Web search failed: {str(search_error)}")
                    state['search_results'] = "" # Ensure it's passed even if empty
            else:
                state['search_results'] = "" # Ensure it's passed even if not used

            # Pass RAG context directly to state for prompt template
            if rag_context_str:
                state['rag_context'] = rag_context_str
                logging.info(f"RAG context added to state:\n{rag_context_str}")
            else:
                state['rag_context'] = ""
                logging.info("No RAG context available")

            # Ensure search results are available in state
            if not state.get('search_results'):
                state['search_results'] = ""
                logging.info("No search results available")

            logging.info(f"Generating content with state variables:\n{state}")
            output = generate_content_with_retries(
                llm=llm,
                prompt=prompt,
                input_vars=state,
                output_format=output_format,
                use_search_engine=use_search_engine,
                search_engine_query=search_engine_query,
                use_rag=bool(rag_system),
                rag_system=rag_system
            )
            logging.info(f"Generated content:\n{output}")
            return {"output": output}
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Workflow error: {error_msg}")
            if "rate limit" in error_msg.lower():
                return {"error": "Rate limit exceeded. Please try again later."}
            elif "timeout" in error_msg.lower():
                return {"error": "Request timed out. Please try again."}
            else:
                return {"error": f"Content generation failed: {error_msg}"}

    workflow = StateGraph(dict)
    workflow.add_node("generate_content", generate_content)
    workflow.set_entry_point("generate_content")
    workflow.add_edge("generate_content", END)

    return workflow.compile()