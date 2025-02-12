from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory
import logging
import os
from typing import List, Dict, Union
from langchain.docstore.document import Document

class RAGSystem:
    def __init__(self, llm, config: Dict = None):
        self.llm = llm
        self.config = config or {}  # Default to an empty dictionary

        self.embedding_model = self.config.get("embedding_model", OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
        )
        self.index_path = self.config.get("index_path", "/Users/vishalroy/Downloads/ContentGenApp/faiss_index")
        self.vector_store = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Try to load existing index with safe deserialization
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True  # Only use with trusted data
                )
                logging.info("Successfully loaded existing vector store.")
            except Exception as e:
                logging.error(f"Error loading vector store: {str(e)}")
                self.vector_store = None


    def ingest(self, data: Union[str, List[Document]]) -> bool:
        """
        Ingests either a single string or a list of Langchain Document objects.
        """
        try:
            if isinstance(data, str):
                texts = self.text_splitter.split_text(data)
                metadatas = [{"source": f"chunk_{i}"} for i in range(len(texts))]
                if self.vector_store:
                    self.vector_store.add_texts(texts, metadatas)
                else:
                    self.vector_store = FAISS.from_texts(texts, self.embedding_model, metadatas)
            elif isinstance(data, list) and all(isinstance(doc, Document) for doc in data):
                texts = self.text_splitter.split_documents(data)
                if self.vector_store:
                    self.vector_store.add_documents(texts)
                else:
                  self.vector_store = FAISS.from_documents(texts, self.embedding_model)
            else:
                raise TypeError("Data must be a string or a list of Document objects.")

            self.vector_store.save_local(self.index_path)
            logging.info("Data ingestion successful.")
            return True
        except Exception as e:
          logging.error(f"Ingestion Error: {e}")
          return False

    def query(self, question: str, k: int = None, use_web_search: bool = True) -> Dict:
        """Query the RAG system with a question and return context, optionally including web search results"""
        if not self.vector_store:
            logging.warning("No documents have been ingested yet")
            return {"answer": "", "source_documents": [], "web_results": None}

        k = k or self.config.get("retriever_k", 3)
        try:
            # Create BM25 retriever for keyword-based search
            bm25_retriever = BM25Retriever.from_documents(self.vector_store.docstore._dict.values())
            bm25_retriever.k = k

            # Create vector retriever for semantic search
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

            # Combine both retrievers into a hybrid retriever
            ensemble_weights = self.config.get("ensemble_weights", [0.3, 0.7])
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=ensemble_weights
            )

            # Add contextual compression for better results
            similarity_threshold = self.config.get("similarity_threshold", 0.7)
            re_ranker = EmbeddingsFilter(embeddings=self.embedding_model, similarity_threshold=similarity_threshold)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=re_ranker,
                base_retriever=hybrid_retriever
            )

            # Initialize web search if enabled
            web_results = None
            if use_web_search:
                try:
                    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
                    search = DuckDuckGoSearchAPIWrapper()
                    web_results = search.run(question) # returns a string
                except Exception as web_error:
                    logging.warning(f"Web search failed: {str(web_error)}")

            # Create conversational chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=compression_retriever,
                memory=self.memory,
                return_source_documents=True
            )

            # Get response from RAG system
            response = chain({"question": question})
            rag_answer = response["answer"]

            # Combine RAG and web search results
            if web_results:
                combined_context = f"{rag_answer}\nWeb Search Results:\n{web_results}"
                # Generate final answer incorporating both sources
                final_response = self.llm.predict(
                    f"Based on the following information, provide a comprehensive answer to the question: {question}\n\n{combined_context}"
                )
                return {
                    "answer": final_response,
                    "source_documents": response.get("source_documents", []),
                    "web_results": web_results,
                }

            return {
                "answer": rag_answer,
                "source_documents": response.get("source_documents", []),
                "web_results": None,
            }

        except Exception as e:
            logging.error(f"Error querying RAG system: {str(e)}")
            return {"answer": "", "source_documents": [], "web_results": None}