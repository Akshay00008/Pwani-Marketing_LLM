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
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # Import StrOutputParser


class RAGSystem:
    def __init__(self, llm, config: Dict = None):
        self.llm = llm
        self.config = config or {}  # Default to an empty dictionary

        self.embedding_model = self.config.get("embedding_model", OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
        )
        self.index_path = self.config.get("index_path", "/Users/vishalroy/Downloads/ContentGenApp/faiss_index")  # remember to change the path.
        self.vector_store = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  #  use output_key with ConversationalRetrievalChain
        )

        # Try to load existing index
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logging.info("Successfully loaded existing vector store.")
            except Exception as e:
                logging.error(f"Error loading vector store: {str(e)}")
                self.vector_store = None


    def ingest_documents(self, documents: List[Document]) -> bool:
        """Ingests a list of Langchain Document objects (optimized)."""
        try:
            if not all(isinstance(doc, Document) for doc in documents):
                raise TypeError("Data must be a list of Document objects.")

            split_docs = self.text_splitter.split_documents(documents)

            if self.vector_store:
                self.vector_store.add_documents(split_docs)
            else:
                self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)

            self.vector_store.save_local(self.index_path)
            logging.info("Data ingestion successful.")
            return True
        except Exception as e:
            logging.error(f"Ingestion Error: {e}")
            return False

    def ingest(self, data: str) -> bool:
        """Ingest string data (kept for flexibility)."""
        try:
            if not isinstance(data, str):
                raise TypeError("Data must be a string")

            texts = self.text_splitter.split_text(data)
            metadatas = [{"source": f"chunk_{i}"} for i in range(len(texts))]

            if self.vector_store:
              self.vector_store.add_texts(texts, metadatas)
            else:
              self.vector_store = FAISS.from_texts(texts, self.embedding_model, metadatas)

            self.vector_store.save_local(self.index_path)
            logging.info("Document ingestion successful.")
            return True

        except Exception as e:
            logging.error(f"Ingestion failed: {e}")
            return False


    def query(self, question: str, k: int = None, use_web_search: bool = True) -> Dict:
        """Query the RAG system (optimized)."""
        if not self.vector_store:
            logging.warning("No documents have been ingested yet")
            return {"answer": "", "source_documents": [], "web_results": None}

        k = k or self.config.get("retriever_k", 3)
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                bm25_future = executor.submit(self._get_bm25_retriever, k)
                vector_future = executor.submit(self._get_vector_retriever, k)
                web_search_future = executor.submit(self._get_web_search, question) if use_web_search else None

                bm25_retriever = bm25_future.result()
                vector_retriever = vector_future.result()

                if bm25_retriever is None:  # Handle the case where BM25 is empty
                    hybrid_retriever = vector_retriever
                else:
                    ensemble_weights = self.config.get("ensemble_weights", [0.3, 0.7])
                    hybrid_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=ensemble_weights
                    )

                similarity_threshold = self.config.get("similarity_threshold", 0.7)
                re_ranker = EmbeddingsFilter(
                    embeddings=self.embedding_model,
                    similarity_threshold=similarity_threshold
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=re_ranker,
                    base_retriever=hybrid_retriever
                )

                web_results = web_search_future.result() if web_search_future else None

                # --- Optimized Chain Construction (using LCEL) ---
                # 1. Define the prompt template
                prompt = PromptTemplate.from_template(
                    "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
                )

                # 2. Create the chain using LCEL
                rag_chain = (
                    {"context": compression_retriever, "question": lambda x: x["question"]}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )

                # 3. Invoke the chain
                rag_answer = rag_chain.invoke({"question": question})

                # --- Get Source Documents ---
                source_documents = compression_retriever.get_relevant_documents(question)


                if web_results:
                    # --- Optimized Combination with Web Results (using LCEL) ---
                    final_prompt = PromptTemplate.from_template(
                        "Based on the following information, provide a comprehensive answer to the question: {question}\n\nRAG Answer:\n{rag_answer}\n\nWeb Search Results:\n{web_results}"
                    )
                    final_chain = (
                        {"rag_answer": lambda x: x["rag_answer"], "web_results": lambda x: x["web_results"], "question": lambda x: x["question"]}
                        | final_prompt
                        | self.llm
                        | StrOutputParser()
                    )
                    final_response = final_chain.invoke({"rag_answer": rag_answer, "web_results": web_results, "question": question})

                    return {
                        "answer": final_response,
                        "source_documents": source_documents,
                        "web_results": web_results,
                    }

                return {
                    "answer": rag_answer,
                    "source_documents": source_documents,
                    "web_results": None,
                }

        except Exception as e:
            logging.error(f"Error querying RAG system: {str(e)}")
            return {"answer": "", "source_documents": [], "web_results": None}

    def _get_bm25_retriever(self, k: int):
        """Helper: Create BM25 retriever (handles empty docstore)."""
        if not self.vector_store or not self.vector_store.docstore._dict:
            return None
        bm25_retriever = BM25Retriever.from_documents(list(self.vector_store.docstore._dict.values()))
        bm25_retriever.k = k
        return bm25_retriever

    def _get_vector_retriever(self, k: int):
        """Helper: Create vector retriever."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def _get_web_search(self, question: str):
        """Helper: Perform web search."""
        try:
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
            search = DuckDuckGoSearchAPIWrapper()
            return search.run(question)
        except Exception as web_error:
            logging.warning(f"Web search failed: {str(web_error)}")
            return None