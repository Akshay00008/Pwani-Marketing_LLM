from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
import os
import warnings
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import GPU FAISS (more robust handling)
try:
    import faiss

    GPU_FAISS_AVAILABLE = hasattr(faiss, 'GpuIndexFlatL2')
    if GPU_FAISS_AVAILABLE:
        logger.info("GPU FAISS is available")
    else:
        logger.info("GPU FAISS is not available, using CPU version")
except ImportError:
    GPU_FAISS_AVAILABLE = False
    logger.warning("FAISS import failed, will use CPU-only version")
    warnings.warn("FAISS import failed, will use CPU-only version", ImportWarning)


class RAGSystemConfig(BaseModel):
    """Configuration for the RAG system."""
    chunk_size: int = Field(500, description="Size of text chunks")
    chunk_overlap: int = Field(50, description="Overlap between text chunks")
    similarity_threshold: float = Field(0.7, description="Similarity threshold for embeddings filter")
    bm25_weight: float = Field(0.3, description="Weight for BM25 retriever")
    vector_weight: float = Field(0.7, description="Weight for vector retriever")
    index_path: str = Field("ContentGenApp/faiss_index", description="Path to save/load FAISS index")
    knowledge_base_path: str = Field("cleaned_cleaned_output.txt", description="path to knowledge base")
    use_summary_memory: bool = Field(False, description="Whether to use ConversationSummaryBufferMemory instead of ConversationBufferMemory")
    max_token_limit: int = Field(3000, description="Max token limit for summary memory (if used)")  # Add a max_token_limit


class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) System.

    This class provides methods for ingesting documents, initializing a knowledge base,
    and querying the system to retrieve relevant information.
    """

    def __init__(self, llm, embedding_model=None, openai_api_key=None, config: Optional[RAGSystemConfig] = None):
        """
        Initializes the RAGSystem.

        Args:
            llm: The language model to use for generation.
            embedding_model: The embedding model to use (defaults to OpenAIEmbeddings).
            openai_api_key: Your OpenAI API key (if using OpenAIEmbeddings).
            config:  RAG System Configuration
        """
        self.llm = llm
        self.embedding_model = embedding_model or OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.config = config or RAGSystemConfig()  # Use default config if none provided
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.vector_store = None
        self.index_path = self.config.index_path
        self.knowledge_base_path = self.config.knowledge_base_path


        # Memory initialization (using config option)
        if self.config.use_summary_memory:
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                max_token_limit=self.config.max_token_limit
            )
        else:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        # Load index (with GPU support) - separate method for clarity
        self._load_existing_index()


    def _load_existing_index(self):
        """Loads an existing FAISS index from disk (if it exists)."""
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True  # ONLY with trusted data
                )
                logger.info(f"Loaded existing FAISS index from {self.index_path}")

                if GPU_FAISS_AVAILABLE:
                    self._move_index_to_gpu()

            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self.vector_store = None  # Ensure vector_store is None on failure


    def _move_index_to_gpu(self):
        """Moves the FAISS index to the GPU (if available)."""
        try:
            res = faiss.StandardGpuResources()
            if hasattr(self.vector_store, 'index'):
                self.vector_store.index = faiss.index_cpu_to_gpu(res, 0, self.vector_store.index)
                logger.info("Successfully moved FAISS index to GPU")
            else:
                logger.warning("FAISS index does not have an 'index' attribute; cannot move to GPU.")
        except Exception as gpu_error:
            logger.warning(f"Failed to move index to GPU: {gpu_error}. Using CPU version.")



    def ingest_documents(self, documents: List[Document]):
        """
        Ingests a list of Langchain Document objects into the RAG system.

        Args:
            documents: A list of Langchain Document objects.
        """
        if not isinstance(documents, list):
            raise TypeError("documents must be a list of Langchain Document objects.")
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All items in documents must be Langchain Document objects.")


        try:
            if self.vector_store is not None:
                logger.info("Adding documents to existing vector store")
                texts = self.text_splitter.split_documents(documents)
                self.vector_store.add_documents(texts) # Add to existing, don't recreate
                self.vector_store.save_local(self.index_path)  # Save updated index
                return True

            texts = self.text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embedding_model
            )
            self.vector_store.save_local(self.index_path)
            logger.info(f"Created new FAISS index and saved to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            return False


    def initialize_knowledge_base(self, content: str):
        """
        Initializes the vector store with content from a string.

        Args:
            content: The text content to initialize the knowledge base with.
        """
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if not content.strip():  # Check for empty or whitespace-only string
            logger.warning("Content is empty.  Knowledge base not initialized.")
            return False

        try:
            texts = self.text_splitter.split_text(content)
            if texts:
                self.vector_store = FAISS.from_texts(
                    texts,
                    self.embedding_model,
                    metadatas=[{"source": f"chunk_{i}"} for i in range(len(texts))]
                )
                self.vector_store.save_local(self.index_path)
                logger.info(f"Initialized knowledge base and saved to {self.index_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            return False

    def _create_retriever(self, documents:List[Document], k: int = 3):
        """Creates and configures the retriever chain."""

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k # set k here

        # Create vector retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[self.config.bm25_weight, self.config.vector_weight]
        )

        # Create contextual compression retriever with EmbeddingsFilter and LLMChainExtractor
        re_ranker = EmbeddingsFilter(embeddings=self.embedding_model, similarity_threshold=self.config.similarity_threshold)
        # Use LLMChainExtractor for more sophisticated compression
        compressor = LLMChainExtractor.from_llm(self.llm)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,  # Use the LLMChainExtractor
            base_retriever=ensemble_retriever
        )
        return compression_retriever


    def query(self, question: str, k: int = 3):
        """
        Queries the RAG system.

        Args:
            question: The question to ask.
            k: The number of documents to retrieve.

        Returns:
            The answer to the question, or an empty string if an error occurs.
        """
        if not isinstance(question, str) or not question.strip():
            logger.warning("Question is empty or invalid.")
            return ""

        if not self.vector_store:
            logger.warning("No documents have been ingested yet.")
            return ""

        try:
            # get documents from vector store. Need raw documents for BM25
            docs = self.vector_store.similarity_search(question, k=k*3)  # Fetch more, compression will reduce
            if not docs:
                logger.info("No documents found for the question.")
                return ""

            retriever = self._create_retriever(docs, k)

            # --- Create the Conversational Chain ---
            #   Using LCEL for better control and transparency
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer questions based on the provided context and chat history. "
                           "If the answer is not in the context or chat history, say 'I don't know'."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Context: {context}\nQuestion: {input}")
            ])

            # Load chat history with proper error handling
            try:
                chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
            except Exception as e:
                logger.warning(f"Error loading chat history: {e}")
                chat_history = []

            # Conversational chain using RunnablePassthrough
            chain = (
                RunnablePassthrough.assign(
                    context=lambda x: "\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["input"])])
                )
                | prompt
                | self.llm
                | StrOutputParser()
            )
            # Invoke the chain with the question and chat history
            response = chain.invoke({"input": question, "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])})

            # Save to memory *after* the LLM call
            self.memory.save_context({"input": question}, {"answer": response})
            return response

        except Exception as e:
            logger.exception(f"Error querying RAG system: {e}")  # More detailed logging
            return ""