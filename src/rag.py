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

class RAGSystem:
    def __init__(self, llm, embedding_model=None, openai_api_key=None):
        self.llm = llm
        self.embedding_model = embedding_model or OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for better context
            chunk_overlap=50  # Reduced overlap to minimize redundancy
        )
        self.vector_store = None
        self.index_path = "/Users/vishalroy/Downloads/ContentGenApp/faiss_index"
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
            except Exception as e:
                logging.error(f"Error loading vector store: {str(e)}")
                self.vector_store = None

    def ingest_documents(self, documents):
        """
        Ingest documents into the RAG system
        """
        try:
            if self.vector_store is not None:
                logging.info("Using existing vector store")
                return True
                
            texts = self.text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embedding_model
            )
            # Save the index
            self.vector_store.save_local(self.index_path)
            return True
        except Exception as e:
            logging.error(f"Error ingesting documents: {str(e)}")
            return False

    def initialize_knowledge_base(self, content):
        """Initialize the vector store with content"""
        texts = self.text_splitter.split_text(content)
        if texts:
            self.vector_store = FAISS.from_texts(
                texts,
                self.embedding_model,
                metadatas=[{"source": f"chunk_{i}"} for i in range(len(texts))]
            )
            # Save the index for future use
            self.vector_store.save_local(self.index_path)
            return True
        return False

    def query(self, question, k=3):
        """Query the RAG system with a question and return context"""
        if not self.vector_store:
            logging.warning("No documents have been ingested yet")
            return ""  # Return empty string instead of raising an error

        try:
            # Create BM25 retriever for keyword-based search
            bm25_retriever = BM25Retriever.from_documents(self.vector_store.docstore._dict.values())
            
            # Create vector retriever for semantic search
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            
            # Combine both retrievers into a hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7]  # Give more weight to semantic search
            )
            
            # Add contextual compression for better results
            re_ranker = EmbeddingsFilter(embeddings=self.embedding_model, similarity_threshold=0.7)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=re_ranker,
                base_retriever=hybrid_retriever
            )
            
            # Create conversational chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=compression_retriever,
                memory=self.memory,
                return_source_documents=True
            )
            
            # Get response
            response = chain({"question": question})
            return response["answer"]
            
        except Exception as e:
            logging.error(f"Error querying RAG system: {str(e)}")
            return ""  # Return empty string on error
