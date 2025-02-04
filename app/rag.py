from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import logging
import os

class RAGSystem:
    def __init__(self, llm, embedding_model=None, openai_api_key=None):
        self.llm = llm
        self.embedding_model = embedding_model or OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.index_path = "/Users/vishalroy/Downloads/ContentGenApp/faiss_index"

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

    def query_knowledge_base(self, question, k=3):
        """Query the RAG system with a question and return context"""
        if not self.vector_store:
            logging.warning("No documents have been ingested yet")
            return ""  # Return empty string instead of raising an error

        try:
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            if not relevant_docs:
                logging.info("No relevant documents found")
                return ""

            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            return context

        except Exception as e:
            logging.error(f"Error querying RAG system: {str(e)}")
            return ""  # Return empty string on error