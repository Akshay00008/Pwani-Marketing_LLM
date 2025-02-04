from pydantic import BaseModel, Field
from typing import List
# Define multiple content types for different marketing needs
class SocialMediaContent(BaseModel):
    platform: str = Field(
        description="Social media platform (Facebook, Instagram, Twitter)"
    )
    post_text: str = Field(description="Main post content")
    hashtags: List[str] = Field(description="Relevant hashtags")
    call_to_action: str = Field(description="Call to action text")
    key_benefits: List[str] = Field(description="Key benefits of the product")

class EmailContent(BaseModel):
    subject_line: str = Field(description="Email subject line")
    preview_text: str = Field(description="Email preview text")
    body: str = Field(description="Main email body")
    call_to_action: str = Field(description="Call to action button text")
    key_benefits: List[str] = Field(description="Key benefits of the product")

class MarketingContent(BaseModel):
    headline: str = Field(description="The main headline for the marketing content")
    body: str = Field(description="The main body of the marketing message")
    call_to_action: str = Field(description="A clear call to action")
    key_benefits: List[str] = Field(description="Key benefits of the product")


# Define brand options with their descriptions
BRAND_OPTIONS = {
    "Diria": "A premium cooking oil brand known for its high quality and versatility in cooking",
    "Frymate": "A trusted cooking oil brand ideal for frying, delivering great taste and performance",
    "Mpishi Poa": "A cooking oil brand that offers superior quality at an affordable price, perfect for everyday use",
    "Pwani SBF": "Specially formulated for high-performance frying with longer-lasting oil quality",
    "Super Chef": "A popular cooking oil brand used by professional chefs for exceptional frying results",
    "Criso": "A cooking oil brand that ensures purity and health with every meal",
    "Fresh Fri": "A leading cooking oil brand that provides freshness and quality, enhancing every meal",
    "Fresh Zait": "A premium cooking oil made from high-quality ingredients, perfect for healthier cooking",
    "Popco": "A cooking oil brand known for its excellent performance and affordable pricing",
    "Salit": "A cooking oil brand that offers great taste and quality, trusted by many Kenyan households",
    "Tiku": "A reliable cooking oil brand that provides purity and exceptional cooking results",
    "Twiga": "A cooking oil brand known for its great value and high quality for everyday cooking needs",
    "Onja": "A trusted brand offering quality oils with a focus on taste and performance",
    "Ndume": "A cooking oil brand that combines quality and affordability for everyday use",
    "Whitewash": "A home care brand offering effective cleaning solutions with outstanding performance",
    "4U": "A home care product line that delivers excellent results in cleaning with a focus on customer satisfaction",
    "Belleza": "A personal care brand offering luxurious skincare products for a refined experience",
    "Fresco": "A personal care brand known for offering effective and gentle beauty products",
    "Sawa": "A trusted personal care brand offering a variety of soaps for hygiene and skincare",
    "Afrisense": "A personal care brand providing a wide range of deodorants and fragrances",
    "Detrex": "A personal care brand specializing in hygiene products with a focus on quality",
    "Diva": "A personal care brand offering beauty and grooming products for a sophisticated lifestyle",
    "Ushindi": "A brand providing quality hygiene products that cater to everyday needs and ensure freshness"
}

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import logging
import os

class RAGSystem: # RAGSystem class - already updated in previous response
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