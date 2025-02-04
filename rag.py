import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.memory import ConversationBufferMemory  # Fixed import
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Define environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL_ID = "gpt-3.5-turbo-16k" # Change to your desired model
LOCAL_DB_PATH = "./faiss_db" # Directory where we store vector db
SOURCE_PDF = "/Users/vishalroy/Downloads/ContentGenApp/qwen_2.5.pdf" # Enter the name of your PDF source


def get_llm(api_key, model_name):
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=2048,
    )
    return llm

# Remove @tool decorator since it's not defined or needed for this functio
def rag_system(llm, rag_system):
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=rag_system.vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=rag_system.memory,
    )
    return rag_chain


# Step 1: Data Loading
def load_and_split_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)
    return docs

# Step 2 and 3: Embed and Index Documents
def embed_and_index_documents(docs, local_db_path):
    embeddings = OpenAIEmbeddings(disallowed_special=())  # Add parameter to suppress output
    vector_db = FAISS.from_documents(documents=docs, embedding=embeddings)
    vector_db.save_local(local_db_path)
    return vector_db

def load_vector_db(local_db_path):
    embeddings = OpenAIEmbeddings(disallowed_special=())  # Add parameter to suppress output
    vector_db = FAISS.load_local(local_db_path, embeddings)
    return vector_db

def main():
    docs = load_and_split_documents(SOURCE_PDF)
    vector_db = embed_and_index_documents(docs, LOCAL_DB_PATH)
    
    embeddings = OpenAIEmbeddings(disallowed_special=())  # Add parameter to suppress output
    # Create BM25 retriever for keyword-based search
    bm25_retriever = BM25Retriever.from_documents(docs)
    
    # Create vector retriever for semantic search
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Combine both retrievers into a hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    re_ranker = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    compression_retriever = ContextualCompressionRetriever(base_compressor=re_ranker, base_retriever=hybrid_retriever)
    
    # Update memory configuration with output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify which output to store in memory
    )
    
    llm = ChatOpenAI(model_name=OPENAI_MODEL_ID, temperature=0.1, max_tokens = 2048)
    
    # Use the create_rag_chain function instead of direct creation
    rag_chain = ConversationalRetrievalChain.from_llm(
        retriever=compression_retriever,
        llm=llm,
        memory=memory
    )
    
    while True:
        query = input("Enter your question (or 'exit' to stop): ")
        if query.lower() == "exit":
            break
        # Use invoke instead of direct call
        response = rag_chain.invoke({"question": query})
        print("\nResponse:", response["answer"])
        # Only print source documents if they exist
        if "source_documents" in response:
            print("\nSource Documents:", response["source_documents"])
    

if __name__ == "__main__":
    main()