import os
import faiss
import gc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

class DataScienceKnowledgeExtractor:
    """
    A class to extract, process, and store knowledge from a PDF document
    for a data science mock interviewer application.
    """
    def __init__(self, pdf_path: str, knowledge_base_dir: str = "knowledge_base"):
        """
        Initializes the extractor.

        Args:
            pdf_path (str): The file path to the PDF document.
            knowledge_base_dir (str): Directory to save the vector store.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        self.pdf_path = pdf_path
        self.knowledge_base_dir = knowledge_base_dir
        self.documents = []
        self.vectorstore = None
        
        print("   -> Initializing embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False,
                'low_cpu_mem_usage': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16  # Smaller batch size for memory efficiency
            }
        )

    def extract_knowledge_from_pdf(self):
        """
        Loads the PDF and splits the text into manageable chunks.
        """
        print("   -> Loading PDF...")
        loader = PyPDFLoader(self.pdf_path)
        raw_documents = loader.load()
        print(f"   -> Loaded {len(raw_documents)} pages from PDF.")

        print("   -> Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " "]
        )
        self.documents = text_splitter.split_documents(raw_documents)
        
        if not self.documents:
            raise ValueError("No documents were extracted from the PDF. Check the PDF content.")
            
        return self.documents

    def create_vector_store(self):
        """Creates a FAISS vector store from the document chunks"""
        if not self.documents:
            raise ValueError("Documents not loaded. Run extract_knowledge_from_pdf() first.")
        
        print("   -> Creating FAISS vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embedding_model
        )
        print("   -> Vector store created.")
        return self.vectorstore

    def save_knowledge_base(self):
        """
        Saves the FAISS vector store to the local directory.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not created. Run create_vector_store() first.")
            
        if not os.path.exists(self.knowledge_base_dir):
            os.makedirs(self.knowledge_base_dir)
            
        print(f"   -> Saving vector store to '{self.knowledge_base_dir}'...")
        self.vectorstore.save_local(self.knowledge_base_dir)
        print("   -> Save complete.")

    @classmethod
    def load_knowledge_base(cls, knowledge_base_dir: str = "knowledge_base"):
        """Loads the FAISS vector store from the local directory"""
        if not os.path.exists(knowledge_base_dir):
            raise FileNotFoundError(f"Knowledge base directory not found at '{knowledge_base_dir}'. Please run the extraction script first.")

        print("   -> Loading knowledge base...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False,
                'low_cpu_mem_usage': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16
            }
        )
        
        vectorstore = FAISS.load_local(knowledge_base_dir, embedding_model, allow_dangerous_deserialization=True)
        gc.collect()
        print("   -> Knowledge base loaded successfully.")
        return vectorstore

