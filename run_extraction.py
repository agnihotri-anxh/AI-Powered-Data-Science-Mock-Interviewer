import os
import sys
from Data_Ingestion import DataScienceKnowledgeExtractor

def main():
    """
    Main function to run the knowledge extraction and build the vector store.
    This script only needs to be run once to set up the knowledge base.
    """
    print("=" * 60)
    print("AI-POWERED DATA SCIENCE MOCK INTERVIEWER")
    print("Knowledge Base Setup")
    print("=" * 60)

    # --- Configuration ---
    # IMPORTANT: Make sure this path is correct.
    pdf_path = "The Hundred-Page Machine Learning Book.pdf"
    knowledge_base_path = "knowledge_base"
    
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"\n[ERROR] PDF file not found at '{pdf_path}'")
        print("Please ensure the PDF file is in the same directory as this script.")
        return False

    # Check if the knowledge base already exists
    if os.path.exists(knowledge_base_path):
        print(f"\nKnowledge base already exists at '{knowledge_base_path}'.")
        user_input = input("Do you want to overwrite it? (y/n): ").lower()
        if user_input != 'y':
            print("\nSetup aborted by user.")
            print("You can now run the web application with 'python app.py'")
            return True

    try:
        # Initialize the extractor
        print("\n1. Initializing knowledge extractor...")
        extractor = DataScienceKnowledgeExtractor(pdf_path)
        
        # Run the full extraction and saving process
        print("\n2. Extracting knowledge from PDF...")
        extractor.extract_knowledge_from_pdf()
        
        print("\n3. Creating vector store...")
        extractor.create_vector_store()
        
        print("\n4. Saving knowledge base...")
        extractor.save_knowledge_base()
        
        # Final success message
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the web application: python app.py")
        print("2. Access the interviewer at http://127.0.0.1:5000")
        
        return True

    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred during the process: {str(e)}")
        print("\nTroubleshooting:")
        print("  - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  - Check that the PDF file is not corrupted.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
