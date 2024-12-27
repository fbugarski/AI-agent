from agents.pdf_agent import summarize_pdf
from agents.wikipedia_agent import fetch_wikipedia_summary
from agents.chromadb_agent import ChromadbAgent
from utils.pdf_processing import extract_text_from_pdf, get_embedding, save_embeddings_to_chromadb
import os

def main():
    print("Welcome to the multi-agent system!")
    user_choice = input("What would you like to do? Type 'pdf' to summarize a PDF, 'wiki' to search Wikipedia, or 'embedding' to generate PDF embeddings: ").strip().lower()
    
    # Definišemo putanju do foldera sa PDF-ovima
    pdf_folder_path = "data"  # Putanja do foldera sa PDF-ovima
    
    # Uzimamo sve PDF fajlove iz foldera
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    
    # Inicijalizacija ChromaDB agenta
    chromadb_agent = ChromadbAgent()

    if user_choice == 'pdf':
        print("Available PDF files:")
        for idx, pdf_file in enumerate(pdf_files, start=1):
            print(f"{idx}. {pdf_file}")
        
        try:
            pdf_choice = int(input(f"Select a PDF file (1-{len(pdf_files)}): ").strip())
            selected_pdf = pdf_files[pdf_choice - 1]
            selected_pdf_path = os.path.join(pdf_folder_path, selected_pdf)
            
            print(f"Using the selected PDF file: {selected_pdf_path}")
            pdf_summary = summarize_pdf(selected_pdf_path)
            print("Summary of the document:", pdf_summary)
        except (ValueError, IndexError):
            print("Invalid selection. Please select a valid number.")
    
    elif user_choice == 'wiki':
        topic = input("Enter a topic to search on Wikipedia: ").strip()
        wikipedia_summary = fetch_wikipedia_summary(topic)
        print("Wikipedia summary:", wikipedia_summary)
    
    elif user_choice == 'embedding':
        print("Generating embeddings for available PDFs and storing in ChromaDB...")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder_path, pdf_file)
            print(f"Processing PDF: {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            embedding = get_embedding(text)
            
            # Dodavanje embeddinga u ChromaDB
            metadata = {"title": pdf_file}
            save_embeddings_to_chromadb(embedding, metadata, collection_name=chromadb_agent.collection.name)
            print(f"Embedding for {pdf_file} added to ChromaDB.")
        
        print("Embeddings generation and storage completed.")

    else:
        print("Invalid option. Please type 'pdf', 'wiki', or 'embedding'.")

if __name__ == "__main__":
    main()
