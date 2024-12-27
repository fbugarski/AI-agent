from utils.pdf_processing import extract_text_from_pdf, get_embedding, save_embeddings_to_chromadb
import chromadb

class ChromadbAgent:
    def __init__(self, collection_name="pdf_embeddings"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    # Pretraga u ChromaDB na osnovu embeddinga
    def query(self, query_text, n_results=5):
        query_embedding = get_embedding(query_text)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results

    # Dodavanje novih embeddinga u ChromaDB
    def add_pdf_to_db(self, pdf_path, metadata):
        text = extract_text_from_pdf(pdf_path)
        embedding = get_embedding(text)
        save_embeddings_to_chromadb(embedding, metadata, collection_name=self.collection.name)

# Primer korišćenja agenta
chromadb_agent = ChromadbAgent()

# Dodaj PDF u bazu
chromadb_agent.add_pdf_to_db("data/Cube AI - Datasheet.pdf", metadata={"title": "Cube AI Datasheet"})

# Pretraga u bazi
query = "What is Cube AI?"
results = chromadb_agent.query(query)
print(results)
