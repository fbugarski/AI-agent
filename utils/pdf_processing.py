import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import numpy as np

# Globalne promenljive za model i tokenizator (učitavaće se samo jednom)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

# Kreiraj klijenta za ChromaDB samo jednom
client = chromadb.Client()

# Funkcija za vađenje teksta iz PDF-a
def extract_text_from_pdf(pdf_path):
    """
    Ekstraktuje sav tekst iz PDF-a
    :param pdf_path: putanja do PDF fajla
    :return: tekst iz PDF-a
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Greška prilikom vađenja teksta iz PDF-a: {e}")
        return ""

# Funkcija za generisanje embeddinga
def get_embedding(text):
    """
    Generiše embedding za dati tekst koristeći model iz HuggingFace biblioteke.
    :param text: tekst za koji treba generisati embedding
    :return: embedding kao numpy array
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        # Uzimanje output-a iz last_hidden_state za bolje rezultate
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Koristi mean na svim tokenima
        return embedding
    except Exception as e:
        print(f"Greška prilikom generisanja embeddinga: {e}")
        return None  # Vraća None u slučaju greške

# Funkcija za skladištenje embeddinga u ChromaDB
def save_embeddings_to_chromadb(embedding, metadata, collection_name="pdf_embeddings"):
    """
    Čuva generisane embeddinge u ChromaDB kolekciji
    :param embedding: numpy array sa embeddingom
    :param metadata: metapodaci povezani sa embeddingom
    :param collection_name: ime kolekcije u ChromaDB-u
    """
    try:
        # Proveriti da li kolekcija već postoji, pa je ili dohvatiti ili kreirati
        collection = client.get_or_create_collection(name=collection_name)

        # Generisanje jedinstvenog ID-a, na osnovu dokumenta ili nekog unikatnog faktora
        unique_id = f"{metadata['title']}_embedding"

        # Dodavanje embeddinga u kolekciju
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[unique_id]  # Koristi jedinstveni ID za svaki dokument
        )
        print(f"Embedding sačuvan u kolekciji: {collection_name}")
    except Exception as e:
        print(f"Greška prilikom skladištenja embeddinga u ChromaDB: {e}")

# Primer kako koristiti funkcije
if __name__ == "__main__":
    # Putanja do PDF-a koji treba obraditi
    pdf_path = "data/Cube AI - Datasheet.pdf"
    
    # Ekstraktovanje teksta iz PDF-a
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        # Generisanje embeddinga za tekst
        embedding = get_embedding(text)
        
        if embedding is not None:  # Proveri da li je embedding uspešno generisan
            # Metapodaci za PDF (npr. naziv dokumenta)
            metadata = {"title": "Cube AI Datasheet"}
            
            # Skladištenje embeddinga u ChromaDB
            save_embeddings_to_chromadb(embedding, metadata)
        else:
            print("Greška: Embedding nije generisan.")
    else:
        print("Greška: Tekst nije izvučen iz PDF-a.")
