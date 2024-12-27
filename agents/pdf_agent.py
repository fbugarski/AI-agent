from transformers import pipeline

# Kreiramo pipeline za sažimanje teksta
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def process_pdf(file_path):
    """
    Process the PDF and extract its text.
    
    Args:
        file_path (str): The path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    # Ovdje ide logika za ekstrakciju teksta iz PDF-a
    # Ovo je samo placeholder funkcija. Implementirajte stvarno čitanje PDF-a
    return "This is the extracted text from the PDF file."

def summarize_pdf(file_path):
    """
    Process the PDF and summarize its content.

    Args:
        file_path (str): The path to the PDF file.
    
    Returns:
        str: A summary of the PDF content.
    """
    # Procesuiranje PDF-a direktno u ovoj funkciji
    text = process_pdf(file_path)
    
    # Sažimanje teksta sa povećanom maksimalnom dužinom sažetka
    summary = summarizer(text, max_length=500, min_length=100, do_sample=False)
    
    return summary[0]['summary_text']
