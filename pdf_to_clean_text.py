import pdfplumber
import re

def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text):
    # Remove bracketed citations like [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove parenthetical citations like (Smith et al., 2020)
    text = re.sub(r'\([^\)]+et al\.,?\s*\d{4}\)', '', text)
    
    # Remove section numbers at start of lines like "1.", "2.1", "3.2.1", etc.
    text = re.sub(r'^\d+(\.\d+)*\s+', '', text, flags=re.MULTILINE)
    
    # Remove all-caps headings or lines that are just numbers
    text = re.sub(r'^[A-Z\s]{3,}$', '', text, flags=re.MULTILINE)
    
    # Optional: collapse multiple spaces/newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

pdf_path = "your_file.pdf"
raw_text = read_pdf(pdf_path)
cleaned_text = clean_text(raw_text)
