import pdfplumber
import re

# def read_pdf(file_path):
#     alltext = ""
    # with pdfplumber.open(file_path) as pdf:
    #     for page in pdf.pages:
    #         text += page.extract_text() + "\n"
    # return text
    
    # with pdfplumber.open(file_path) as pdf:
    #     for page in pdf.pages:
    #         # Get the page dimensions
    #         width = page.width
    #         height = page.height
            
    #         # Define left and right column bounding boxes
    #         left_bbox = (0, 0, width / 2, height)
    #         right_bbox = (width / 2, 0, width, height)
            
    #         # Extract text from left column
    #         left_text = page.within_bbox(left_bbox).extract_text() or ""
            
    #         # Extract text from right column
    #         right_text = page.within_bbox(right_bbox).extract_text() or ""
            
    #         # Combine left -> right
    #         page_text = left_text + "\n" + right_text
    #         all_text += page_text + "\n"
    
    # return all_text
    
# import fitz  # PyMuPDF
# import re
# import os

from pdfminer.high_level import extract_text

def read_pdf2(filepath):
    text = extract_text(filepath)
    return text
# print(text)

# def read_pdf1(
#     pdf_path: str, 
#     column_margin: int = 10,
#     top_margin_percent: float = 0.08, # Ignore top 8% of the page
#     bottom_margin_percent: float = 0.08 # Ignore bottom 8% of the page
# ) -> str:
#     """
#     Extracts text from a 2-column PDF, ignoring headers and footers
#     by using positional margins.
#     """
#     if not os.path.exists(pdf_path):
#         return "Error: PDF file not found."
    
#     doc = fitz.open(pdf_path)
#     full_text = ""
    
#     for page in doc:
#         # Get page dimensions
#         page_width = page.rect.width
#         page_height = page.rect.height
        
#         # Calculate the pixel values for the margins
#         y0 = page_height * top_margin_percent
#         y1 = page_height * (1 - bottom_margin_percent)
        
#         mid_x = page_width / 2

#         # Define columns within the content area (between the margins)
#         left_column = fitz.Rect(0, y0, mid_x - column_margin / 2, y1)
#         right_column = fitz.Rect(mid_x + column_margin / 2, 0, page_width, y1)
        
#         # Extract text from each column's content area
#         full_text += page.get_text(clip=left_column)
#         full_text += page.get_text(clip=right_column)
        
#     doc.close()
#     return full_text

import re

def clean_text(text):
    # Remove bracketed citations like [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove parenthetical citations like (Smith et al., 2020)
    text = re.sub(r'\([^\)]+et al\.,?\s*\d{4}\)', '', text)
    
    # Remove section numbers at start of lines like "1.", "2.1", "3.2.1", etc.
    text = re.sub(r'^\d+(\.\d+)*\s+', '', text, flags=re.MULTILINE)
    
    # Remove all-caps headings or lines that are just numbers
    text = re.sub(r'^[A-Z\s]{3,}$', '', text, flags=re.MULTILINE)
    
    # --- Remove table-like lines ---
    # Heuristic: lines with lots of whitespace-separated "columns" or digits
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Count words separated by 2+ spaces or tabs
        if len(re.findall(r'  +|\t', line)) >= 2:
            continue  # likely a table row → skip
        # Skip lines with many pipe characters (markdown-style tables)
        if line.count("|") >= 2:
            continue
        cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    # Optional: collapse multiple spaces/newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


# def clean_text(text):
#     # Remove bracketed citations like [1], [12], etc.
#     text = re.sub(r'\[\d+\]', '', text)
    
#     # Remove parenthetical citations like (Smith et al., 2020)
#     text = re.sub(r'\([^\)]+et al\.,?\s*\d{4}\)', '', text)
    
#     # Remove section numbers at start of lines like "1.", "2.1", "3.2.1", etc.
#     text = re.sub(r'^\d+(\.\d+)*\s+', '', text, flags=re.MULTILINE)
    
#     # Remove all-caps headings or lines that are just numbers
#     text = re.sub(r'^[A-Z\s]{3,}$', '', text, flags=re.MULTILINE)
    
#     # Optional: collapse multiple spaces/newlines
#     text = re.sub(r'\n\s*\n', '\n\n', text)
#     text = re.sub(r'[ \t]+', ' ', text)
    
#     return text.strip()

# pdf_path = "your_file.pdf"
# raw_text = read_pdf(pdf_path)
# cleaned_text = clean_text(raw_text)
