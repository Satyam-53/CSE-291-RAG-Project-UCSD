#!/usr/bin/env python
# coding: utf-8

# Prepare the dataset.

# In[4]:


# Add parent directory to sys.path
import sys
import os
import re
import importlib
import pdfminer
from pdfminer.high_level import extract_text
from tqdm import tqdm
import shutil


# Code to extract the dataset
def read_pdf(filepath): # takes in the absolute filepath
    text = extract_text(filepath)
    return text


# Code to clean the dataset
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


def save_cleaned_text(text, output_dir, filename):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the full output path
    output_path = os.path.join(output_dir, f"{filename}.txt")

    # Save text to the file (UTF-8 handles special characters)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    # print(f"✅ Saved cleaned text to: {output_path}")


def clean_raw_pdf_text(raw_text):
    raw_lines = raw_text.strip().split("\n")
    clean_text = ''
    for line in raw_lines:
        if line.strip().lower() in ['', 'abstract', 'introduction', 'discussion', 'results']:
            continue
        elif line.strip().lower() == 'references':
            break
        else:
            if line[0]=='-':
                line = line[1:]
                if len(line) == 0:
                    continue
            if line[-1]=='-':
                line = line[:-1]
                if len(line) == 0:
                    continue
            else:
                line = line + ' '

            clean_text += line
    return clean_text


def process_pdfs(input_dir, output_dir):
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    for file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(input_dir, file)
        raw_text = read_pdf(pdf_path)

        cleaned_text = clean_raw_pdf_text(raw_text)
        
        cleaned_text = re.sub(r"\[\d+\]", '', cleaned_text)
        cleaned_text = re.sub(r'\([^\)]+et al\.,?\s*\d{4}\)', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'https://\S+\s*', '', cleaned_text)
        cleaned_text = re.sub(r'www.\S+\s*', '', cleaned_text)

        filename = os.path.splitext(file)[0]
        save_cleaned_text(cleaned_text, output_dir, filename)


def process_tabular_files(input_dir, output_dir):
    # Collect all .csv and .tsv files
    tabular_files = [f for f in os.listdir(input_dir) if f.endswith((".csv", ".tsv"))]

    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(tabular_files, desc="Processing CSV/TSV files", unit="file"):
        input_path = os.path.join(input_dir, file)
        # output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".txt")

        # Read the file contents
        with open(input_path, "r", encoding="utf-8") as infile:
            content = infile.read()

        # Replace only the correct delimiter
        if file.endswith(".csv"):
            cleaned_content = content.replace(",", "|")
        elif file.endswith(".tsv"):
            cleaned_content = content.replace("\t", "|")
        else:
            continue  # safety check
        
        filename = os.path.splitext(file)[0]
        save_cleaned_text(cleaned_content, output_dir, filename)

        # # Save the cleaned version
        # with open(output_path, "w", encoding="utf-8") as outfile:
        #     outfile.write(cleaned_content)

def process_text_files(input_dir, output_dir):
    os.makedirs(input_dir, exist_ok=True)
    
    text_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for file in text_files:
        dst_path = os.path.join(output_dir, file)
        src_path = os.path.join(input_dir, file)
        
        shutil.copy2(src_path, dst_path)

def copy_dataset(dataset_dir, raw_dataset_dir):
    os.makedirs(raw_dataset_dir, exist_ok=True)
    for root, dirs, files in tqdm(os.walk(dataset_dir), desc="Copying files", unit="file"):
        for file in files:
            dst_path = os.path.join(raw_dataset_dir, file)
            src_path = os.path.join(root, file)
            
            shutil.copy2(src_path, dst_path)
            

def main():
    # Get relevant directory paths
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    notebook_dir = os.getcwd()
    dataset_dir = os.path.join(parent_dir, "dataset")

    sys.path.append(parent_dir)

    #Processing Cell
    raw_dataset_dir = os.path.join(parent_dir, "raw_dataset")
    copy_dataset(dataset_dir, raw_dataset_dir)
    processed_dataset_dir = os.path.join(parent_dir, "processed_dataset")
    process_pdfs(raw_dataset_dir, processed_dataset_dir)
    process_tabular_files(raw_dataset_dir, processed_dataset_dir)
    process_text_files(raw_dataset_dir, processed_dataset_dir)

if __name__=='__main__':
    main()




