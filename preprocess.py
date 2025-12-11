import sys
import os
import re
import shutil
from pdfminer.high_level import extract_text
from tqdm import tqdm


def read_pdf(filepath):
    return extract_text(filepath)


def clean_pdf_text(text):

    text = text.replace("\r", " ")

    # Remove Email Addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '',
        text
    )

    # Remove Citations: [1], [1,2], [3–6], [1, 3, 5]
    text = re.sub(
        r'\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]',
        '',
        text
    )

    # Remove patterns like 8:197, 12:440–445, 3:10-12
    text = re.sub(
        r'\b\d{1,4}\s*[:]\s*\d{1,5}(?:\s*[-–]\s*\d{1,5})?\b',
        '',
        text
    )

    # Remove special symbols often used in footnotes
    text = re.sub(r'[\*\†\‡]+', '', text)

    # Remove long author lists
    text = re.sub(
        r'(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s[A-Z]{1,3},\s*){2,}',
        '',
        text
    )

    # Remove "X et al." with or without year, inside or outside parentheses (case-insensitive)
    text = re.sub(
        r'\(?\b[A-Za-z][A-Za-z\'\-\u00C0-\u017F]*(?:\s+[A-Za-z][A-Za-z\'\-\u00C0-\u017F]*)*\s+et\.?\s*al\.?(?:,?\s*\d{4})?\)?',
        '',
        text,
        flags=re.IGNORECASE
    )


    # Remove parenthetical number-only citations: (1), (1,2), (3-5)
    text = re.sub(
        r'\(\s*\d+(?:\s*[-–,]\s*\d+)*\s*\)',
        '',
        text
    )

    # Remove URLs
    text = re.sub(r'https?://[^\s\)\]]+', '', text)
    text = re.sub(r'www\.[^\s\)\]]+', '', text)

    # Fix hyphenated word breaks
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip section headers (ALL CAPS)
        if re.fullmatch(r'[A-Z][A-Z\s\-]{3,}', stripped):
            continue

        # Skip table-like rows
        if stripped.count("|") >= 2:
            continue

        # Skip rows with lots of spacing & numbers (likely tables)
        if (
            sum(c.isdigit() for c in stripped) > len(stripped) * 0.5 and
            len(re.findall(r'\s{2,}', stripped)) >= 2
        ):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)


    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def save_cleaned_text(text, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_pdfs(input_dir, output_dir):

    pdf_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".pdf"))

    for file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(input_dir, file)

        raw_text = read_pdf(pdf_path)
        cleaned_text = clean_pdf_text(raw_text)

        filename = os.path.splitext(file)[0]
        save_cleaned_text(cleaned_text, output_dir, filename)


def process_tabular_files(input_dir, output_dir):

    tabular_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith((".csv", ".tsv"))
    )

    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(tabular_files, desc="Processing CSV/TSV", unit="file"):
        path = os.path.join(input_dir, file)

        with open(path, "r", encoding="utf-8") as infile:
            content = infile.read()

        if file.endswith(".csv"):
            cleaned = content.replace(",", "|")
        else:
            cleaned = content.replace("\t", "|")

        filename = os.path.splitext(file)[0]
        save_cleaned_text(cleaned, output_dir, filename)

def process_text_files(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    text_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".txt")
    )

    for file in text_files:
        shutil.copy2(
            os.path.join(input_dir, file),
            os.path.join(output_dir, file)
        )

def copy_dataset(dataset_dir, raw_dataset_dir):

    os.makedirs(raw_dataset_dir, exist_ok=True)

    for root, dirs, files in tqdm(os.walk(dataset_dir), desc="Copying files"):
        for file in files:
            shutil.copy2(
                os.path.join(root, file),
                os.path.join(raw_dataset_dir, file)
            )

def main():
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
    dataset_dir = os.path.join(parent_dir, "dataset")

    raw_dataset_dir = os.path.join(parent_dir, "raw_dataset")
    processed_dataset_dir = os.path.join(parent_dir, "processed_dataset")

    # Step 1: Copy dataset
    copy_dataset(dataset_dir, raw_dataset_dir)

    # Step 2: Process PDFs
    process_pdfs(raw_dataset_dir, processed_dataset_dir)

    # Step 3: Process CSV/TSV
    process_tabular_files(raw_dataset_dir, processed_dataset_dir)

    # Step 4: Copy text files
    process_text_files(raw_dataset_dir, processed_dataset_dir)


if __name__ == "__main__":
    main()
