import os
import fitz  # PyMuPDF
import camelot
import pdfplumber

input_dir = "arxiv_pdfs_pyrochlore"
output_dir = "parsed_texts"
os.makedirs(output_dir, exist_ok=True)

def extract_text_paragraphs(pdf_path):
    paragraphs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Split text into paragraphs by double newlines
                    for para in text.split('\n\n'):
                        clean_para = para.strip().replace('\n', ' ')
                        if len(clean_para) > 30:  # skip short lines
                            paragraphs.append(clean_para)
    except Exception as e:
        print(f"❌ Text extraction failed: {pdf_path} ({e})")
    return paragraphs

def extract_tables_as_paragraphs(pdf_path):
    table_paragraphs = []
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for i, table in enumerate(tables):
            df = table.df
            para = f"[Table {i+1}]\n" + df.to_string(index=False, header=True)
            table_paragraphs.append(para)
    except Exception as e:
        print(f"⚠️ Table extraction failed (Camelot): {pdf_path} ({e})")
    return table_paragraphs

for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        print(f"🔍 Parsing: {filename}")

        text_paragraphs = extract_text_paragraphs(pdf_path)
        table_paragraphs = extract_tables_as_paragraphs(pdf_path)

        all_paragraphs = text_paragraphs + table_paragraphs

        output_file = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
        with open(output_file, "w", encoding="utf-8") as f:
            for para in all_paragraphs:
                f.write(para + "\n\n")

        print(f"✅ Saved: {output_file} ({len(all_paragraphs)} paragraphs)")
