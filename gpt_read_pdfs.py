import os
import fitz  # PyMuPDF
import openai
import csv
import time

# Set your OpenAI API key
openai.api_key = "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"

# Directory with PDFs
pdf_dir = "arxiv_pdfs_pyrochlore"
output_csv = "thermal_conductivity_extracted.csv"

# Function: extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Function: ask GPT to extract values
def extract_thermal_conductivity(text):
    prompt = (
        "Extract all thermal conductivity values (in W/mK or similar) and their corresponding material names "
        "from this scientific article. Present results in a structured list like: 'Material: X, Thermal Conductivity: Y W/mK'.\n\n"
        "Text:\n"
        f"{text[:3000]}"  # truncate to avoid context limit
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("Error querying OpenAI:", e)
        return "ERROR"

# Prepare CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PDF File", "Extracted Thermal Conductivity Info"])

    # Process all PDFs
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"🔍 Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            if text:
                result = extract_thermal_conductivity(text)
                writer.writerow([filename, result])
                time.sleep(2)  # Rate limit buffer
