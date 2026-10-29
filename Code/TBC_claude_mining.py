'''
Gen AI agent: Enhanced Data Extraction Pipeline with Claude
Author: Jackson Wu (223103762)
Email: jackson911007@gmail.com
Description: Improved pipeline for extracting material properties from PDFs using Claude via AWS Bedrock
'''

import os
import json
import csv
import pandas as pd
import time
import re
import html
from typing import List, Tuple, Dict, Any
import boto3
from botocore.exceptions import ClientError

# Import table extraction libraries
try:
    import camelot
    _HAVE_CAMELOT = True
except Exception:
    _HAVE_CAMELOT = False

import pdfplumber

# ==================== AWS Bedrock Configuration ====================
REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ==================== Directory Configuration ====================
base_location = '/mnt/c/Users/223103762/Documents (Local Only)/LLM_test/data_extraction/'
pdf_folder = base_location + '2026_new_data/'
output_txt = base_location + 'txt/'
output_csv_folder = base_location + 'output_csv/'

# Create output directories if they don't exist
os.makedirs(output_txt, exist_ok=True)
os.makedirs(output_csv_folder, exist_ok=True)

# ==================== Claude Bedrock Helper ====================
def invoke_claude(prompt: str, max_tokens: int = 16000) -> str:
    """Send a prompt to Claude via AWS Bedrock and return the text response."""
    client = boto3.client("bedrock-runtime", region_name=REGION)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,  # Lower temperature for more consistent extraction
    }

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
    except ClientError as err:
        code = err.response["Error"]["Code"]
        raise SystemExit(f"Bedrock call failed ({code}): {err}") from err

    payload = json.loads(response["body"].read())
    return "".join(block.get("text", "") for block in payload.get("content", []))

# ==================== Text Processing Utilities ====================
ELEM_RE = re.compile(r"^[A-Z][a-z]?$")
NUM_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

def _merge_split_decimals(tokens: List[str]) -> List[str]:
    """Merge decimals that were split during extraction."""
    out, i = [], 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i].isdigit()
            and tokens[i+1] in (".", "·", "⋅", "․")
            and tokens[i+2].isdigit()
        ):
            out.append(tokens[i] + "." + tokens[i+2])
            i += 3
        else:
            out.append(tokens[i])
            i += 1
    return out

def _join_elem_nums_in_row(cells: List[str]) -> List[str]:
    """Join element symbols with their numbers (e.g., Al 2 O 3 -> Al2O3)."""
    cells = [re.sub(r"(?<=\d),(?=\d)", ".", c) for c in cells]
    cells = _merge_split_decimals(cells)
    out, i = [], 0
    while i < len(cells):
        tok = cells[i]
        if ELEM_RE.match(tok) and i + 1 < len(cells) and NUM_RE.match(cells[i + 1]):
            out.append(f"{tok}{cells[i + 1]}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out

# Super/subscript handling
SUP_MAP = str.maketrans({"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾"})
SUB_MAP = str.maketrans({"0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉","+":"₊","-":"₋","=":"₌","(":"₍",")":"₎"})
SUP_TO_PLAIN = str.maketrans({v:k for k,v in SUP_MAP.items()})
SUB_TO_PLAIN = str.maketrans({v:k for k,v in SUB_MAP.items()})

_ELEM_RE = r"(?:[A-Z][a-z]?)"
_FORMULA_NUM = re.compile(rf"({_ELEM_RE}|\))(\d+)")
_SUP_POW = re.compile(r"(\^)(\(?[0-9nmkxy+\-]+\)?)")
_CHARGE_1 = re.compile(rf"({_ELEM_RE}\d*)([+-])")
_CHARGE_2 = re.compile(rf"({_ELEM_RE}\d*)(\^\d+[+-])")
_UNIT_POW = re.compile(r"\b([A-Za-zμ]+)\^([0-9+\-]+)\b")
_ISOTOPE = re.compile(r"\b([0-9]{1,3})-([A-Z][a-z]?)\b")

def _to_sub(s: str) -> str:
    return s.translate(SUB_MAP)

def _to_sup(s: str) -> str:
    return s.translate(SUP_MAP)

def _preserve_super_sub(text: str) -> str:
    """Convert common patterns to Unicode super/subscripts."""
    if not text:
        return text

    text = _FORMULA_NUM.sub(lambda m: m.group(1) + _to_sub(m.group(2)), text)
    text = _UNIT_POW.sub(lambda m: f"{m.group(1)}{_to_sup(m.group(2))}", text)
    text = _SUP_POW.sub(lambda m: _to_sup(m.group(2)), text).replace("^", "")
    text = _CHARGE_2.sub(lambda m: m.group(1) + _to_sup(m.group(2)[1:]), text)
    text = _CHARGE_1.sub(lambda m: m.group(1) + _to_sup(m.group(2)), text)
    text = _ISOTOPE.sub(lambda m: f"{_to_sup(m.group(1))}{m.group(2)}", text)

    return text

def _sparsify_text(block: str) -> str:
    """Make text sparse and preserve super/subscripts."""
    if not block:
        return ""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in block.splitlines()]
    lines = [ln for ln in lines if ln]
    txt = "\n".join(lines)
    return _preserve_super_sub(txt)

def _normalize_pdf_glyphs(s: str) -> str:
    """Normalize PDF glyph artifacts."""
    s = s.replace("·", ".").replace("⋅", ".").replace("․", ".")
    for ch in (" "," "," "," "," "," ","​","‌","‍"):
        s = s.replace(ch, " ")
    return s

def _sparsify_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and sparsify table dataframes."""
    if df is None:
        return None
    df = df.map(lambda x: (str(x).strip() if x is not None else ""))
    df = df.map(_normalize_pdf_glyphs)
    df = df.map(_preserve_super_sub)
    df = df.loc[~(df.eq("").all(axis=1))]
    df = df.loc[:, ~(df.eq("").all(axis=0))]
    return df.reset_index(drop=True)

def _df_to_sparse_string(df: pd.DataFrame, join_chemistry: bool = True) -> str:
    """Convert DataFrame to TSV-like string."""
    if df is None or df.empty:
        return ""
    df = df.map(lambda x: (str(x).strip() if x is not None else ""))
    df = df.map(_normalize_pdf_glyphs)
    df = df.map(_preserve_super_sub)

    rows = df.astype(str).values.tolist()
    out_lines = []
    for row in rows:
        row = [c.strip() for c in row]
        if join_chemistry:
            row = _join_elem_nums_in_row(row)
        line = "\t".join(row)
        out_lines.append(line)
    return "\n".join(out_lines)

# ==================== Table Extraction ====================
def _extract_tables_with_camelot(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables using Camelot."""
    out = []
    if not _HAVE_CAMELOT:
        return out

    def _run(flavor: str):
        try:
            return camelot.read_pdf(pdf_path, pages="all", flavor=flavor, suppress_stdout=True)
        except Exception:
            return None

    for flavor in ("lattice", "stream"):
        tables = _run(flavor)
        if tables and tables.n > 0:
            for t in tables:
                try:
                    df = _sparsify_table_df(t.df)
                    out.append({
                        "page": t.page,
                        "flavor": flavor,
                        "data_tsv": _df_to_sparse_string(df)
                    })
                except Exception:
                    continue
            if out:
                break
    return out

def _extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables using pdfplumber."""
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    raw_tables = page.extract_tables()
                except Exception:
                    raw_tables = []
                for tbl in raw_tables or []:
                    try:
                        df = pd.DataFrame(tbl)
                        df = _sparsify_table_df(df)
                        if df is not None and not df.empty:
                            results.append({
                                "page": i,
                                "flavor": "plumber",
                                "data_tsv": _df_to_sparse_string(df)
                            })
                    except Exception:
                        continue
    except Exception:
        pass
    return results

def extract_sparse_tables_and_context(pdf_file: str) -> List[str]:
    """Extract text, tables, and context from PDF."""
    contexts: List[str] = []
    tables: List[str] = []

    fname = os.path.basename(pdf_file)

    # Extract text
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    txt = page.extract_text(x_tolerance=1.5, y_tolerance=3.0) or ""
                except Exception as e:
                    print(f"  ⚠ Text extract error in page {i}: {e}")
                    txt = ""
                sparse = _sparsify_text(txt)
                if sparse:
                    contexts.append(f"{fname}::p{i}::\n{sparse}")
    except Exception as e:
        print(f"  ❌ PDF reading error: {e}")

    # Extract tables
    extracted = _extract_tables_with_camelot(pdf_file)
    if not extracted:
        try:
            extracted = _extract_tables_with_pdfplumber(pdf_file)
        except Exception as e:
            print(f"  ⚠ Table extraction error: {e}")
            extracted = []

    for t in extracted:
        if t.get("data_tsv"):
            tables.append(f"{fname}::p{t['page']}::{t['flavor']}::\n{t['data_tsv']}")

    print(f"  📄 Extracted {len(contexts)} text blocks and {len(tables)} tables")
    return contexts + tables

# ==================== Data Cleaning ====================
def clean_claude_output(raw_text: str) -> pd.DataFrame:
    """Clean Claude's JSON output and convert to DataFrame."""
    text = raw_text.strip()

    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE|re.MULTILINE).strip()

    # Strip any prose/headings Claude adds before the JSON array (e.g. "# Extracted Materials Data\n")
    # Claude sometimes prefixes the JSON with a markdown heading or explanation sentence.
    bracket_idx = text.find("[")
    if bracket_idx > 0:
        text = text[bracket_idx:]

    # HTML unescape
    text = html.unescape(text)

    # Strip quotes if wrapped
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Normalize escaped characters
    text = text.replace(r"\&quot;", '"')
    text = text.replace(r"\\\"", r"\"")
    text = text.replace(r"\"", '"')
    text = text.replace(r"\/", "/")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing error: {e}")
        print(f"  Raw text (first 500 chars): {text[:500]}")
        return pd.DataFrame()

    if not isinstance(data, list):
        print(f"  ⚠ Expected list, got {type(data)}")
        return pd.DataFrame()

    if not data:
        print(f"  ℹ No data extracted")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# ==================== Main Extraction Function ====================
def claude_extract_information_csv(pdf_file: str, pdf_name: str):
    """Extract material properties from PDF using Claude."""
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_name}")
    print(f"{'='*60}")

    start = time.perf_counter()

    # Extract text and tables
    print("Step 1: Extracting content from PDF...")
    paragraph_info = extract_sparse_tables_and_context(pdf_file)
    paragraph = "\n".join(paragraph_info)

    # Save raw extracted text
    txt_output_path = os.path.join(output_txt, f"{pdf_name}_extracted.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(paragraph)
    print(f"  ✅ Raw text saved: {txt_output_path}")

    # Prepare prompt for Claude
    print("Step 2: Sending to Claude for data extraction...")
    prompt = f"""You are a materials science data extraction expert. Extract ALL relevant material properties from the following scientific document.

Extract the following properties for each distinct material found:

1. Material_Name: Full material name (use complete names, not abbreviations)
2. Chemical_formula: Chemical formula if available (convert subscripts/superscripts to normal numbers)
3. Crystal_structure: Structure type (e.g., pyrochlore, fluorite, cubic, hexagonal, cubic, tetragonal.. etc.)
4. Crystal_measured_temperature_K: Temperature at which crystal structure was observed (in Kelvin)
5. Thermal_Conductivity_W_mK: Thermal conductivity value in W/(m·K)
6. Thermal_Conductivity_measured_temperature_K: Temperature at which thermal conductivity was measured (in Kelvin)
7. Thermal_expansion_coefficient_/K: Thermal expansion coefficient in /K
8. Thermal_expansion_coefficient_measured_temperature_K: Temperature at which TEC was measured (in Kelvin)
9. Hardness_GPa: Hardness value in GPa
10. Hardness_GPa_type: Type of hardness measurement (e.g., Vickers, Knoop, Rockwell)
10. Toughness_MPam0.5: Fracture toughness in MPa·m^0.5
11. Toughness_measured_temperature_K: Temperature at which toughness was measured (in Kelvin)
12. Relative_density_or_porosity_%: Relative density or porosity in percentage
13. DOI: DOI of the paper or literature reference

IMPORTANT INSTRUCTIONS:
- Convert ALL temperatures to Kelvin: °C to K = C + 273.15, °F to K = (F - 32) × 5/9 + 273.15
- Use "300" for room temperature (RT) or "room temperature"
- If temperature not specified, use "Not available"
- Convert fractions in formulas to decimal (e.g., "1/2" → "0.5", "2/3" → "0.67")
- For ranges like "2~3" or "2-3", report as "2-3"
- For approximate values like "~2", report as "~2"
- ONLY extract experimental data (ignore theoretical/simulated values)
- Extract data from BOTH text and tables
- If a property is not found, use "Not available"

Return ONLY a valid JSON array with NO additional text, explanation, or markdown formatting. Each entry must follow this exact structure:

[
  {{
    "Material_Name": "value",
    "Chemical_formula": "value",
    "Crystal_structure": "value",
    "Crystal_measured_temperature_K": "value",
    "Thermal_Conductivity_W_mK": "value",
    "Thermal_Conductivity_measured_temperature_K": "value",
    "Thermal_expansion_coefficient_/K": "value",
    "Thermal_expansion_coefficient_measured_temperature_K": "value",
    "Hardness_GPa": "value",
    "Hardness_GPa_type": "value",
    "Toughness_MPam0.5": "value",
    "Toughness_measured_temperature_K": "value",
    "Relative_density_or_porosity_%": "value",
    "DOI": "value"
  }}
]

Document content:
{paragraph}
"""

    # Call Claude
    try:
        raw_response = invoke_claude(prompt, max_tokens=16000)
        print(f"  📥 Received response from Claude ({len(raw_response)} chars)")
    except Exception as e:
        print(f"  ❌ Error calling Claude: {e}")
        return pd.DataFrame()

    # Save raw Claude output
    claude_output_path = os.path.join(output_txt, f"{pdf_name}_claude_response.txt")
    with open(claude_output_path, "w", encoding="utf-8") as f:
        f.write(raw_response)
    print(f"  ✅ Claude response saved: {claude_output_path}")

    # Parse and save CSV
    print("Step 3: Parsing Claude output...")
    df = clean_claude_output(raw_response)

    if not df.empty:
        csv_output_path = os.path.join(output_csv_folder, f"{pdf_name}_extracted_data.csv")
        df.to_csv(csv_output_path, index=False)
        print(f"  ✅ CSV saved: {csv_output_path}")
        print(f"  📊 Extracted {len(df)} material records")
    else:
        print(f"  ⚠ No data extracted or parsing failed")

    end = time.perf_counter()
    print(f"  ⏱ Total time: {end - start:.2f} seconds")

    return df

# ==================== Main Processing Loop ====================
def process_all_pdfs():
    """Process all PDFs in the 2026_new_data folder."""
    print("\n" + "="*60)
    print("PDF Data Extraction Pipeline - Claude Edition")
    print("="*60)

    # Get all PDF files
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    print(f"\nFound {len(pdf_files)} PDF files in {pdf_folder}")

    processed = 0
    skipped = 0
    failed = 0

    for pdf_file in sorted(pdf_files):
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(pdf_folder, pdf_file)

        # Check if already processed
        csv_output_path = os.path.join(output_csv_folder, f"{pdf_name}_extracted_data.csv")
        if os.path.exists(csv_output_path):
            print(f"\n⏭ Skipping {pdf_file} (already processed)")
            skipped += 1
            continue

        # Process PDF
        try:
            df = claude_extract_information_csv(pdf_path, pdf_name)
            if not df.empty:
                processed += 1
            else:
                failed += 1
        except Exception as e:
            # PDFs can fail for several reasons: password-protected or encrypted files,
            # corrupted/malformed PDF structure, scanned image-only PDFs with no extractable text,
            # unsupported encoding, or AWS Bedrock API errors (timeout, token limit exceeded).
            print(f"\n❌ Failed to process {pdf_file}: {e}")
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"✅ Successfully processed: {processed}")
    print(f"⏭ Skipped (already done): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_csv_folder}")

# ==================== Concatenate All Output CSVs ====================
def concat_all_csvs() -> pd.DataFrame:
    """Concatenate all extracted CSVs in output_csv_folder into a single DataFrame."""
    csv_files = [f for f in os.listdir(output_csv_folder) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in output folder.")
        return pd.DataFrame()

    dfs = []
    for fname in sorted(csv_files):
        fpath = os.path.join(output_csv_folder, fname)
        try:
            df = pd.read_csv(fpath)
            # Track which source file each row came from
            df['source_file'] = fname
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Could not read {fname}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(output_csv_folder, 'all_extracted_data.csv')
    combined.to_csv(combined_path, index=False)
    print(f"✅ Combined {len(dfs)} CSVs into {len(combined)} rows → {combined_path}")
    return combined


if __name__ == "__main__":
    # Process all PDFs in batch
    process_all_pdfs()

    # Concatenate all output CSVs into one file
    combined_df = concat_all_csvs()
    print(combined_df)

    # Or process a single PDF (uncomment below):
    # pdf_name = "2_good_review"
    # pdf_path = os.path.join(pdf_folder, f"{pdf_name}.pdf")
    # df = claude_extract_information_csv(pdf_path, pdf_name)
    # print(df)
