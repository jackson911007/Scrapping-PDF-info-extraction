'''
Gen AI agent: Jacky
Author: Jackson Wu (223103762)
Email: jackson911007@gmail.com
'''
import os # PyMuPDF
import camelot
import pdfplumber
import google.generativeai as genai
import json
import csv
import pandas as pd
import ast
import re
from typing import List, Tuple, Dict, Any
from google import genai
import html
import re
import time

import pandas as pd
try:
    import camelot
    _HAVE_CAMELOT = True
except Exception:
    _HAVE_CAMELOT = False



# Paste your actual API key here
api_key = 'AIzaSyBSQBqJhJjYz6umhqUS13PZ-xAnXhXg4cg'

base_location = '//projects//PFAS_polymer_mlaas//LLM//TBC_project//'
pdf_folder = base_location + 'Data//'

# This is the PDf information
output_csv = base_location + 'Output//'
output_sparse_txt = base_location + 'Output//sparse_txt//'
output_txt = base_location + 'Output//txt//'
output_csv_final = base_location + 'Output//data_csv//'

#---------------------Gemini------------------------------
# Configure the Generative AI client

client = genai.Client(api_key=api_key)

# # Generate content
response = client.models.generate_content(model = "gemini-2.5-flash", contents = "Say hello in one sentence.")

# Print the response
print("✅ Gemini response:", response.text)
#----------------------------------------------------------


ELEM_RE = re.compile(r"^[A-Z][a-z]?$")
NUM_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

def _merge_split_decimals(tokens: List[str]) -> List[str]:
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



# --- Superscript/Subscript maps ---
# SUP_MAP = str.maketrans({
#     "0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹",
#     "+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾","n":"ⁿ","x":"ˣ","y":"ʸ","m":"ᵐ","k":"ᵏ"
# })
# SUB_MAP = str.maketrans({
#     "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉",
#     "+":"₊","-":"₋","=":"₌","(":"₍",")":"₎","a":"ₐ","e":"ₑ","i":"ᵢ","j":"ⱼ","o":"ₒ","r":"ᵣ","u":"ᵤ","v":"ᵥ","x":"ₓ"
# })

# Optional: map superscripts/subscripts back to plain digits/signs at the very end
SUP_MAP = str.maketrans({"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾"})
SUB_MAP = str.maketrans({"0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉","+":"₊","-":"₋","=":"₌","(":"₍",")":"₎"})
SUP_TO_PLAIN = str.maketrans({v:k for k,v in SUP_MAP.items()})
SUB_TO_PLAIN = str.maketrans({v:k for k,v in SUB_MAP.items()})

# Common chemistry/units patterns -> sub/sup conversions
_ELEM_RE = r"(?:[A-Z][a-z]?)"                # H, He, Al, ...
_FORMULA_NUM = re.compile(rf"({_ELEM_RE}|\))(\d+)")  # e.g., Al2, O3, )3
_SUP_POW = re.compile(r"(\^)(\(?[0-9nmkxy+\-]+\)?)")  # ^2, ^(n+1), ^-1
_CHARGE_1 = re.compile(rf"({_ELEM_RE}\d*)([+-])")     # Cl-, Fe2+
_CHARGE_2 = re.compile(rf"({_ELEM_RE}\d*)(\^\d+[+-])")# SO4^2-
_UNIT_POW = re.compile(r"\b([A-Za-zμ]+)\^([0-9+\-]+)\b")  # m^2, s^-1, μm^2
_ISOTOPE = re.compile(r"\b([0-9]{1,3})-([A-Z][a-z]?)\b")  # 14-C -> ¹⁴C

def _to_sub(s: str) -> str:
    return s.translate(SUB_MAP)

def _to_sup(s: str) -> str:
    return s.translate(SUP_MAP)

def _preserve_super_sub(text: str) -> str:
    """Heuristically rewrite common patterns to Unicode super/sub scripts."""
    if not text:
        return text

    # Element counts as subscripts: Al2O3 -> Al₂O₃
    text = _FORMULA_NUM.sub(lambda m: m.group(1) + _to_sub(m.group(2)), text)

    # Unit exponents: m^2 -> m² ; s^-1 -> s⁻¹
    text = _UNIT_POW.sub(lambda m: f"{m.group(1)}{_to_sup(m.group(2))}", text)

    # Caret powers anywhere: 10^6 -> 10⁶ ; x^(n+1) -> x⁽ⁿ⁺¹⁾
    text = _SUP_POW.sub(lambda m: _to_sup(m.group(2)), text).replace("^", "")

    # Charges: Cl- -> Cl⁻ ; Fe2+ -> Fe₂⁺
    text = _CHARGE_2.sub(lambda m: m.group(1) + _to_sup(m.group(2)[1:]), text)
    text = _CHARGE_1.sub(lambda m: m.group(1) + _to_sup(m.group(2)), text)

    # Isotopes: 14-C -> ¹⁴C
    text = _ISOTOPE.sub(lambda m: f"{_to_sup(m.group(1))}{m.group(2)}", text)

    return text


def _sparsify_text(block: str) -> str:
    """
    Make text 'sparse': collapse excessive spaces, preserve single newlines,
    strip trailing spaces. Then apply super/sub preservation.
    """
    if not block:
        return ""
    # Normalize whitespace but keep paragraph structure
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in block.splitlines()]
    lines = [ln for ln in lines if ln]  # drop fully blank lines
    txt = "\n".join(lines)
    return _preserve_super_sub(txt)


def _normalize_pdf_glyphs(s: str) -> str:
    # Replace middle dots and rare decimal points with "."
    s = s.replace("·", ".").replace("⋅", ".").replace("․", ".")  # U+2024 one dot leader
    # Normalize non-breaking/thin spaces
    for ch in ("\u00A0","\u2009","\u200A","\u2002","\u2003","\u202F","\u200B","\u200C","\u200D"):
        s = s.replace(ch, " ")
    return s


def _sparsify_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """Trim cells, drop all-empty rows/cols, reset index/columns."""
    if df is None:
        return None
    # Convert to strings and strip
    df = df.map(lambda x: (str(x).strip() if x is not None else ""))
    # Normalize PDF glyph artifacts in every cell
    df = df.map(_normalize_pdf_glyphs)
    # Apply your super/sub preservation on cells too
    df = df.map(_preserve_super_sub)

    # Drop all-empty rows/cols
    df = df.loc[~(df.eq("").all(axis=1))]
    df = df.loc[:, ~(df.eq("").all(axis=0))]
    # Optionally de-duplicate header-like first row if it's identical to columns
    return df.reset_index(drop=True)


def _df_to_sparse_string(df: pd.DataFrame, join_chemistry: bool = True, to_plain_digits: bool = False) -> str:
    """
    Return a compact TSV-like string for the table.
    - Normalizes glyphs and applies super/sub per cell
    - Optionally joins element-number fragments within rows
    - Optionally converts superscripts/subscripts back to plain digits/signs
    """
    if df is None or df.empty:
        return ""
    # Normalize per cell first
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
        if to_plain_digits:
            line = line.translate(SUP_TO_PLAIN).translate(SUB_TO_PLAIN)
        out_lines.append(line)
    return "\n".join(out_lines)


def _extract_tables_with_camelot(pdf_path: str) -> List[Dict[str, Any]]:
    out = []
    if not _HAVE_CAMELOT:
        return out

    def _run(flavor: str):
        try:
            return camelot.read_pdf(pdf_path, pages="all", flavor=flavor, suppress_stdout=True)
        except Exception:
            return None

    # Try lattice then stream
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
                break  # prefer first successful flavor
    return out


def _extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
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


def extract_sparse_tables_and_context(pdf_file) -> Tuple[List[str], List[str]]:   # need 
    """
    Walk a directory of PDF files and return:
      contexts: list[str]  -> per-page sparsified text blocks with super/sub preserved
      tables:   list[str]  -> sparsified tables as TSV-like strings (page-prefixed)

    Each list entry is prefixed with 'filename::pN::' to preserve provenance.
    """
    contexts: List[str] = []
    tables: List[str] = []

    # if not os.path.isdir(pdf_dir):
    #     raise NotADirectoryError(f"Not a directory: {pdf_dir}")

    # pdf_files = sorted(
    #     [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    # )

    pdf_path = pdf_file

    #for pdf_path in pdf_files:
    fname = os.path.basename(pdf_path)
    # --- Extract text ---
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    txt = page.extract_text(x_tolerance=1.5, y_tolerance=3.0) or ""
                except Exception:
                    txt = ""
                sparse = _sparsify_text(txt)
                if sparse:
                    contexts.append(f"{fname}::p{i}::\n{sparse}")
    except Exception:
            # Skip unreadable PDFs but keep going
            print(f"❌ This PDF has error")

    # --- Extract tables (Camelot first, then fallback) ---
    extracted = _extract_tables_with_camelot(pdf_path)
    if not extracted:
        extracted = _extract_tables_with_pdfplumber(pdf_path)

    for t in extracted:
        if t.get("data_tsv"):
            tables.append(f"{fname}::p{t['page']}::{t['flavor']}::\n{t['data_tsv']}")

    with open(output_sparse_txt + name + "_sparse_output.txt", "w", encoding="utf-8") as f:
        json.dump(contexts + tables, f, ensure_ascii=False, indent=2) 


    return contexts + tables




# ---------- Helpers ----------
def clean_txt_output_df(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()  # data is a dict or list
       
    # Split by newline characters into a list of lines
    lines = str(data).split("\\n")
    # Optional: remove empty lines and strip trailing carriage returns
    lines = [ln.rstrip("\r") for ln in lines if ln.strip() != ""]
    
    # 1) Join lines into one text blob
    text = "\n".join(lines)
    
    # 2) HTML-decode entities (&quot; -> ", &#39; -> ')
    text = html.unescape(text)
    
    # 3) Strip leading/trailing quotes if present
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    
    # 4) Remove ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE|re.MULTILINE).strip()
    
    # 5) The payload still has backslash-escaped quotes from \&quot; -> " and sometimes \" -> "
    #    Normalize sequences
    text = text.replace(r"\&quot;", '"')  # \&quot; -> "
    text = text.replace(r"\\\"", r"\"")  # collapse double-escaping
    text = text.replace(r"\"", '"')      # \" -> "
    text = text.replace(r"\/", "/")      # optional: unescape slashes
    
    # 6) Now parse as JSON
    data = json.loads(text)  # this should be a list[dict]
    
    # Sanity check
    assert isinstance(data, list) and all(isinstance(x, dict) for x in data), "Expected list of dicts"

    dffs = []
    for diction in data:
        values = [i for i in diction.values()]
        keys = [i for i in diction.keys()]
        blank = pd.DataFrame(values)
        data_value = blank.T
        data_value.columns = keys
        dffs.append(data_value)
    data_clean = pd.concat(dffs, axis = 0)
    return data_clean




#---------------Gemini data extraction---------------------

def gemini_extract_information_csv(pdf_file):
    
    start = time.perf_counter()
    paragraph_info = extract_sparse_tables_and_context(pdf_file)
    print(f"✅ Sparsing data complete")

    # Combine the list of strings into a single paragraph
    paragraph = "\n".join(paragraph_info)


    # 2) Prompt
    prompt = f"""
    Your task is to extract the crystal strucutre, toughness (KIC) with toughness unit (MPa∙m1/2), thermal_expansion_coefficient_/K and thermal_Conductivity_in_W/mK record the temperature measured for each properties
    For each distinct material and its corresponding toughness value found, \n
    -1. Material_Name (e.g., Silicon, Graphene, Al2O3, specific alloy composition if mentioned) - (use full names, not abbreviations). 
    -2  Chemical formula if available. \n
    -3. for temperature describe_in_Kelvin (Convert Celsius or Fahrenheit to Kelvin. Use '300' for 'room temperature' or 'RT'. If not specified, use 'Not available'.) \n
    -4. Toughness_MPam0.5 (Report the numerical value. If a range is given like '2~3', report '2-3'. If approximate like '~2', report '~2'. Do not include units in this field.) and measured temperature \n
    -5. subscript or superscipt numbers in chemical formula like ('₂', '³', '₂₃') change them to normal numbers like ('2', '3', '23') \n
    -6  Record the crystal structure/strucutre like (pyrochlore, flourite ...)  \n
    -7 Record the thermal_expansion_coefficient_/K (CTE) in /K unit is there is any and measured temperature  \n
    -8 Record Thermal_Conductivity_in_W/mK in in_W/mK unit and measured temperature  \n
    -7. Change fraction in formula like ("1/2", "2/3", "1/3" ... etc) to float (0.5, 0.66, 0.33) \n
    -8. Record the DOI of this paper or coresponding literature  \n



    "IMPORTANT INSTRUCTIONS:\n
                    - If multiple data points are found in the chunk, list each on a new line.\n
                    - Convert temperatures: Celsius to Kelvin (K = C + 273.15), Fahrenheit to Kelvin (K = (F - 32) * 5/9 + 273.15).\n
                    - Room temperature (RT) should be 300 K.\n"
                    - If no temperature report it is room temperature 300K  \n

    Return ONLY a JSON array (no prose, no markdown fences). Each entry:
    {{
      "Material_Name": "X",
      "Chemical_formula:" "C"
      "Toughness_MPam0.5": "Y",
      "Toughness_measured_temperature_K": "Z" 
      "Thermal_expansion_coefficient_/K": "D"
      "Thermal_expansion_coefficient_measured_temperature_K": "E"
      "Thermal_Conductivity_W_mK": "F"
      "Thermal_Conductivity_measured_temperature_K": "H"
      "Strucutre": "A"
      "DOI": "B"
    }}


    Text: {paragraph}

    """

    # 3) Call Gemini with JSON-forcing config (supported in current SDKs)
    #    Falls back gracefully if your SDK version ignores this field.
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt,
    )

    raw = getattr(response, "text", "") or ""
    print("🔎 Raw model output (first 500 chars):\n", raw[:500], "...\n")

    # save the raw file in json
    with open(output_txt + name + "_gemini_snippet.txt", "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2) 
    
    print('txt infoirmation saved at:', output_txt + name + "_gemini_snippet.txt")

    # save the csv file
    dataframe = clean_txt_output_df(output_txt + name + "_gemini_snippet.txt")
    dataframe.to_csv(output_csv_final + name + '_literature_digest.csv', index  = False) 
    
    end = time.perf_counter()
    print(f"Total time: {end - start:.6f} seconds")

    return dataframe


if __name__ == "__main__":
    name = 'toughness_33'
    pdf_name = name + '.pdf'
    pdf = pdf_folder + pdf_name
    print(gemini_extract_information_csv(pdf))
    #print(extract_sparse_tables_and_context(pdf))