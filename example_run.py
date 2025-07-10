import os
import pandas as pd
from chemdataextractor import Document
from chemdataextractor.model.units.thermoelectric_models import ThermCond
from chemdataextractor.split_and_stitch.new_parsing import te_parse

# 📂 Define folder containing .txt files
folder = os.path.join(os.path.dirname(__file__), "parsed_texts")
chosen_model = ThermCond

# 📊 Initialize output
df1 = pd.DataFrame()
records_holder = []

print("📂 Scanning folder:", folder)
print("------------------------------------------------------------")

# 🔄 Process all .txt files
for article in sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt")):
    article_path = os.path.join(folder, article)
    print(f"\n🔍 Processing file: {article}")

    try:
        with open(article_path, 'r', encoding='utf-8') as file:
            text = file.read()
        d = Document(text)
        d.models = [chosen_model]  # use thermal conductivity model
        print("✅ Successfully parsed text.")
    except Exception as e:
        print(f"❌ Failed to parse {article}: {e}")
        continue

    # Extract title
    try:
        title = d.titles[0].text
        print(f"📄 Title extracted: {title}")
    except (IndexError, UnicodeError):
        title = "title_fail"
        print("⚠️  Could not extract title.")

    # Extract records
    try:
        raw_records = [r for r in d.records if isinstance(r, chosen_model)]
        print(f"📈 Raw thermal conductivity records found: {len(raw_records)}")
    except Exception as e:
        print(f"⚠️  Failed to collect records: {e}")
        continue

    # Attempt structured parsing with te_parse
    try:
        df1, records_holder = te_parse(
            d, df1, records_holder, filename=article, title=title, given_model=chosen_model
        )
        print(f"📥 te_parse successful: Total structured records so far: {len(df1)}")
    except Exception as e:
        print(f"⚠️  te_parse failed on {article}: {e}")

# 💾 Save structured data to CSV
output_csv = "thermal_conductivity_extracted.csv"
df1.to_csv(output_csv, index=False)
print("\n💾 All done. Saved structured records to:", output_csv)
print(f"✅ Final structured record count: {len(df1)}")
