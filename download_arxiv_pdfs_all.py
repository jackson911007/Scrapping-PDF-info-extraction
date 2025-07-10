import arxiv
import os
import time

# Output folder
output_dir = "arxiv_pdfs_oxides_structures"
os.makedirs(output_dir, exist_ok=True)

# Set up arXiv client
client = arxiv.Client(
    page_size=100,
    delay_seconds=3.0,
    num_retries=5
)

# Updated search query including oxide
search = arxiv.Search(
    query='(pyrochlore OR scheelite OR zircon OR fluorite OR garnet OR oxide) AND "thermal conductivity"',
    max_results=1000,
    sort_by=arxiv.SortCriterion.Relevance
)

pdf_paths = []
total_downloaded = 0

for result in client.results(search):
    safe_id = result.get_short_id().replace("/", "_")
    pdf_path = os.path.join(output_dir, f"{safe_id}.pdf")

    if not os.path.exists(pdf_path):
        try:
            result.download_pdf(filename=pdf_path)
            pdf_paths.append(pdf_path)
            total_downloaded += 1
            print(f"✅ Downloaded: {result.title.strip()}")
        except Exception as e:
            print(f"❌ Failed to download {safe_id}: {e}")
    else:
        print(f"⚠️ Already exists: {safe_id}")

print("\n📦 Total downloaded papers:", total_downloaded)
