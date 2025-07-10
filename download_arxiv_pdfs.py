import arxiv
import os
import time

# Output directory
output_dir = "arxiv_pdfs_pyrochlore_all"
os.makedirs(output_dir, exist_ok=True)

def is_relevant(text):
    text = text.lower()
    return "pyrochlore" in text and "thermal conductivity" in text

# Parameters
max_results_per_page = 1000
total_downloaded = 0
start_index = 0
pdf_paths = []

while True:
    results = arxiv.query(
        query="pyrochlore AND thermal conductivity",
        max_results=max_results_per_page,
        start=start_index,
        sort_by="relevance"
    )

    print(f"\n🔍 Fetched {len(results)} results from index {start_index}")
    if not results:
        break

    for result in results:
        title = result['title'].strip()
        abstract = result['summary'].strip()
        combined_text = f"{title}. {abstract}"

        if is_relevant(combined_text):
            arxiv_id = result['id'].split('/')[-1]
            safe_id = arxiv_id.replace("/", "_")
            pdf_path = os.path.join(output_dir, f"{safe_id}.pdf")

            if not os.path.exists(pdf_path):
                try:
                    arxiv.download(result, dirpath=output_dir, slugify=lambda x: safe_id)
                    pdf_paths.append(pdf_path)
                    total_downloaded += 1
                    print(f"✅ Downloaded: {title}")
                    time.sleep(1.5)
                except Exception as e:
                    print(f"❌ Failed to download {safe_id}: {e}")
            else:
                print(f"⚠️ Already exists: {safe_id}")

    if len(results) < max_results_per_page:
        break

    start_index += max_results_per_page
    time.sleep(3)

print("\n📦 Total relevant pyrochlore papers downloaded:", total_downloaded)
