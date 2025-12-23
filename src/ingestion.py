import fitz  # PyMuPDF
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "pages")
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ericsson_sample.pdf")

def dissect_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)
    print(f"Dissecting '{source_name}' ({len(doc)} pages)...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text_data = []
    image_metadata = []

    for i, page in enumerate(doc):
        page_num = i + 1

        # 1 EXTRACT TEXT & METADATA
        text = page.get_text()
        text_data.append({
            "text": text,
            "metadata": {
                "source": source_name,
                "page": page_num,
                "type": "text"
            }
        })

        # 2 SAVE IMAGE
        filename = f"page-{page_num:03d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(save_path):
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            pix.save(save_path)

        image_metadata.append({"page": page_num, "path": save_path})

    print(f"Extracted data from {len(text_data)} pages.")
    return text_data, image_metadata

if __name__ == "__main__":
    dissect_pdf(PDF_PATH)