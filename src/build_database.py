import os
import json
import shutil
import time
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import dissect_pdf, PDF_PATH, PROJECT_ROOT

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY not found in environment.")
    GOOGLE_API_KEY = input("Please paste your API Key here: ").strip()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

DB_PATH = os.path.join(PROJECT_ROOT, "data", "vector_db")
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_summaries.json")


def build_database():
    print("STARTING: Building RAG Database...")

    text_data, image_metadata_list = dissect_pdf(PDF_PATH)
    image_summaries = []
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            image_summaries = json.load(f)
    print(f"Loaded {len(image_summaries)} image summaries.")

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print("\n--- PHASE 3: Processing Text ---")
    for item in text_data:
        splits = text_splitter.split_text(item["text"])

        # Eeumerate chunk ID
        for idx, split in enumerate(splits):
            chunk_metadata = item["metadata"].copy()
            chunk_metadata["chunk_id"] = idx

            doc = Document(
                page_content=split,
                metadata=chunk_metadata
            )
            documents.append(doc)

    print("\n--- PHASE 4: Processing Images ---")
    page_lookup = {img['path']: img['page'] for img in image_metadata_list}

    for img in image_summaries:
        path = img['image_path']
        desc = img['description']
        page_num = page_lookup.get(path, 0)

        doc = Document(
            page_content=f"Item: Figure/Chart\nPage: {page_num}\nDescription: {desc}",
            metadata={
                "source": path,
                "page": page_num,
                "type": "image",
                "chunk_id": 0
            }
        )
        documents.append(doc)

    #I had issues with duplication so just deleting old vector database
    print(f"\n--- PHASE 5: Creating Vector Database ---")
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("Old database deleted.")
            time.sleep(1)
        except PermissionError:
            print("ERROR: Could not delete database. Stop Streamlit and try again.")
            return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"DONE. Database saved at {DB_PATH}")
    print(f"   - Total Documents: {len(documents)}")


if __name__ == "__main__":
    build_database()