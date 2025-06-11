from extract.pdf_reader import extract_text_from_pdf
from extract.openai_extractor import extract_items_from_text
from embed.create_product_list_embeddings import create_vector_db, query_vector_db
import sys
import json
import os
import argparse

PRODUCT_LIST_PATH = "data/product_list.csv"
PDF_FOLDER_PATH = "data/POs"

def create_product_list_embeddings():
    print("[0] Creating product list embeddings...")
    create_vector_db(csv_path=PRODUCT_LIST_PATH)

def process_pdfs_in_folder():
    print(f"[1] Scanning folder for PDFs: {PDF_FOLDER_PATH}")
    if not os.path.isdir(PDF_FOLDER_PATH):
        print(f"Error: {PDF_FOLDER_PATH} is not a valid directory.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the directory.")
        sys.exit(0)

    for pdf_file in pdf_files:
        full_path = os.path.join(PDF_FOLDER_PATH, pdf_file)
        print(f"\n--- Processing: {pdf_file} ---")

        print("[2] Extracting text from PDF...")
        text = extract_text_from_pdf(full_path)

        print("[3] Sending to structured extraction...")
        result = extract_items_from_text(text)

        print("[4] Extracted JSON:")
        print(json.dumps(result, indent=2))

        print("[5] Enhancing with product matches...")
        for item in result:
            req_item = item.get("Request Item", "").strip()
            if req_item:
                try:
                    match = query_vector_db(req_item, top_k=1)
                    item["product_match"] = match[0] if match else None
                except Exception as e:
                    print(f"Failed to match '{req_item}': {e}")
                    item["product_match"] = None

        print("[5] Final JSON with product matches:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF item extractor tool")
    parser.add_argument("--embed-product-list", action="store_true", help="Create product list embeddings")
    parser.add_argument("--process-pdfs", action="store_true", help="Process all PDFs in the input folder")

    args = parser.parse_args()

    if args.embed_product_list:
        create_product_list_embeddings()

    if args.process_pdfs:
        process_pdfs_in_folder()

    if not args.embed_product_list and not args.process_pdfs:
        print("Usage:")
        print("  --embed-product-list     Create product list embeddings")
        print("  --process-pdfs           Process all PDFs in the folder")
        sys.exit(1)