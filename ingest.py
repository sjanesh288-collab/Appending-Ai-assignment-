"""Ingestion pipeline: PDF -> chunks -> OpenAI embeddings -> Pinecone.

Usage:
    python ingest.py --pdf data/Ebook-Agentic-AI.pdf
    python ingest.py --url <direct-pdf-url>          # downloads first
    python ingest.py --pdf data/Ebook-Agentic-AI.pdf --reset   # wipe index first
"""
import argparse
import hashlib
import os
import time
from typing import List

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

import config

BATCH_SIZE = 100


def download_pdf(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


def load_and_chunk(pdf_path: str) -> List[Document]:
    pages = PyPDFLoader(pdf_path).load()  # one Document per page, metadata["page"] is 0-based
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]  # drop near-empty chunks
    for c in chunks:
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1  # human-friendly 1-based page
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")
    return chunks


def get_index(pc: Pinecone):
    name = config.PINECONE_INDEX
    if name not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{name}' ...")
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    return pc.Index(name)


def ingest(pdf_path: str, reset: bool = False) -> None:
    config.require_keys()
    chunks = load_and_chunk(pdf_path)

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = get_index(pc)
    if reset:
        try:
            index.delete(delete_all=True)
            print("Cleared existing vectors.")
        except Exception as exc:  # empty namespace can raise on some versions
            print(f"Reset skipped: {exc}")

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    source = os.path.basename(pdf_path)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c.page_content for c in batch]
        vectors = embeddings.embed_documents(texts)
        records = []
        for c, vec in zip(batch, vectors):
            uid = hashlib.md5(f"{source}:{c.metadata['page']}:{c.page_content}".encode()).hexdigest()
            records.append({
                "id": uid,
                "values": vec,
                "metadata": {"text": c.page_content, "page": c.metadata["page"], "source": source},
            })
        index.upsert(vectors=records)
        print(f"Upserted {start + len(batch)}/{len(chunks)}")

    print("Done. Index stats:", index.describe_index_stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone")
    parser.add_argument("--pdf", default=config.PDF_PATH, help="Local path to the PDF")
    parser.add_argument("--url", help="Direct URL to download the PDF from")
    parser.add_argument("--reset", action="store_true", help="Delete existing vectors first")
    args = parser.parse_args()

    path = download_pdf(args.url, args.pdf) if args.url else args.pdf
    if not os.path.exists(path):
        raise SystemExit(f"PDF not found at {path}. Put it in data/ or pass --url.")
    ingest(path, reset=args.reset)
