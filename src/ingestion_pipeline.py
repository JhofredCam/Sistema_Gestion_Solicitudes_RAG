"""Ingestion pipeline for loading documents, chunking text, and persisting embeddings.

This script reads `.txt` files from a documents directory, splits them into chunks,
generates embeddings with Google Generative AI, and stores them in a local Chroma
vector database for retrieval use cases.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # for chunking
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Loads environment variables (e.g., GOOGLE_API_KEY) from a local .env file.
load_dotenv()

def load_documents(docs_path="docs", glob_pattern="*.html", loader_cls=BSHTMLLoader):
    """Load documents from a directory using a glob pattern.

    Args:
        docs_path: Directory that contains source documents.
        glob_pattern: Pattern used to select files (e.g., ``"*.txt"``).
        loader_cls: LangChain document loader class used for each matched file.

    Returns:
        list: Loaded LangChain Document objects.

    Raises:
        FileNotFoundError: If ``docs_path`` does not exist or no matching files are found.
    """
    
    # check if path exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    # DirectoryLoader applies `loader_cls` to each file matching `glob_pattern`.
    loader = DirectoryLoader(
        path=docs_path,
        glob=glob_pattern,
        loader_cls=loader_cls,
        loader_kwargs={"open_encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No {glob_pattern} files found in {docs_path}")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"\tSource: {doc.metadata['source']}")
        print(f"\tLength: {len(doc.page_content)} characters")
        print(f"\tContent preview: {doc.page_content[:100]}...")
        print(f"\tMetadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=512, chunk_overlap=0):
    """Split loaded documents into chunks for embedding.

    Args:
        documents: Sequence of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        splitter: Text splitter class to instantiate.
        separator: Preferred split separator (e.g., ``" "`` to split on spaces).

    Returns:
        list: Chunked LangChain Document objects.
    """
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')

    # Smaller chunks can improve retrieval precision but may increase index size.
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separator=separator
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"\tSource: {chunk.metadata['source']}")
        print(f"\tLength: {len(chunk.page_content)} characters")
        print(f"\tContent: \n{chunk.page_content}")
        print(f"\tMetadata: {chunk.metadata}")
        print("-" * 50)
    print(f"\nTotal chunks: {len(chunks)}\n")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist a local Chroma vector store from document chunks.

    Args:
        chunks: Chunked LangChain Document objects to index.
        persist_directory: Local path where Chroma persists its files.

    Returns:
        Chroma: Persisted vector store instance.
    """

    # Uses HF embedding model.
    embedding_model = HuggingFaceEmbeddings(model="intfloat/multilingual-e5-small")

    # Build and persist a local vector index on disk.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        # cosine similarity for comparing vectors
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore

def main():
    """Run the end-to-end ingestion workflow."""
    # 1. Load files
    documents = load_documents()

    # 2. Chunking files
    chunks = split_documents(documents,
                            chunk_size=256,
                            chunk_overlap=48,
                            )    
    # 3. Embedding and storing in DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()