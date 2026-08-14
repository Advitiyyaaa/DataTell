"""
migrate_to_chroma_cloud.py — Phase 8: One-shot migration of local ChromaDB → Chroma Cloud
===========================================================================================

Run this ONCE after setting up your Chroma Cloud account.

Usage:
    # Verify how many chunks are in the local index (no cloud writes)
    python scripts/migrate_to_chroma_cloud.py --verify

    # Full migration: push all embedded chunks to Chroma Cloud
    python scripts/migrate_to_chroma_cloud.py

Prerequisites:
    pip install chromadb
    CHROMA_TENANT=your-tenant-id   in .env
    CHROMA_DATABASE=datatell        in .env
    CHROMA_API_KEY=your-api-key    in .env

Sign up at https://trychroma.com — you get $5 free credits (more than enough for 38 static chunks).

How it works:
    1. Reads chunks + embeddings from the local chroma_db/ PersistentClient
    2. Connects to Chroma Cloud via CloudClient
    3. Creates (or gets) the collection with the same cosine-space config
    4. Pushes all 38 document chunks including their pre-computed embeddings
       (no re-embedding needed — embeddings are already stored locally)
    5. Verifies chunk counts match

Why no re-embedding?
    ChromaDB stores the raw embeddings alongside documents. We extract them
    directly from the local collection (get() with include=['embeddings']).
    This saves Gemini/sentence-transformer API calls.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

root_dir = Path(__file__).parent.parent
load_dotenv(dotenv_path=root_dir / ".env")

import os
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "datatell")
CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY")

LOCAL_CHROMA_DIR     = root_dir / "chroma_db"
COLLECTION_NAME      = "datatell_docs"
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"


def verify_local() -> None:
    """Print local collection stats without connecting to cloud."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if not LOCAL_CHROMA_DIR.exists():
        sys.exit(f"ERROR: local chroma_db/ not found at {LOCAL_CHROMA_DIR}\n"
                 "Run rag_pipeline.py once first to build the local index.")

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    local_client = chromadb.PersistentClient(path=str(LOCAL_CHROMA_DIR))
    collection = local_client.get_collection(
        name=COLLECTION_NAME, embedding_function=ef
    )
    count = collection.count()
    print(f"Local ChromaDB collection '{COLLECTION_NAME}': {count} chunks")
    print("\nSample chunk IDs:")
    sample = collection.get(limit=5)
    for cid in sample["ids"]:
        print(f"  {cid}")


def migrate() -> None:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    # ── 1. Validate env ────────────────────────────────────────────────────────
    if not all([CHROMA_TENANT, CHROMA_DATABASE, CHROMA_API_KEY]):
        sys.exit(
            "ERROR: CHROMA_TENANT, CHROMA_DATABASE, and CHROMA_API_KEY must be set in .env\n"
            "Sign up at https://trychroma.com"
        )

    # ── 2. Open local collection ───────────────────────────────────────────────
    if not LOCAL_CHROMA_DIR.exists():
        sys.exit(f"ERROR: local chroma_db/ not found at {LOCAL_CHROMA_DIR}")

    print("Reading local ChromaDB collection...")
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    local_client = chromadb.PersistentClient(path=str(LOCAL_CHROMA_DIR))
    local_col = local_client.get_collection(
        name=COLLECTION_NAME, embedding_function=ef
    )
    local_count = local_col.count()
    print(f"  Found {local_count} chunks locally")

    # Fetch all chunks WITH embeddings (avoids re-embedding)
    all_data = local_col.get(
        limit=local_count,
        include=["embeddings", "documents", "metadatas"],
    )

    ids        = all_data["ids"]
    embeddings = all_data["embeddings"]
    documents  = all_data["documents"]
    metadatas  = all_data["metadatas"]

    print(f"  Retrieved {len(ids)} chunks with pre-computed embeddings\n")

    # ── 3. Connect to Chroma Cloud ─────────────────────────────────────────────
    print(f"Connecting to Chroma Cloud (tenant={CHROMA_TENANT}, db={CHROMA_DATABASE})...")
    cloud_client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY,
    )
    print("Connected.\n")

    # ── 4. Create (or get) the cloud collection ────────────────────────────────
    cloud_col = cloud_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    cloud_existing = cloud_col.count()
    print(f"Cloud collection '{COLLECTION_NAME}': {cloud_existing} existing chunks")

    if cloud_existing == local_count:
        print("✅ Cloud collection already has the correct number of chunks — nothing to do.")
        return

    if cloud_existing > 0:
        print(f"Deleting {cloud_existing} stale chunks before re-upload...")
        cloud_client.delete_collection(COLLECTION_NAME)
        cloud_col = cloud_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── 5. Upload in batches ───────────────────────────────────────────────────
    print(f"Uploading {local_count} chunks to Chroma Cloud...")
    BATCH = 50
    for start in range(0, len(ids), BATCH):
        end = min(start + BATCH, len(ids))
        cloud_col.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Uploaded {end}/{len(ids)}")

    # ── 6. Verify ──────────────────────────────────────────────────────────────
    cloud_count = cloud_col.count()
    if cloud_count == local_count:
        print(f"\n[OK] Migration complete - {cloud_count} chunks in Chroma Cloud!")
    else:
        print(f"\n[WARN] Count mismatch: local={local_count}, cloud={cloud_count}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate local ChromaDB → Chroma Cloud")
    parser.add_argument("--verify", action="store_true", help="Show local stats only, no cloud writes")
    args = parser.parse_args()

    if args.verify:
        verify_local()
    else:
        migrate()
