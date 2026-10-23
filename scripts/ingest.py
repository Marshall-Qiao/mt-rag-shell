"""Explicit document import. Invoking an import calls the configured embedding API."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing your own documents")
    parser.add_argument("--format", choices=("pdf", "txt"), default="pdf")
    parser.add_argument("--collection", choices=("default", "reference"), default="default")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("Source must be an existing directory")
    pattern = f"*.{args.format}"
    if not any(args.source.rglob(pattern)):
        parser.error(f"Source contains no {pattern} files; nothing was imported")

    from LongTermMemory.store_manager import StoreManager
    from module.settings import get_settings

    settings = get_settings()
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
        parser.error("OPENAI_API_KEY is required for document embeddings")
    manager = StoreManager(
        persist_directory=settings.vector_db_path,
        collection_name=getattr(settings, f"{args.collection}_collection"),
        embedding_model=settings.embedding_model,
    )
    manager.add_files(str(args.source), file_pattern=pattern, category=args.collection)


if __name__ == "__main__":
    main()
