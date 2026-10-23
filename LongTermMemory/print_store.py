from datetime import datetime
from prettytable import PrettyTable
from pprint import pprint

def print_sources(store_sources):
    table = PrettyTable()
    table.field_names = ["Source", "Embeddings", "Creation Time", "Created At"]
    sources = [metadata.get('source') for metadata in store_sources["metadatas"]]

    source_to_info = {
        metadata['source']: {
            'created_at': metadata['created_at'],
            'count': 0 
        }
        for metadata in store_sources["metadatas"]
    }

    for source in sources:
        if source in source_to_info:
            source_to_info[source]['count'] += 1

    for source, info in source_to_info.items():
        created_at = datetime.fromtimestamp(info['created_at']).strftime('%Y-%m-%d %H:%M:%S')
        table.add_row([source, info['count'], created_at, info['created_at']])

    pprint(table)


def print_embeddings(store_docs):
    table = PrettyTable()
    table.field_names = ["Document ID", "Source", "Creation Time", "Document Content", "Embedding"]
    for doc_id, metadata, document, embedding in zip(store_docs["ids"], store_docs["metadatas"], store_docs["documents"], store_docs["embeddings"]):
        source = metadata.get("source", "N/A")
        created_at = metadata.get("created_at", "N/A")
        document_preview = document[:30].replace('\n', ' ')
        table.add_row([doc_id, source, created_at, document_preview, embedding[:2]])
    pprint(table)