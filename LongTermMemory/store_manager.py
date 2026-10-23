# https://github.com/langchain-ai/langchain/blob/e806e9de38e796d3c7d2ed069d59186717bef60e/libs/partners/chroma/langchain_chroma/vectorstores.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from LongTermMemory.metadata import update_docs_metadata, DocCategory
from LongTermMemory.files import find_non_added_files
from LongTermMemory.file_loaders import pdf_loader, text_loader
from chromadb.config import Settings

DEFAULT_STORE_PARAMS = {
    "persist_directory": "./data/vector-store",
    "collection_name": "rag-default",
    "embedding_model": "text-embedding-3-large"
}

REFERENCE_STORE_PARAMS = {
    "persist_directory": "./data/vector-store",
    "collection_name": "rag-reference",
    "embedding_model": "text-embedding-3-large"
}

TEST_DATA_FOLDERS = {
    "default": "./data/documents/default",
    "reference": "./data/documents/reference"
}

class StoreManager:
    def __init__(self, persist_directory=DEFAULT_STORE_PARAMS["persist_directory"], collection_name=DEFAULT_STORE_PARAMS["collection_name"], embedding_model=DEFAULT_STORE_PARAMS["embedding_model"]):
        from module.settings import get_settings
        self.embedding_model = (
            OpenAIEmbeddings(model=embedding_model, api_key=get_settings().openai_api_key)
            if isinstance(embedding_model, str) else embedding_model
        )
        self.collection_name = collection_name
        self.store = Chroma(collection_name=collection_name, embedding_function=self.embedding_model,persist_directory=persist_directory, client_settings=Settings(anonymized_telemetry=False))
        self.retriever = self.store.as_retriever()
        
    def _add_docs_to_vector_store(self, docs):
        print("Total Pages in the original document: ", len(docs))
        print("Splitting documents...")
        # split documents
        # https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
        text_splitter = SemanticChunker(self.embedding_model, sentence_split_regex=r"(?<=[。！？¡¿\.\?\!；;])\s*(?=[\w\d])")
        splittedDocs = text_splitter.split_documents(docs)
        print("Adding documents to vector store, chunks: ", len(splittedDocs))
        self.store.add_documents(splittedDocs)
        print("Documents added to vector store")
    
    def add_files(self, folder_path=TEST_DATA_FOLDERS["default"], file_pattern="*.pdf", category=DocCategory.DEFAULT.value):
        """
        Add new files to the vector store from the given folder path.
        Only add files that are not already in the vector store.
        """
        # Find new files to add
        if file_pattern not in {"*.pdf", "*.txt"}:
            raise ValueError("Only PDF and text files are supported")
        files_to_add = find_non_added_files(self.store, folder_path, file_pattern=file_pattern)['new_sources']
        if (len(files_to_add) == 0):
            print("No new files to add")
            return
        
        # Load and filter documents
        originalDocs = pdf_loader(folder_path) if file_pattern == "*.pdf" else text_loader(folder_path)
        update_docs_metadata(originalDocs, category=category, source_root=folder_path)
        filteredDocs = [doc for doc in originalDocs if (doc.metadata['source'], doc.metadata['created_at'] ) in files_to_add]
        self._add_docs_to_vector_store(filteredDocs)

    def temp_add_reference_files(self, folder_path=TEST_DATA_FOLDERS["reference"]):
        return self.add_files(folder_path, file_pattern="*.txt", category=DocCategory.REFERENCE.value)

    def list_docs(self, source):
        """
        Return all documents from the given source in the vector store.
        """
        return self.store.get(
            where={"source": source},
            include=["metadatas", "documents", "embeddings"] 
        )
    

    def list_sources(self):
        """
        Return all source files in the vector store.
        """
        return self.store.get(
            include=["metadatas"] 
        )
        
    def delete_source(self, source):
        """
        Delete all documents from the given source in the vector store.
        """
        ids = self.store.get(where={"source": source})["ids"]
        if len(ids) == 0:
            print(f"No documents found in source {source}")
            return
        print(f"Deleting {len(ids)} documents from source {source}")
        self.store.delete(ids)
