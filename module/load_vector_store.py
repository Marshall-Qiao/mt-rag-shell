import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from module.settings import get_settings

def load_vector_store(vectorDBPath, embeddings):
    isVectorDBExists = os.path.exists(vectorDBPath)
    vector_store = Chroma(
        collection_name=get_settings().default_collection,
        embedding_function=embeddings,
        persist_directory=vectorDBPath,  # Where to save data locally, remove if not necessary
        # other params...
    )

    if isVectorDBExists:
        print("Using existing vector store")
    else:
        loader = PyPDFDirectoryLoader(
            "./data/documents/default",
            extract_images=False,
            glob="*.pdf",
        )
        originalDocs = loader.load()

        # split documents
        # https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
        text_splitter = SemanticChunker(embeddings)
        splittedDocs = text_splitter.split_documents(originalDocs)
        print("Pages in the original document: ", len(originalDocs))
        print("Length of chunks after splitting pages: ", len(splittedDocs))
        print("Adding documents to vector store")

        vector_store.add_documents(splittedDocs)
        print("Documents added to vector store")
    return vector_store
