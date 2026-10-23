from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader

def pdf_loader(folder_path):
    """
    Loads all PDFs in the given folder path into a list of Documents.
    """
    loader = PyPDFDirectoryLoader(
        folder_path,
        extract_images=False,
        glob="**/*.pdf",
    )
    return loader.load()
   
def text_loader(folder_path):
    """
    Loads all text files in the given folder path into a list of Documents.
    """
    loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    return loader.load()
