from pathlib import Path

def getUniqueStoreSources(sources):
    """ 
    Return a set of unique (file_name, created_at) from a list of sources.
    :param sources: a list of sources from a vector store, each source contains metadata about 'source' and 'created_at' 
    :return: a set of unique (file_name, created_at) from the sources.
    """
    unique_files = set()
    for metadata in sources['metadatas']:
        file_name = metadata['source']
        created_at = metadata['created_at']
        unique_files.add((file_name, created_at))
    return unique_files


def find_non_added_files(vector_store, local_files_path='./data/documents/default', file_pattern = '*.pdf'):
    """
    Return a dictionary with two keys: 'existing_sources' and 'new_sources'. 
    'existing_sources' is a set of unique (file_name, created_at) from the given vector store.
    'new_sources' is a set of all local files (file_name, created_at) that are not in the vector store yet.
    """
    sources = vector_store.get(include=["metadatas"])
    unique_sources = getUniqueStoreSources(sources)
    local_files = get_files_created_at(local_files_path, file_pattern = file_pattern)
    return {
        "existing_sources": unique_sources,
        "new_sources": local_files - unique_sources
    }

def get_files_created_at(folder_path, file_pattern = '*.pdf'):
    """
    Return a set of (file_name, created_at) for all PDF files in the given folder and its subfolders.
    """
    local_files = set()
    for file_path in Path(folder_path).rglob(file_pattern):
        modified_at = file_path.stat().st_mtime
        source = file_path.relative_to(folder_path).as_posix()
        local_files.add((source, modified_at))
    return local_files
