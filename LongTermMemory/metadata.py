import os
from pathlib import Path
from enum import Enum

class DocCategory(Enum):
    REFERENCE = "reference"
    DEFAULT = "default"
    REFERENCE_VIDEO_SCRIPTS = "reference_video_scripts" # unused
    REFERENCE_ANALYSIS = "reference_analysis" # unused

sourceWeights = {
    DocCategory.REFERENCE.value: 1.5,
    DocCategory.DEFAULT.value: 1.0,
}

def update_docs_metadata(docs, category=DocCategory.DEFAULT.value, source_root=None):
    # for local files
    # metadata should not rely on file whole path
    # revisit when using remote files
    for i, doc in enumerate(docs):
        file_path = doc.metadata['source']
        file_created_at = os.path.getmtime(file_path)
        doc.metadata['source'] = (
            Path(file_path).resolve().relative_to(Path(source_root).resolve()).as_posix()
            if source_root is not None else os.path.basename(file_path)
        )
        doc.metadata['file_created_at'] = file_created_at
        # Portable fallback, not the document's actual publication date.
        doc.metadata['created_at'] = file_created_at
        doc.metadata['category'] = category


