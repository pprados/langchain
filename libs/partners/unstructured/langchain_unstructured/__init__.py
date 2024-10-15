from importlib import metadata

from langchain_unstructured.document_loaders import UnstructuredLoader, \
    UnstructuredPDFLoader, UnstructuredPDFParser

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "UnstructuredPDFLoader",
    "UnstructuredPDFParser",
    "UnstructuredLoader",
    "__version__",
]
