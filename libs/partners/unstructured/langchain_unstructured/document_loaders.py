"""Unstructured document loader."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import IO, Any, Callable, Iterator, Optional, cast, Literal, Union, Dict, \
    BinaryIO, Tuple

import numpy as np
from PIL import Image
from typing_extensions import TypeAlias, List
from unstructured_client import UnstructuredClient  # type: ignore
from unstructured_client.models import operations, shared  # type: ignore

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import CONVERT_IMAGE_TO_TEXT, \
    ImagesPdfParser, PDFMinerParser, purge_metadata, _format_image_str
from langchain_community.document_loaders.pdf import BasePDFLoader
from langchain_core.document_loaders.base import BaseLoader, BaseBlobParser
from langchain_core.documents import Document

Element: TypeAlias = Any

logger = logging.getLogger(__file__)

_DEFAULT_URL = "https://api.unstructuredapp.io/general/v0/general"

class UnstructuredPDFParser(ImagesPdfParser):
    """Unstructured document loader interface.

    Setup:
        Install ``langchain-unstructured`` and set environment variable ``UNSTRUCTURED_API_KEY``.

        .. code-block:: bash
            pip install -U langchain-unstructured
            export UNSTRUCTURED_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python
            from langchain_unstructured import UnstructuredPDFParser

            loader = UnstructuredPDFParser(
                file_path = "example.pdf",
                api_key=UNSTRUCTURED_API_KEY,
                partition_via_api=True,
                chunking_strategy="by_title",
                strategy="fast",
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            1 2 0 2
            {'source': './example_data/layout-parser-paper.pdf', 'coordinates': {'points': ((16.34, 213.36), (16.34, 253.36), (36.34, 253.36), (36.34, 213.36)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'file_directory': './example_data', 'filename': 'layout-parser-paper.pdf', 'languages': ['eng'], 'last_modified': '2024-07-25T21:28:58', 'page_number': 1, 'filetype': 'application/pdf', 'category': 'UncategorizedText', 'element_id': 'd3ce55f220dfb75891b4394a18bcb973'}


    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            1 2 0 2
            {'source': './example_data/layout-parser-paper.pdf', 'coordinates': {'points': ((16.34, 213.36), (16.34, 253.36), (36.34, 253.36), (36.34, 213.36)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'file_directory': './example_data', 'filename': 'layout-parser-paper.pdf', 'languages': ['eng'], 'last_modified': '2024-07-25T21:28:58', 'page_number': 1, 'filetype': 'application/pdf', 'category': 'UncategorizedText', 'element_id': 'd3ce55f220dfb75891b4394a18bcb973'}


    Load URL:
        .. code-block:: python

            loader = UnstructuredLoader(web_url="https://www.example.com/")
            print(docs[0])

        .. code-block:: none

            page_content='Example Domain' metadata={'category_depth': 0, 'languages': ['eng'], 'filetype': 'text/html', 'url': 'https://www.example.com/', 'category': 'Title', 'element_id': 'fdaa78d856f9d143aeeed85bf23f58f8'}

        .. code-block:: python

            print(docs[1])

        .. code-block:: none

            page_content='This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.' metadata={'languages': ['eng'], 'parent_id': 'fdaa78d856f9d143aeeed85bf23f58f8', 'filetype': 'text/html', 'url': 'https://www.example.com/', 'category': 'NarrativeText', 'element_id': '3652b8458b0688639f973fe36253c992'}

    """

    def __init__(
            self,
            *,
            password: Optional[str] = None,  # FIXME PPR https://github.com/Unstructured-IO/unstructured/pull/3721
            mode: Literal["single", "paged", "elements"] = "single",
            pages_delimitor: str = "\n\n",
            extract_images: bool = False,
            partition_via_api: bool = False,
            post_processors: Optional[list[Callable[[str], str]]] = None,
            # SDK parameters
            api_key: Optional[str] = None,
            client: Optional[UnstructuredClient],
            url: Optional[str] = None,
            web_url: Optional[str] = None,
            images_to_text: CONVERT_IMAGE_TO_TEXT,
            **unstructured_kwargs: Any,

    ) -> None:
        """Initialize the parser.

        Args:
        """
        super().__init__(extract_images, images_to_text)
        if client is not None:
            disallowed_params = [("api_key", api_key), ("url", url)]
            bad_params = [
                param for param, value in disallowed_params if value is not None
            ]

            if bad_params:
                raise ValueError(
                    "if you are passing a custom `client`, you cannot also pass these "
                    f"params: {', '.join(bad_params)}."
                )

        unstructured_api_key = api_key or os.getenv("UNSTRUCTURED_API_KEY") or ""
        unstructured_url = url or os.getenv("UNSTRUCTURED_URL") or _DEFAULT_URL
        self.client = client or UnstructuredClient(
            api_key_auth=unstructured_api_key, server_url=unstructured_url
        )

        self.password = password
        _valid_modes = {"single", "elements", "paged"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        if extract_images and unstructured_kwargs.get("strategy") == "fast":
            logger.warning("Change strategy to 'auto' to extract images")
            unstructured_kwargs["strategy"] = "auto"
        if extract_images:
            if partition_via_api:
                logger.warning("extract_images is not supported with partition_via_api")
            else:
                unstructured_kwargs["extract_images_in_pdf"] = True
        self.images_to_text = images_to_text
        self.client = client
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        self.unstructured_kwargs = unstructured_kwargs

        self.client = client or UnstructuredClient(
            api_key_auth=unstructured_api_key, server_url=unstructured_url
        )
        if web_url:
            self.unstructured_kwargs["url"] = web_url

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        unstructured_kwargs = self.unstructured_kwargs.copy()
        if not self.partition_via_api:
            unstructured_kwargs["metadata_filename"] = (
                    blob.path or blob.metadata.get("source"))
        page_number=1
        with (blob.as_bytes_io() as pdf_file_obj):
            _single_doc_loader=_SingleDocumentLoader(
                file=pdf_file_obj,
                password=self.password,
                partition_via_api=self.partition_via_api,
                post_processors=self.post_processors,
                # SDK parameters
                client=self.client,
                **unstructured_kwargs,
            )
            path = Path(blob.source or blob.path)
            doc_metadata=purge_metadata(
                _single_doc_loader._get_metadata() | {
                    "source": blob.source,
                    "file_directory": str(path.parent),
                    "filename": path.name,
                    "filetype": blob.mimetype,
                    })
            if self.mode == "elements":
                 yield from _single_doc_loader.lazy_load()
            elif self.mode in ("paged","single"):

                page_content=[]
                page_break = False
                for doc in _single_doc_loader.lazy_load():
                    if page_break:
                        page_content.append(self.pages_delimitor)
                        page_break = False

                    if (doc.metadata.get("category") == "Image"
                        and "image_path" in doc.metadata):
                        image=np.array(Image.open(doc.metadata["image_path"]))
                        image_text=next(self.convert_image_to_text([image]))
                        if image_text:
                            page_content.append(_format_image_str.format(image_text=image_text))

                    elif doc.metadata.get("category") == "Table":
                        pass
                    elif doc.metadata.get("category") == "Title":
                        page_content.append("# "+doc.page_content)
                    elif doc.metadata.get("category") == "Header":
                        pass
                    elif doc.metadata.get("category") == "Footer":
                        pass
                    elif doc.metadata.get("category") == "PageBreak":
                        if self.mode == "paged":
                            yield Document(page_content="\n".join(page_content),
                                           metadata=(doc_metadata|
                                                     {"page": page_number}))
                            page_content.clear()
                            page_number += 1
                        else:
                            page_break = True
                    else:
                        # NarrativeText, UncategorizedText, Formula, FigureCaption,
                        # ListItem, Address, EmailAddress
                        if doc.metadata.get("category") not in ["NarrativeText", "UncategorizedText","Formula","FigureCaption","ListItem","Address","EmailAddress"]:
                            logger.warning(f"Unknown category {doc.metadata.get('category')}")
                        page_content.append(doc.page_content)
                if self.mode == "single":
                    yield Document(page_content="\n".join(page_content),
                                   metadata=doc_metadata)


class UnstructuredPDFLoader(BasePDFLoader):
    def __init__(self,
                 file_path: Union[str, List[str], Path, List[Path]],
                 *,
                 headers: Optional[Dict] = None,
                 mode: Literal["single", "paged"] = "single",
                 pages_delimitor: str = "\n\n",
                 extract_images: bool = False,
                 partition_via_api: bool = False,
                 post_processors: Optional[list[Callable[[str], str]]] = None,
                 # SDK parameters
                 api_key: Optional[str] = None,
                 client: Optional[UnstructuredClient] = None,
                 password:Optional[str] = None,
                 **unstructured_kwargs: Any,
                 ) -> None:
        super().__init__(file_path, headers=headers)

        self.parser = UnstructuredPDFParser(
            mode=mode,
            pages_delimitor=pages_delimitor,
            extract_images=extract_images,
            client=client,
            partition_via_api=partition_via_api,
            post_processors=post_processors,
            password=password,
            api_key=api_key,
            **unstructured_kwargs,
        )

    def lazy_load(
            self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(),
                                  path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


class UnstructuredLoader(BaseLoader):
    """Unstructured document loader interface.

    Setup:
        Install ``langchain-unstructured`` and set environment variable ``UNSTRUCTURED_API_KEY``.

        .. code-block:: bash
            pip install -U langchain-unstructured
            export UNSTRUCTURED_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python
            from langchain_unstructured import UnstructuredLoader

            loader = UnstructuredLoader(
                file_path = ["example.pdf", "fake.pdf"],
                api_key=UNSTRUCTURED_API_KEY,
                partition_via_api=True,
                chunking_strategy="by_title",
                strategy="fast",
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            1 2 0 2
            {'source': './example_data/layout-parser-paper.pdf', 'coordinates': {'points': ((16.34, 213.36), (16.34, 253.36), (36.34, 253.36), (36.34, 213.36)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'file_directory': './example_data', 'filename': 'layout-parser-paper.pdf', 'languages': ['eng'], 'last_modified': '2024-07-25T21:28:58', 'page_number': 1, 'filetype': 'application/pdf', 'category': 'UncategorizedText', 'element_id': 'd3ce55f220dfb75891b4394a18bcb973'}


    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            1 2 0 2
            {'source': './example_data/layout-parser-paper.pdf', 'coordinates': {'points': ((16.34, 213.36), (16.34, 253.36), (36.34, 253.36), (36.34, 213.36)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'file_directory': './example_data', 'filename': 'layout-parser-paper.pdf', 'languages': ['eng'], 'last_modified': '2024-07-25T21:28:58', 'page_number': 1, 'filetype': 'application/pdf', 'category': 'UncategorizedText', 'element_id': 'd3ce55f220dfb75891b4394a18bcb973'}


    Load URL:
        .. code-block:: python

            loader = UnstructuredLoader(web_url="https://www.example.com/")
            print(docs[0])

        .. code-block:: none

            page_content='Example Domain' metadata={'category_depth': 0, 'languages': ['eng'], 'filetype': 'text/html', 'url': 'https://www.example.com/', 'category': 'Title', 'element_id': 'fdaa78d856f9d143aeeed85bf23f58f8'}

        .. code-block:: python

            print(docs[1])

        .. code-block:: none

            page_content='This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.' metadata={'languages': ['eng'], 'parent_id': 'fdaa78d856f9d143aeeed85bf23f58f8', 'filetype': 'text/html', 'url': 'https://www.example.com/', 'category': 'NarrativeText', 'element_id': '3652b8458b0688639f973fe36253c992'}


    References
    ----------
    https://docs.unstructured.io/api-reference/api-services/sdk
    https://docs.unstructured.io/api-reference/api-services/overview
    https://docs.unstructured.io/open-source/core-functionality/partitioning
    https://docs.unstructured.io/open-source/core-functionality/chunking
    """  # noqa: E501

    def __init__(
            self,
            file_path: Optional[str | Path | list[str] | list[Path]] = None,
            *,
            file: Optional[IO[bytes] | list[IO[bytes]]] = None,
            partition_via_api: bool = False,
            post_processors: Optional[list[Callable[[str], str]]] = None,
            # SDK parameters
            api_key: Optional[str] = None,
            client: Optional[UnstructuredClient] = None,
            url: Optional[str] = None,
            web_url: Optional[str] = None,
            **kwargs: Any,
    ):
        """Initialize loader."""
        if file_path is not None and file is not None:
            raise ValueError("file_path and file cannot be defined simultaneously.")
        if client is not None:
            disallowed_params = [("api_key", api_key), ("url", url)]
            bad_params = [
                param for param, value in disallowed_params if value is not None
            ]

            if bad_params:
                raise ValueError(
                    "if you are passing a custom `client`, you cannot also pass these "
                    f"params: {', '.join(bad_params)}."
                )

        unstructured_api_key = api_key or os.getenv("UNSTRUCTURED_API_KEY") or ""
        unstructured_url = url or os.getenv("UNSTRUCTURED_URL") or _DEFAULT_URL

        self.client = client or UnstructuredClient(
            api_key_auth=unstructured_api_key, server_url=unstructured_url
        )

        self.file_path = file_path
        self.file = file
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        self.unstructured_kwargs = kwargs
        if web_url:
            self.unstructured_kwargs["url"] = web_url

    def lazy_load(self) -> Iterator[Document]:
        """Load file(s) to the _UnstructuredBaseLoader."""

        def load_file(
                f: Optional[IO[bytes]] = None, f_path: Optional[str | Path] = None
        ) -> Iterator[Document]:
            """Load an individual file to the _UnstructuredBaseLoader."""
            return _SingleDocumentLoader(
                file=f,
                file_path=f_path,
                partition_via_api=self.partition_via_api,
                post_processors=self.post_processors,
                # SDK parameters
                client=self.client,
                **self.unstructured_kwargs,
            ).lazy_load()

        if isinstance(self.file, list):
            for f in self.file:
                yield from load_file(f=f)
            return

        if isinstance(self.file_path, list):
            for f_path in self.file_path:
                yield from load_file(f_path=f_path)
            return

        # Call _UnstructuredBaseLoader normally since file and file_path are not lists
        yield from load_file(f=self.file, f_path=self.file_path)


class _SingleDocumentLoader(BaseLoader):
    """Provides loader functionality for individual document/file objects.

    Encapsulates partitioning individual file objects (file or file_path) either
    locally or via the Unstructured API.
    """

    def __init__(
            self,
            file_path: Optional[str | Path] = None,
            *,
            client: UnstructuredClient,
            file: Optional[IO[bytes]] = None,
            partition_via_api: bool = False,
            post_processors: Optional[list[Callable[[str], str]]] = None,
            password: Optional[str] = None,
            **kwargs: Any,
    ):
        """Initialize loader."""
        self.file_path = str(file_path) if isinstance(file_path, Path) else file_path
        self.file = file
        self.password = password
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        # SDK parameters
        self.client = client
        self.unstructured_kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        elements_json = (
            self._post_process_elements_json(self._elements_json)
            if self.post_processors
            else self._elements_json
        )
        file_metadata = purge_metadata(self._get_metadata())
        for element in elements_json:
            metadata = file_metadata.copy()
            metadata.update(element.get("metadata"))  # type: ignore
            metadata.update(
                {"category": element.get("category") or element.get("type")}
            )
            metadata.update({"element_id": element.get("element_id")})
            yield Document(
                page_content=cast(str, element.get("text")), metadata=metadata
            )

    @property
    def _elements_json(self) -> list[dict[str, Any]]:
        """Get elements as a list of dictionaries from local partition or via API."""
        if self.partition_via_api:
            return self._elements_via_api

        return self._convert_elements_to_dicts(self._elements_via_local)

    @property
    def _elements_via_local(self) -> list[Element]:
        try:
            from unstructured.partition.auto import partition  # type: ignore
        except ImportError:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        if self.file and self.unstructured_kwargs.get("metadata_filename") is None:
            raise ValueError(
                "If partitioning a fileIO object, metadata_filename must be specified"
                " as well.",
            )

        return partition(
            file=self.file, filename=self.file_path, password=self.password,
            **self.unstructured_kwargs
        )  # type: ignore

    @property
    def _elements_via_api(self) -> list[dict[str, Any]]:
        """Retrieve a list of element dicts from the API using the SDK client."""
        client = self.client
        req = self._sdk_partition_request
        response = client.general.partition(req)  # type: ignore
        if response.status_code == 200:
            return json.loads(response.raw_response.text)
        raise ValueError(
            f"Receive unexpected status code {response.status_code} from the API.",
        )

    @property
    def _file_content(self) -> bytes:
        """Get content from either file or file_path."""
        if self.file is not None:
            return self.file.read()
        elif self.file_path:
            with open(self.file_path, "rb") as f:
                return f.read()
        raise ValueError("file or file_path must be defined.")

    @property
    def _sdk_partition_request(self) -> operations.PartitionRequest:
        return operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=self._file_content, file_name=str(self.file_path)
                ),
                **self.unstructured_kwargs,
            ),
        )

    def _convert_elements_to_dicts(
            self, elements: list[Element]
    ) -> list[dict[str, Any]]:
        return [element.to_dict() for element in elements]

    # def _get_metadata(self) -> dict[str, Any]:
    #     """Get file_path metadata if available."""
    #     return {"source": self.file_path} if self.file_path else {}
    #
    def _get_metadata(self) -> Dict[str, Any]:
        from pdfminer.pdfpage import PDFPage, PDFParser, PDFDocument
        # Create a PDF parser object associated with the file object.
        parser = PDFParser(self.file)
        # Create a PDF document object that stores the document structure.
        doc = PDFDocument(parser, password=self.password)
        metadata = {}

        for info in doc.info:
            metadata.update(info)
        for k, v in metadata.items():
            try:
                metadata[k] = PDFMinerParser.resolve_and_decode(v)
            except Exception as e:  # pragma: nocover
                # This metadata value could not be parsed. Instead of failing the PDF
                # read, treat it as a warning only if `strict_metadata=False`.
                logger.warning(
                    f'[WARNING] Metadata key "{k}" could not be parsed due to '
                    f"exception: {str(e)}"
                )

        # Count number of pages.
        metadata["total_pages"] = len(list(PDFPage.create_pages(doc)))
        return metadata


    def _post_process_elements_json(
            self, elements_json: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply post processing functions to extracted unstructured elements.

        Post processing functions are str -> str callables passed
        in using the post_processors kwarg when the loader is instantiated.
        """
        if self.post_processors:
            for element in elements_json:
                for post_processor in self.post_processors:
                    element["text"] = post_processor(str(element.get("text")))
        return elements_json
