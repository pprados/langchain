"""Module contains common parsers for PDFs."""

# PPR:
# - ajouter paramètre image pour unstructured (alias de parametre extract_images)
#  - ajouter un media "tableau", ajouter page_numbers
# - dans load_and_split(), retourner des medias (image, texte, tableau)
# - voir comment invoquer un LLM multimodale avec cela. Soit après le load_and_split()
# soit lors du load_and_split().
from __future__ import annotations

import html
import logging
import re
import warnings
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union, List, Tuple, Literal, BinaryIO, )
from urllib.parse import urlparse

import numpy as np

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core._api.deprecation import (
    deprecated,
)
from langchain_core.documents import Document

if TYPE_CHECKING:
    import pymupdf.pymupdf
    import pdfminer.layout
    import pdfplumber.page
    import pypdf._page
    import pypdfium2._helpers.page
    from pypdf import PageObject
    from textractor.data.text_linearization_config import TextLinearizationConfig
    from pdfplumber.utils import text, geometry  # import WordExctractor, TextMap

_PDF_FILTER_WITH_LOSS = ["DCTDecode", "DCT", "JPXDecode"]
_PDF_FILTER_WITHOUT_LOSS = [
    "LZWDecode",
    "LZW",
    "FlateDecode",
    "Fl",
    "ASCII85Decode",
    "A85",
    "ASCIIHexDecode",
    "AHx",
    "RunLengthDecode",
    "RL",
    "CCITTFaxDecode",
    "CCF",
    "JBIG2Decode",
]

logger = logging.getLogger(__name__)


@deprecated(since="3.0.0",
            alternative="Use Parser.convert_image_to_text()")
def extract_from_images_with_rapidocr(
        images: Sequence[Union[Iterable[np.ndarray], bytes]],
) -> str:
    """Extract text from images with RapidOCR.

    Args:
        images: Images to extract text from.

    Returns:
        Text extracted from images.

    Raises:
        ImportError: If `rapidocr-onnxruntime` package is not installed.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise ImportError(
            "`rapidocr-onnxruntime` package not found, please install it with "
            "`pip install rapidocr-onnxruntime`"
        )
    ocr = RapidOCR()
    text = ""
    for img in images:
        result, _ = ocr(img)
        if result:
            result = [text[1] for text in result]
            text += "\n".join(result)
    return text


class OCRPdfParser(BaseBlobParser):
    """Abstract interface for blob parsers with OCR."""

    def convert_image_to_text(
            self,
            images: Sequence[Union[Iterable[np.ndarray], bytes]],
    ) -> str:
        """Extract text from images.
        Can be overloaded to use another OCR algorithm, or to use
        a multimodal model to describe the images.

        Args:
            images: Images to extract text from.

        Returns:
            Text extracted from images.

        Raises:
            ImportError: If `rapidocr-onnxruntime` package is not installed.
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ImportError(
                "`rapidocr-onnxruntime` package not found, please install it with "
                "`pip install rapidocr-onnxruntime`"
            )
        ocr = RapidOCR()
        text = ""
        for img in images:
            result, _ = ocr(img)
            if result:
                result = [text[1] for text in result]
                text += "\n".join(result)
        return text


class PyPDFParser(OCRPdfParser):
    """Load `PDF` using `pypdf`"""

    def __init__(
            self,
            *,
            password: Optional[Union[str, bytes]] = None,
            mode: Literal["flow", "page"] = "page",
            extract_images: bool = False,

            extract_tables: bool = False,
            extraction_mode: Literal["plain", "page"] = "plain",
            extraction_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.password = password
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.mode = mode
        self.extraction_mode = extraction_mode
        self.extraction_kwargs = extraction_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "`pypdf` package not found, please install it with "
                "`pip install pypdf`"
            )

        def _extract_text_from_page(page: "PageObject") -> str:
            """
            Extract text from image given the version of pypdf.
            """
            if pypdf.__version__.startswith("3"):
                return page.extract_text()
            else:
                return page.extract_text(
                    extraction_mode=self.extraction_mode, **self.extraction_kwargs
                )

        with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)

            if self.mode == "page":
                yield from [
                    Document(
                        page_content=_extract_text_from_page(page=page)
                                     + self.extract_images_from_page(page),
                        metadata={"source": blob.source, "page": page_number},
                        # type: ignore[attr-defined]
                    )
                    for page_number, page in enumerate(pdf_reader.pages)
                ]
            elif self.mode == "flow":
                text = "".join(
                    _extract_text_from_page(page=page) + self.extract_images_from_page(
                        page)
                    for page in pdf_reader.pages
                )
                yield Document(page_content=text, metadata={"source": blob.source})
            else:
                raise ValueError("mode must be flow, plain or page")

    def extract_images_from_page(self, page: pypdf._page.PageObject) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images or "/XObject" not in page["/Resources"].keys():
            return ""

        xObject = page["/Resources"]["/XObject"].get_object()  # type: ignore
        images = []
        for obj in xObject:
            if xObject[obj]["/Subtype"] == "/Image":
                if xObject[obj]["/Filter"][1:] in _PDF_FILTER_WITHOUT_LOSS:
                    height, width = xObject[obj]["/Height"], xObject[obj]["/Width"]

                    images.append(
                        np.frombuffer(xObject[obj].get_data(), dtype=np.uint8).reshape(
                            height, width, -1
                        )
                    )
                elif xObject[obj]["/Filter"][1:] in _PDF_FILTER_WITH_LOSS:
                    images.append(xObject[obj].get_data())
                else:
                    warnings.warn("Unknown PDF Filter!")
        return self.convert_image_to_text(images)


class PDFMinerParser(OCRPdfParser):
    """Parse `PDF` using `PDFMiner`."""

    def __init__(self,
                 *,
                 password: Optional[str] = None,
                 mode: Literal["flow", "page"] = "page",
                 extract_images: bool = False,

                 concatenate_pages: Optional[bool] = None,
                 ):
        """Initialize a parser based on PDFMiner.

        Args:
            extract_images: Whether to extract images from PDF.
            mode: Extraction mode to use. Either "flow" or "page".
            concatenate_pages: Depreceted. If True, concatenate all PDF pages into one a single
                               document. Otherwise, return one document per page.
        """
        if mode not in ["flow", "page"]:
            raise ValueError("mode must be flow or page")
        self.extract_images = extract_images
        self.mode = mode
        self.password = password
        if concatenate_pages is not None:
            warnings.warn(
                "`concatenate_pages` parameter is deprecated. "
                "Use `mode='flow'` instead."
            )
            mode = "flow" if concatenate_pages else "page"
            if mode != self.mode:
                warnings.warn(
                    f"Overriding `concatenate_pages` to "
                    f"`mode='{mode}'`")

    @staticmethod
    def decode_text(s: Union[bytes, str]) -> str:
        """
        Decodes a PDFDocEncoding string to Unicode.
        Adds py3 compatibility to pdfminer's version.
        """
        from pdfminer.utils import PDFDocEncoding

        if isinstance(s, bytes) and s.startswith(b"\xfe\xff"):
            return str(s[2:], "utf-16be", "ignore")
        try:
            ords = (ord(c) if isinstance(c, str) else c for c in s)
            return "".join(PDFDocEncoding[o] for o in ords)
        except IndexError:
            return str(s)

    @staticmethod
    def resolve_and_decode(obj: Any) -> Any:
        """Recursively resolve the metadata values."""
        from pdfminer.psparser import PSLiteral

        if hasattr(obj, "resolve"):
            obj = obj.resolve()
        if isinstance(obj, list):
            return list(map(PDFMinerParser.resolve_and_decode, obj))
        elif isinstance(obj, PSLiteral):
            return PDFMinerParser.decode_text(obj.name)
        elif isinstance(obj, (str, bytes)):
            return PDFMinerParser.decode_text(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = PDFMinerParser.resolve_and_decode(v)
            return obj

        return obj

    def _get_metadata(
            self,
            fp: BinaryIO,
            password: str = "",
            caching: bool = True,
    ) -> Iterator[Tuple["PDFPage", Dict[str, Any]]]:
        from pdfminer.pdfpage import PDFPage, PDFParser, PDFDocument
        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        doc = PDFDocument(parser, password=password, caching=caching)
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

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""

        if not self.extract_images:
            try:
                from pdfminer.high_level import extract_text
            except ImportError:
                raise ImportError(
                    "`pdfminer` package not found, please install it with "
                    "`pip install pdfminer.six`"
                )

            with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
                if self.mode == "flow":
                    file_metadata = self._get_metadata(pdf_file_obj,
                                                       password=self.password)
                    text = extract_text(pdf_file_obj, password=self.password)
                    metadata = {**file_metadata,
                                **{"source": blob.source}}  # type: ignore[attr-defined]
                    yield Document(page_content=text, metadata=metadata)
                elif self.mode == "page":
                    from pdfminer.pdfpage import PDFPage

                    file_metadata = self._get_metadata(pdf_file_obj,
                                                       password=self.password)
                    pages = PDFPage.get_pages(pdf_file_obj, password=self.password)
                    for i, _ in enumerate(pages):
                        text = extract_text(pdf_file_obj, page_numbers=[i],
                                            password=self.password)
                        metadata = {**file_metadata, **{"source": blob.source,
                                                        "page": str(
                                                            i)}}  # type: ignore[attr-defined]
                        yield Document(page_content=text, metadata=metadata)
                else:
                    raise ValueError(
                        "mode must be flow or page")
        else:
            import io

            from pdfminer.converter import PDFPageAggregator, TextConverter
            from pdfminer.layout import LAParams
            from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
            from pdfminer.pdfpage import PDFPage

            text_io = io.StringIO()
            with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
                pages = PDFPage.get_pages(pdf_file_obj, password=self.password)
                rsrcmgr = PDFResourceManager()
                device_for_text = TextConverter(rsrcmgr, text_io, laparams=LAParams())
                device_for_image = PDFPageAggregator(rsrcmgr, laparams=LAParams())
                interpreter_for_text = PDFPageInterpreter(rsrcmgr, device_for_text)
                interpreter_for_image = PDFPageInterpreter(rsrcmgr, device_for_image)
                for i, page in enumerate(pages):
                    interpreter_for_text.process_page(page)
                    interpreter_for_image.process_page(page)
                    content = text_io.getvalue() + self.extract_images_from_page(
                        device_for_image.get_result()
                    )
                    text_io.truncate(0)
                    text_io.seek(0)
                    metadata = {"source": blob.source,
                                "page": str(i)}  # type: ignore[attr-defined]
                    yield Document(page_content=content, metadata=metadata)

    def extract_images_from_page(self, page: pdfminer.layout.LTPage) -> str:
        """Extract images from page and get the text with RapidOCR."""
        import pdfminer

        def get_image(layout_object: Any) -> Any:
            if isinstance(layout_object, pdfminer.layout.LTImage):
                return layout_object
            if isinstance(layout_object, pdfminer.layout.LTContainer):
                for child in layout_object:
                    return get_image(child)
            else:
                return None

        images = []

        for img in list(filter(bool, map(get_image, page))):
            if img.stream["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    np.frombuffer(img.stream.get_data(), dtype=np.uint8).reshape(
                        img.stream["Height"], img.stream["Width"], -1
                    )
                )
            elif img.stream["Filter"].name in _PDF_FILTER_WITH_LOSS:
                images.append(img.stream.get_data())
            else:
                warnings.warn("Unknown PDF Filter!")
        return self.convert_image_to_text(images)


class PyMuPDFParser(OCRPdfParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(
            self,
            *,
            password: Optional[str] = None,
            # mode: Literal["flow", "page"] = "page",  # FIXME
            extract_images: bool = False,

            text_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pymupdf.Page.get_text()``.
        """
        self.password = password
        self.text_kwargs = text_kwargs or {}
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""

        import pymupdf

        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            if blob.data is None:  # type: ignore[attr-defined]
                doc = pymupdf.open(file_path)
            else:
                doc = pymupdf.open(stream=file_path, filetype="pdf")
            if doc.is_encrypted:
                doc.authenticate(self.password)
            yield from [
                Document(
                    page_content=self._get_page_content(doc, page, blob),
                    metadata=self._extract_metadata(doc, page, blob),
                )
                for page in doc
            ]

    def _get_page_content(
            self, doc: pymupdf.pymupdf.Document, page: pymupdf.pymupdf.Page, blob: Blob
    ) -> str:
        """
        Get the text of the page using PyMuPDF and RapidOCR and issue a warning
        if it is empty.
        """
        content = page.get_text(**self.text_kwargs) + self._extract_images_from_page(
            doc, page
        )

        if not content:
            warnings.warn(
                f"Warning: Empty content on page "
                f"{page.number} of document {blob.source}"
            )

        return content

    def _extract_metadata(
            self, doc: pymupdf.pymupdf.Document, page: pymupdf.pymupdf.Page, blob: Blob
    ) -> dict:
        """Extract metadata from the document and page."""
        return dict(
            {
                "source": blob.source,  # type: ignore[attr-defined]
                "file_path": blob.source,  # type: ignore[attr-defined]
                "page": page.number,
                "total_pages": len(doc),
            },
            **{
                k: doc.metadata[k]
                for k in doc.metadata
                if isinstance(doc.metadata[k], (str, int))
            },
        )

    def _extract_images_from_page(
            self, doc: pymupdf.pymupdf.Document, page: pymupdf.pymupdf.Page
    ) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""
        import pymupdf

        img_list = page.get_images()
        imgs = []
        for img in img_list:
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            imgs.append(
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, -1
                )
            )
        return self.convert_image_to_text(imgs)


class PyMuPDF4LLMParser(OCRPdfParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(
            self,
            *,
            # password: Optional[str] = None,  # FIXME
            mode: Literal["flow", "page"] = "page",

            to_markdown_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        # self.password = password
        _to_markdown_kwargs = to_markdown_kwargs or {}
        if mode == "flow":
            del _to_markdown_kwargs["page_chunks"]
        elif mode == "page":
            _to_markdown_kwargs["page_chunks"] = True
        else:
            raise ValueError("mode must be flow or page")
        self.to_markdown_kwargs = _to_markdown_kwargs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pymupdf4llm
        # FIXME
        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            if blob.data is None:  # type: ignore[attr-defined]
                for mu_doc in pymupdf4llm.to_markdown(
                        file_path,
                        **self.to_markdown_kwargs,
                ):
                    yield Document(
                        page_content=mu_doc['text'],
                        metadata=self._purge_metadata(mu_doc['metadata'])
                    )
                    # TODO: extraire les images. Voir PyMuPDFParser
            else:
                raise NotImplementedError("stream not implemented")

    _map_key = {
        "page_count": "total_pages",
        "file_path": "source"
    }
    _date_key = [
        "creationdate", "moddate"
    ]

    def _purge_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Purge metadata from unwanted keys."""
        new_metadata = {}
        for k in metadata.keys():
            if k.lower() in PyMuPDF4LLMParser._date_key:
                try:
                    new_metadata[k] = datetime.strptime(
                        metadata[k].replace("'", ""),
                        "D:%Y%m%d%H%M%S%z").isoformat("T")
                except ValueError:
                    new_metadata[k] = metadata[k]
            elif k.lower() in PyMuPDF4LLMParser._map_key:
                # Normliaze key with others PDF parser
                new_metadata[PyMuPDF4LLMParser._map_key[k.lower()]] = metadata[k]
            elif isinstance(metadata[k], (str, int)):
                new_metadata[k] = metadata[k]
        return new_metadata


class PyPDFium2Parser(OCRPdfParser):
    """Parse `PDF` with `PyPDFium2`."""

    def __init__(self,
                 *,
                 password: Optional[str] = None,
                 # mode: Literal["flow", "page"] = "page",  # FIXME
                 extract_images: bool = False
                 ) -> None:
        """Initialize the parser."""
        try:
            import pypdfium2  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )
        self.password = password
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pypdfium2

        # pypdfium2 is really finicky with respect to closing things,
        # if done incorrectly creates seg faults.
        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            pdf_reader = pypdfium2.PdfDocument(file_path,
                                               password=self.password,
                                               autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    content += "\n" + self._extract_images_from_page(page)
                    page.close()
                    metadata = {"source": blob.source,
                                "page": page_number}  # type: ignore[attr-defined]
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()

    def _extract_images_from_page(self, page: pypdfium2._helpers.page.PdfPage) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""

        import pypdfium2.raw as pdfium_c

        images = list(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)))

        images = list(map(lambda x: x.get_bitmap().to_numpy(), images))
        return self.convert_image_to_text(images)


class PDFPlumberParser(OCRPdfParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(
            self,
            *,
            password: Optional[str] = None,
            mode: Literal["flow", "page", "layout"] = "page",
            extract_images: bool = False,

            text_kwargs: Optional[Mapping[str, Any]] = None,
            extract_tables_settings: Optional[Mapping[str, Any]] = None,
            extract_tables: Optional[Literal["csv", "markdown", "html"]] = None,
            dedupe: bool = False,
            include_page_breaks: bool = False,  # FIXME vs unstructured
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.password = password
        self.text_kwargs = text_kwargs or {}
        self.extract_tables_settings = extract_tables_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_y_tolerance": 5,
            "intersection_x_tolerance": 15,
        }

        self.mode = mode
        self.dedupe = dedupe
        self.extract_images = extract_images
        self.extract_tables = extract_tables

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            doc = pdfplumber.open(file_path,
                                  password=self.password)  # open document
            if self.mode == "page":
                yield from self._process_mode_pages(blob, doc)
            elif self.mode == "layout":
                yield from self._process_mode_layout(blob, doc)
            elif self.mode == "flow":
                yield from self._process_mode_plain(blob, doc)
            else:
                raise ValueError("mode must be flow, layout or page")

    def _process_mode_plain(self, blob: Blob, doc: pdfplumber.pdf.PDF):
        # TODO: avec ou sans tables
        from pdfplumber.utils import geometry  # import WordExctractor, TextMap
        contents = []
        tables_as_html = []
        images = []
        for page in doc.pages:
            tables_bbox: List[Tuple[
                float, float, float, float]] = self._extract_tables_bbox_from_page(
                page)
            tables_content = self._extract_tables_from_page(page)
            images_bbox = [geometry.obj_to_bbox(image) for image in page.images]
            images_content = self._extract_images_from_page(page)
            images.extend(images_content)
            page_content = self._process_page_content(
                page,
                tables_bbox,
                tables_content,
                images_bbox,
                images_content)
            contents.append(page_content)
            tables_as_html.extend([self._convert_table(table)
                                   for
                                   table in tables_content])
        yield Document(
            page_content="\n".join(contents),
            metadata=dict(
                {
                    "source": blob.source,  # type: ignore[attr-defined]
                    "file_path": blob.source,  # type: ignore[attr-defined]
                    "total_pages": len(doc.pages),
                    "tables_as_html": tables_as_html,
                    "images": images,
                },
                **{
                    k: doc.metadata[k]
                    for k in doc.metadata
                    if type(doc.metadata[k]) in [str, int]
                }),

        )

    def _process_mode_pages(self, blob: Blob, doc: pdfplumber.pdf.PDF):
        from pdfplumber.utils import geometry  # import WordExctractor, TextMap
        for page in doc.pages:
            tables_bbox: List[Tuple[
                float, float, float, float]] = self._extract_tables_bbox_from_page(
                page)
            tables_content = self._extract_tables_from_page(page)
            images_bbox = [geometry.obj_to_bbox(image) for image in page.images]
            images_content = self._extract_images_from_page(page)
            page_content = self._process_page_content(
                page,
                tables_bbox,
                tables_content,
                images_bbox,
                images_content)
            yield Document(
                page_content=page_content,
                metadata=dict(
                    {
                        "source": blob.source,  # type: ignore[attr-defined]
                        "file_path": blob.source,  # type: ignore[attr-defined]
                        "page": page.page_number - 1,
                        "total_pages": len(doc.pages),
                        "tables_as_html": [self._convert_table_to_html(table)
                                           for
                                           table in tables_content],
                        "images": images_content,
                    },
                    **{
                        k: doc.metadata[k]
                        for k in doc.metadata
                        if type(doc.metadata[k]) in [str, int]
                    }),

            )

    def _process_mode_layout(self, blob: Blob, doc: pdfplumber.pdf.PDF):
        from pdfplumber.utils import geometry
        for page in doc.pages:
            tables_bbox: List[Tuple[
                float, float, float, float]] = self._extract_tables_bbox_from_page(
                page)
            tables_content = self._extract_tables_from_page(page)
            images_bbox = [geometry.obj_to_bbox(image) for image in page.images]
            images_content = self._extract_images_from_page(page)
            for content in self._split_page_content(
                    page, tables_bbox, tables_content, images_bbox, images_content,
                    **self.text_kwargs):
                if isinstance(content, str):
                    yield Document(
                        page_content=content,
                        metadata=dict(
                            {
                                "source": blob.source,
                                # type: ignore[attr-defined]
                                "file_path": blob.source,
                                # type: ignore[attr-defined]
                                "page": page.page_number - 1,
                                "total_pages": len(doc.pages),
                                "tables_as_html": [
                                    self._convert_table_to_html(table)
                                    for
                                    table in tables_content]
                            },
                            **{
                                k: doc.metadata[k]
                                for k in doc.metadata
                                if type(doc.metadata[k]) in [str, int]
                            }),
                    )
                elif isinstance(content, np.ndarray):
                    # If change the interface to return list[BaseMedia]
                    # in place of list[Document]
                    # yield Blob.from_data(content,
                    #                      mime_type="image/jpeg",  # FIXME
                    #                      metadata={
                    #                          "source": blob.source,
                    #                          "page": page.page_number - 1,
                    #                          "total_pages": len(doc.pages),
                    #                      }
                    #                      )
                    yield Document(self.convert_image_to_text([content]),
                                   metadata=dict({
                                       "source": blob.source,
                                       "type": "image",  # FIXME vs unstructured
                                       "page": page.page_number - 1,
                                       "total_pages": len(doc.pages),
                                       "images": [content]
                                   },
                                       **{
                                           k: doc.metadata[k]
                                           for k in doc.metadata
                                           if type(doc.metadata[k]) in [str, int]
                                       }),

                                   )
                else:
                    yield Document(
                        page_content=self._convert_table(content),
                        metadata=dict(
                            {
                                "source": blob.source,
                                "type": "table",  # FIXME
                                "file_path": blob.source,
                                "page": page.page_number - 1,
                                "total_pages": len(doc.pages),
                                "tables_as_html": [
                                    self._convert_table_to_html(table)
                                    for
                                    table in tables_content]
                            },
                            **{
                                k: doc.metadata[k]
                                for k in doc.metadata
                                if type(doc.metadata[k]) in [str, int]
                            }),
                    )
        return

    def _process_page_content(
            self,
            page: pdfplumber.page.Page,
            tables_bbox: List[Tuple[float, float, float, float]],
            tables_content: List[str],
            images_bbox: List[Tuple[float, float, float, float]],
            images_content: List[np.ndarray],
    ) -> str:
        page.extract_words()
        result = []
        for content in self._split_page_content(
                page, tables_bbox, tables_content, images_bbox, images_content,
                **self.text_kwargs
        ):
            if isinstance(content, str):
                result.append(content)
            elif isinstance(content, np.ndarray):
                result.append(self.convert_image_to_text([content]))
            else:
                result.append(self._convert_table(content))  # FIXME
        return " ".join(result)

    def _split_page_content(
            self,
            page: pdfplumber.page.Page,
            tables_bbox: List[Tuple[float, float, float, float]],
            tables_content: List[str],
            images_bbox: List[Tuple[float, float, float, float]],
            images_content: List[np.ndarray],
            **kwargs: Any,
    ) -> List[Union[str, List[List[str]]]]:
        """Process the page content based on dedupe."""
        from pdfplumber.utils import text, geometry  # import WordExctractor, TextMap

        # Iterate over words. If a word is in a table,
        # yield the accumulated text, and the table
        # A the word is in a previously see table, ignore it
        # Finish with the accumulated text
        kwargs.update(
            {
                "keep_blank_chars": True,
                # "use_text_flow": True,
                # "presorted": True,
                "layout_bbox": kwargs.get(
                    "layout_bbox") or geometry.objects_to_bbox(page.chars),
            }
        )

        chars = page.dedup_chars() if self.dedupe else page.chars
        extractor = text.WordExtractor(
            **{k: kwargs[k] for k in text.WORD_EXTRACTOR_KWARGS if k in kwargs}
        )
        wordmap = extractor.extract_wordmap(chars)
        extract_wordmaps = []
        used_arrays = [False] * len(tables_bbox)
        for (word, o) in wordmap.tuples:
            # print(f"  Try with '{word['text']}' ...")
            is_table = False
            word_bbox = geometry.obj_to_bbox(word)
            for i, table_bbox in enumerate(tables_bbox):
                if geometry.get_bbox_overlap(word_bbox, table_bbox):
                    # Find a world in a table
                    # print("  Find in an array")
                    is_table = True
                    if not used_arrays[i]:
                        # First time I see a word in this array
                        # Yield the previous part
                        if extract_wordmaps:
                            new_wordmap = text.WordMap(tuples=extract_wordmaps)
                            new_textmap = new_wordmap.to_textmap(
                                **{k: kwargs[k] for k in text.TEXTMAP_KWARGS if
                                   k in kwargs}
                            )
                            # print(f"yield {new_textmap.to_string()}")
                            yield new_textmap.to_string()
                            extract_wordmaps.clear()
                        # and yield the table
                        used_arrays[i] = True
                        # print(f"yield table {i}")
                        yield tables_content[i]
                    else:
                        # print(f"  saute yield sur tableau deja vu")
                        pass
                    break
            if not is_table:
                # print(f'  Add {word["text"]}')
                extract_wordmaps.append((word, o))
        if extract_wordmaps:
            # Text after the array ?
            new_wordmap = text.WordMap(tuples=extract_wordmaps)
            new_textmap = new_wordmap.to_textmap(
                **{k: kwargs[k] for k in text.TEXTMAP_KWARGS if k in kwargs}
            )
            # print(f"yield {new_textmap.to_string()}")
            yield new_textmap.to_string()
        # Add images-
        for content in images_content:
            yield content

    def _extract_images_from_page(self, page: pdfplumber.page.Page) -> List[
        np.ndarray]:  # FIXME
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return []

        images = []
        for img in page.images:
            if img["stream"]["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    np.frombuffer(img["stream"].get_data(), dtype=np.uint8).reshape(
                        img["stream"]["Height"], img["stream"]["Width"], -1
                    )
                )
            elif img["stream"]["Filter"].name in _PDF_FILTER_WITH_LOSS:
                images.append(np.frombuffer(img["stream"].get_data(), dtype=np.uint8))
            else:
                warnings.warn("Unknown PDF Filter!")

        return images

    def _extract_tables_bbox_from_page(self,
                                       page: pdfplumber.page.Page,
                                       ) -> List["PDFPlumberTable"]:

        if not self.extract_tables:
            return []
        from pdfplumber.table import TableSettings
        table_settings = self.extract_tables_settings
        tset = TableSettings.resolve(table_settings)
        return [table.bbox for table in page.find_tables(tset)]

    def _extract_tables_from_page(self,
                                  page: pdfplumber.page.Page,
                                  ) -> List["PDFPlumberTable"]:
        if not self.extract_tables:
            return []
        table_settings = self.extract_tables_settings
        tables_list = page.extract_tables(table_settings)
        return tables_list

    def _convert_table(self, table: List[List[str]]) -> str:
        format = self.extract_tables
        if format is None:
            return ""
        if format == "markdown":
            return self._convert_table_to_markdown(table)
        elif format == "html":
            return self._convert_table_to_html(table)
        elif format == "csv":
            return self._convert_table_to_csv(table)
        else:
            raise ValueError(f"Unknown table format: {format}")

    def _convert_table_to_csv(self, table: List[List[str]]) -> str:
        """Output table content as a string in Github-markdown format."""
        clean = True
        if not table:
            return ""
        col_count = len(table[0])
        output = ""

        # skip first row in details if header is part of the table
        # j = 0 if self.header.external else 1

        # iterate over detail rows
        for row in table:
            line = ""
            for i, cell in enumerate(row):
                # output None cells with empty string
                cell = "" if cell is None else cell.replace("\n", " ")
                line += cell + ","
            line += "\n"
            output += line
        return output + "\n"

    def _convert_table_to_html(self, table: List[List[str]]) -> str:
        """Output table content as a string in HTML format.

        If clean is true, markdown syntax is removed from cell content."""
        if not len(table):
            return ""
        output = "<table>\n"
        clean = True

        # iterate over detail rows
        for row in table:
            line = "<tr>"
            for i, cell in enumerate(row):
                # output None cells with empty string
                cell = "" if cell is None else cell.replace("\n", " ")
                if clean:  # remove sensitive syntax
                    cell = html.escape(cell.replace("-", "&#45;"))
                line += "<td>" + cell + "</td>"
            line += "</tr>\n"
            output += line
        return output + "</table>\n"

    def _convert_table_to_markdown(self, table: List[List[str]]) -> str:
        """Output table content as a string in Github-markdown format."""
        clean = True
        if not table:
            return ""
        col_count = len(table[0])
        output = "|"

        output += "\n"
        output += "|" + "|".join("" for i in range(col_count)) + "|\n"
        output += "|" + "|".join("---" for i in range(col_count)) + "|\n"

        # skip first row in details if header is part of the table
        # j = 0 if self.header.external else 1

        # iterate over detail rows
        for row in table:
            line = "|"
            for i, cell in enumerate(row):
                # output None cells with empty string
                cell = "" if cell is None else cell.replace("\n", " ")
                if clean:  # remove sensitive syntax
                    cell = html.escape(cell.replace("-", "&#45;"))
                line += cell + "|"
            line += "\n"
            output += line
        return output + "\n"


REGEX = Union[re, str]


class PDFRouterParser(BaseBlobParser):
    """
    Parse PDFs using different parsers based on the metadata of the PDF.
    The routes are defined as a list of tuples, where each tuple contains
    the regex pattern for the producer, creator, and page, and the parser to use.
    The parser is used if the regex pattern matches the metadata of the PDF.
    Use the route in the correct order, as the first matching route is used.
    Add a default route (None, None, None, parser) at the end to catch all PDFs.

    Sample:
    ```python
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
    from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
    from langchain_community.document_loaders.parsers import PDFPlumberParser
    routes = [
        ("Microsoft", "Microsoft", None, PyMuPDFParser()),
        ("LibreOffice", None, None, PDFPlumberParser()),
        (None, None, None, PyPDFium2Parser())
    ]
    loader = PDFRouterLoader(filename, routes)
    loader.load()
    ```
    """

    def __init__(
            self,
            routes: List[
                Tuple[
                    Optional[REGEX], Optional[REGEX], Optional[REGEX], BaseBlobParser]],
            *,
            password: Optional[str] = None,
    ):
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )
        super().__init__()
        self.password = password
        new_routes = []
        for producer, create, page, parser in routes:
            if isinstance(producer, str):
                producer = re.compile(producer)
            if isinstance(create, str):
                create = re.compile(create)
            if isinstance(page, str):
                page = re.compile(page)
            new_routes.append((producer, create, page, parser))
        self.routes = new_routes

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        from pypdf import PdfReader

        with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]

            with PdfReader(pdf_file_obj,
                           password=self.password) as reader:
                if reader.metadata:
                    producer, create = reader.metadata.producer, reader.metadata.creator
                    page1 = reader.pages[0].extract_text()
                for re_producer, re_create, re_page, parser in self.routes:
                    is_producer = (not re_producer
                                   or re_producer.search(producer))
                    is_create = (not re_create
                                 or re_create.search(create))
                    is_page = (not re_page
                               or re_page.search(page1))
                    if is_producer and is_create and is_page:
                        yield from parser.lazy_parse(blob)
                        break


# %% --------- Online pdf loader ---------
class AmazonTextractPDFParser(BaseBlobParser):
    """Send `PDF` files to `Amazon Textract` and parse them.

    For parsing multi-page PDFs, they have to reside on S3.

    The AmazonTextractPDFLoader calls the
    [Amazon Textract Service](https://aws.amazon.com/textract/)
    to convert PDFs into a Document structure.
    Single and multi-page documents are supported with up to 3000 pages
    and 512 MB of size.

    For the call to be successful an AWS account is required,
    similar to the
    [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
    requirements.

    Besides the AWS configuration, it is very similar to the other PDF
    loaders, while also supporting JPEG, PNG and TIFF and non-native
    PDF formats.

    ```python
    from langchain_community.document_loaders import AmazonTextractPDFLoader
    loader=AmazonTextractPDFLoader("example_data/alejandro_rosalez_sample-small.jpeg")
    documents = loader.load()
    ```

    One feature is the linearization of the output.
    When using the features LAYOUT, FORMS or TABLES together with Textract

    ```python
    from langchain_community.document_loaders import AmazonTextractPDFLoader
    # you can mix and match each of the features
    loader=AmazonTextractPDFLoader(
        "example_data/alejandro_rosalez_sample-small.jpeg",
        textract_features=["TABLES", "LAYOUT"])
    documents = loader.load()
    ```

    it will generate output that formats the text in reading order and
    try to output the information in a tabular structure or
    output the key/value pairs with a colon (key: value).
    This helps most LLMs to achieve better accuracy when
    processing these texts.

    """

    def __init__(
            self,
            textract_features: Optional[Sequence[int]] = None,
            client: Optional[Any] = None,
            *,
            linearization_config: Optional["TextLinearizationConfig"] = None,
    ) -> None:
        """Initializes the parser.

        Args:
            textract_features: Features to be used for extraction, each feature
                               should be passed as an int that conforms to the enum
                               `Textract_Features`, see `amazon-textract-caller` pkg
            client: boto3 textract client
            linearization_config: Config to be used for linearization of the output
                                  should be an instance of TextLinearizationConfig from
                                  the `textractor` pkg
        """

        try:
            import textractcaller as tc
            import textractor.entities.document as textractor

            self.tc = tc
            self.textractor = textractor

            if textract_features is not None:
                self.textract_features = [
                    tc.Textract_Features(f) for f in textract_features
                ]
            else:
                self.textract_features = []

            if linearization_config is not None:
                self.linearization_config = linearization_config
            else:
                self.linearization_config = self.textractor.TextLinearizationConfig(
                    hide_figure_layout=True,
                    title_prefix="# ",
                    section_header_prefix="## ",
                    list_element_prefix="*",
                )
        except ImportError:
            raise ImportError(
                "Could not import amazon-textract-caller or "
                "amazon-textract-textractor python package. Please install it "
                "with `pip install amazon-textract-caller` & "
                "`pip install amazon-textract-textractor`."
            )

        if not client:
            try:
                import boto3

                self.boto3_textract_client = boto3.client("textract")
            except ImportError:
                raise ImportError(
                    "Could not import boto3 python package. "
                    "Please install it with `pip install boto3`."
                )
        else:
            self.boto3_textract_client = client

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Iterates over the Blob pages and returns an Iterator with a Document
        for each page, like the other parsers If multi-page document, blob.path
        has to be set to the S3 URI and for single page docs
        the blob.data is taken
        """

        url_parse_result = urlparse(
            str(blob.path)) if blob.path else None  # type: ignore[attr-defined]
        # Either call with S3 path (multi-page) or with bytes (single-page)
        if (
                url_parse_result
                and url_parse_result.scheme == "s3"
                and url_parse_result.netloc
        ):
            textract_response_json = self.tc.call_textract(
                input_document=str(blob.path),  # type: ignore[attr-defined]
                features=self.textract_features,
                boto3_textract_client=self.boto3_textract_client,
            )
        else:
            textract_response_json = self.tc.call_textract(
                input_document=blob.as_bytes(),  # type: ignore[attr-defined]
                features=self.textract_features,
                call_mode=self.tc.Textract_Call_Mode.FORCE_SYNC,
                boto3_textract_client=self.boto3_textract_client,
            )

        document = self.textractor.Document.open(textract_response_json)

        for idx, page in enumerate(document.pages):
            yield Document(
                page_content=page.get_text(config=self.linearization_config),
                metadata={"source": blob.source, "page": idx + 1},
                # type: ignore[attr-defined]
            )


class DocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Form Recognizer) and chunks at character level."""

    def __init__(self, client: Any, model: str):
        warnings.warn(
            "langchain_community.document_loaders.parsers.pdf.DocumentIntelligenceParser"
            "and langchain_community.document_loaders.pdf.DocumentIntelligenceLoader"
            " are deprecated. Please upgrade to "
            "langchain_community.document_loaders.DocumentIntelligenceLoader "
            "for any file parsing purpose using Azure Document Intelligence "
            "service."
        )
        self.client = client
        self.model = model

    def _generate_docs(self, blob: Blob, result: Any) -> Iterator[
        Document]:  # type: ignore[valid-type]
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "source": blob.source,  # type: ignore[attr-defined]
                    "page": p.page_number,
                },
            )
            yield d

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:  # type: ignore[attr-defined]
            poller = self.client.begin_analyze_document(self.model, file_obj)
            result = poller.result()

            docs = self._generate_docs(blob, result)

            yield from docs
