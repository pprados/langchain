"""Module contains common parsers for PDFs."""
import base64
import html
import logging
import threading
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union, List, Literal, Callable, )
from urllib.parse import urlparse

import numpy as np
import pdfplumber
import pymupdf
import pypdf  # PPR
import pypdfium2
from PIL import Image
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core._api.deprecation import (
    deprecated,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, BasePromptTemplate

if TYPE_CHECKING:
    import pymupdf.pymupdf
    import pdfplumber.page
    from pdfplumber.utils import text, geometry  # import WordExctractor, TextMap
    import pypdf._page
    import pypdfium2._helpers.page
    from pypdf import PageObject
    from textractor.data.text_linearization_config import TextLinearizationConfig

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

_format_image_str = "\n{image_text}\n"
_join_images = "\n"
_default_page_delimitor = "\f"  # PPR: \f ?


def purge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Purge metadata from unwanted keys."""
    new_metadata = {}
    map_key = {
        "page_count": "total_pages",
        "file_path": "source",
    }
    for k, v in metadata.items():
        if k.startswith("/"):
            k = k[1:]
        k = k.lower()
        if k in ["creationdate", "moddate"]:
            try:
                new_metadata[k] = datetime.strptime(
                    v.replace("'", ""),
                    "D:%Y%m%d%H%M%S%z").isoformat("T")
            except ValueError:
                new_metadata[k] = v
        elif k in map_key:
            # Normliaze key with others PDF parser
            new_metadata[map_key[k]] = v
        elif isinstance(v, (str, int)):
            new_metadata[k] = v
    return new_metadata


def _merge_text_and_extras(extra, text_from_page):
    # insert image/table, if possible, between two paragraphs
    if extra:
        sep = "\n\n"
        pos = text_from_page.rfind(sep)
        if pos == -1:
            sep = "\n"
            pos = text_from_page.rfind(sep)
        if pos != -1:
            all_text = (text_from_page[:pos] +
                        sep +
                        extra +
                        text_from_page[pos:])
        else:
            all_text = text_from_page + extra
    else:
        all_text = text_from_page
    return all_text


@deprecated(since="3.0.0",
            alternative="Use Parser.images_to_text()")
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


# Type to change the function to convert images to text.
CONVERT_IMAGE_TO_TEXT = Optional[Callable[
    [Iterable[np.ndarray]], Iterator[str]]]


def convert_images_to_text_with_rapidocr(
        # Default to text format to be compatible with previous versions.
        *,
        format: Literal["text", "markdown", "html"] = "text",
) -> CONVERT_IMAGE_TO_TEXT:
    """
        Return a function to convert images to text using RapidOCR.
        Note: RapidOCR is compatible english and chinese languages.
        Args:
            format: Format of the output text. Either "text" or "markdown".
    """

    def _convert_images_to_text(images: Iterable[np.ndarray]) -> \
            Iterator[str]:
        """Extract text from images.
        Can be overloaded to use another OCR algorithm, or to use
        a multimodal model to describe the images.

        Args:
            images: Images to extract text from.

        Yield:
            Text extracted from each image.

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

        for img in images:
            ocr_result, _ = ocr(img)
            result = ("\n".join([text[1] for text in ocr_result])).strip()
            if result:
                if format == "markdown":
                    result = result.replace("]", r"\\]")
                    result = f'![{result}](.)'
                elif format == "html":
                    result = f'<img alt="{html.escape(result, quote=True)}" />'
            logger.debug(f"RapidOCR text: " + result.replace('\n', '\\n'))
            yield result

    return _convert_images_to_text


def convert_images_to_text_with_tesseract(
        # Default to text format to be compatible with previous versions.
        *,
        format: Literal["text", "markdown", "html"] = "text",
        langs: List[str] = ["eng"],
) -> CONVERT_IMAGE_TO_TEXT:
    """
        Return a function to convert images to text using RapidOCR.
        Note: RapidOCR is compatible english and chinese languages.
        Args:
            format: Format of the output text. Either "text" or "markdown".
    """

    def _convert_images_to_text(images: Iterable[np.ndarray]) -> \
            Iterator[str]:
        """Extract text from images.
        Can be overloaded to use another OCR algorithm, or to use
        a multimodal model to describe the images.

        Args:
            images: Images to extract text from.

        Yield:
            Text extracted from each image.

        Raises:
            ImportError: If `rapidocr-onnxruntime` package is not installed.
        """
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "`pytesseract` package not found, please install it with "
                "`pip install pytesseract`"
            )

        for img in images:
            result = pytesseract.image_to_string(
                img,
                lang="+".join(langs)
            ).strip()
            if result:
                if format == "markdown":
                    result = result.replace("]", r"\\]")
                    result = f'![{result}](.)'
                elif format == "html":
                    result = f'<img alt="{html.escape(result, quote=True)}" />'
            logger.debug(f"Tesseract text: " + result.replace('\n', '\\n'))
            yield result

    return _convert_images_to_text


_prompt_images_to_description = PromptTemplate.from_template(
    """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval \
    and extract all the text from the image."""
)


def convert_images_to_description(
        model: BaseChatModel,
        *,
        prompt: BasePromptTemplate = _prompt_images_to_description,
        format: Literal["text", "markdown", "html"] = "markdown",
) -> CONVERT_IMAGE_TO_TEXT:
    """
    Return a function to convert images to text using a multimodal model.
        Args:
            model: Multimodal model to use to describe the images.
            prompt: Optional prompt to use to describe the images.
            format: Format of the output text. Either "text" or "markdown".

        Returns:
            A function to extract text from images using the multimodal model.
    """

    def _convert_images_to_description(
            images: Iterable[np.ndarray],
    ) -> Iterator[str]:
        """Describe an image and extract text.
        Use a multimodal model to describe the images.

        Args:
            images: Images to extract text from.

        Yield:
            Text extracted from each image.

        Raises:
            ImportError: If `rapidocr-onnxruntime` package is not installed.
        """

        chat = model
        for image in images:  # FIXME: Add a batch processing
            image_bytes = io.BytesIO()
            Image.fromarray(image).save(image_bytes, format="PNG")
            img_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            msg = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt.format()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                        ]
                    )
                ]
            )
            result = msg.content
            if result:
                if format == "markdown":
                    result = result.replace("]", r"\\]")
                    result = f'![{result}](.)'
                elif format == "html":
                    result = f'<img alt="{html.escape(result, quote=True)}" />'
            logger.debug(f"LLM description: " + result.replace('\n', '\\n'))
            yield result

    return _convert_images_to_description


class ImagesPdfParser(BaseBlobParser):
    """Abstract interface for blob parsers with OCR."""

    def __init__(self,
                 extract_images: bool,
                 images_to_text: CONVERT_IMAGE_TO_TEXT,
                 ):
        """Extract text from images.

        Args:
            extract_images: Whether to extract images from PDF.
            images_to_text: Function to extract text from images.
        """
        self.extract_images = extract_images
        self.convert_image_to_text = (images_to_text
                                      or convert_images_to_text_with_rapidocr())


class PyPDFParser(ImagesPdfParser):
    """Load `PDF` using `pypdf`"""

    def __init__(
            self,
            password: Optional[Union[str, bytes]] = None,
            extract_images: bool = False,
            *,  # Move on top ?
            mode: Literal["single", "paged"] = "paged",
            pages_delimitor: str = _default_page_delimitor,
            images_to_text: CONVERT_IMAGE_TO_TEXT = None,
            extract_tables: Optional[Literal["markdown"]] = None,  # FIXME

            extraction_mode: Literal["plain", "layout"] = "plain",
            extraction_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a parser based on PyPDF.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "paged".
            pages_delimitor: Delimitor to use between pages.
            May be <!--PAGE BREAK -->
            extract_images: Whether to extract images from PDF.
            images_to_text: Function to extract text from images.

            extraction_mode: PyPDF extraction mode to use. Either "plain" or "layout".
            extraction_kwargs: Keyword arguments to pass to the PyPDF extraction method.
        """
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )
        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
        self.password = password
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.extraction_mode = extraction_mode
        self.extraction_kwargs = extraction_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob.
        Insert image, if possible, between two paragraphs.
        In this way, a paragraph can be continued on the next page.
        """

        def _extract_text_from_page(page: "PageObject") -> str:
            """
            Extract text from image given the version of pypdf.
            """

            def before(operator, operand_arguments,
                       current_transformation_matrix, text_matrix):
                pass

            def after(operator, operand_arguments,
                      current_transformation_matrix, text_matrix):
                pass

            def text(text, current_transformation_matrix,
                     text_matrix, font_dictionary, font_size):
                pass

            if pypdf.__version__.startswith("3"):
                return page.extract_text()
            else:
                return page.extract_text(
                    extraction_mode=self.extraction_mode, **self.extraction_kwargs,
                    visitor_operand_before=before,
                    visitor_operand_after=after,
                    visitor_text=text,
                )

        with (blob.as_bytes_io() as pdf_file_obj):  # type: ignore[attr-defined]
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)

            doc_metadata = purge_metadata(
                pdf_reader.metadata | {
                    "source": blob.source,
                    "total_pages": len(pdf_reader.pages),
                })
            single_texts = []
            for page_number, page in enumerate(pdf_reader.pages):
                text_from_page = _extract_text_from_page(page=page)
                image_from_page = self.extract_images_from_page(page)
                all_text = _merge_text_and_extras(image_from_page, text_from_page)
                if self.mode == "paged":
                    yield Document(
                        page_content=all_text,
                        metadata=doc_metadata | {"page": page_number},
                    )
                else:
                    single_texts.append(all_text)
            if self.mode == "single":
                yield Document(
                    page_content=self.pages_delimitor.join(single_texts),
                    metadata=doc_metadata
                )

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
                    images.append(
                        np.array(Image.open(io.BytesIO(xObject[obj].get_data()))))

                else:
                    warnings.warn("Unknown PDF Filter!")
        return _format_image_str.format(
            image_text=_join_images.join(
                [text for text in self.convert_image_to_text(images)])
        )


from pdfminer.converter import *
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import *


class PDFMinerParser(ImagesPdfParser):
    """Parse `PDF` using `PDFMiner`."""

    def __init__(self,
                 extract_images: bool = False,
                 *,
                 password: Optional[str] = None,
                 mode: Literal["single", "paged"] = "single",
                 pages_delimitor: str = _default_page_delimitor,
                 images_to_text: CONVERT_IMAGE_TO_TEXT = None,

                 concatenate_pages: Optional[bool] = None,
                 ):
        """Initialize a parser based on PDFMiner.

        Args:
            password: Password to open the PDF.
            extract_images: Whether to extract images from PDF.
            images_to_text: Function to extract text from images.
            mode: Extraction mode to use. Either "single" or "paged".
            pages_delimitor: Delimitor to use between pages.
            concatenatef_pages: Deprecated. If True, concatenate all PDF pages into one a single
                               document. Otherwise, return one document per page.
        """
        try:
            import pdfminer  # noqa:F401
        except ImportError:
            raise ImportError(
                "pdfminer package not found, please install it with `pip install pdfminer`"
            )
        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.password = password
        self.extract_images = extract_images
        self.images_to_text = images_to_text
        self.pages_delimitor = pages_delimitor
        if extract_images:
            logger.warning("To replicate a bug from the previous version, "
                           "force the mode to 'paged'")  # PPR
            self.mode = "paged"

        if concatenate_pages is not None:
            warnings.warn(
                "`concatenate_pages` parameter is deprecated. "
                "Use `mode='single'` instead."
            )
            mode = "single" if concatenate_pages else "paged"
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
    ) -> Dict[str, Any]:
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
        import io
        with blob.as_bytes_io() as pdf_file_obj, TemporaryDirectory() as tempdir:
            pages = PDFPage.get_pages(pdf_file_obj, password=self.password)
            rsrcmgr = PDFResourceManager()
            doc_metadata = purge_metadata(
                self._get_metadata(pdf_file_obj, password=self.password))
            doc_metadata["source"] = blob.source

            class Visitor(PDFLayoutAnalyzer):
                def __init__(
                        self,
                        rsrcmgr: PDFResourceManager,
                        pageno: int = 1,
                        laparams: Optional[LAParams] = None,
                ) -> None:
                    super().__init__(rsrcmgr,
                                     pageno=pageno,
                                     laparams=laparams)

                def receive_layout(me, ltpage: LTPage) -> None:
                    def render(item: LTItem) -> None:
                        if isinstance(item, LTTextLine):
                            text_io.write("\n")
                        elif isinstance(item, LTTextContainer):
                            text_io.write(item.get_text())
                        elif isinstance(item, LTText):
                            text_io.write(item.get_text())
                        elif isinstance(item, LTImage):
                            if self.extract_images and self.images_to_text:
                                from pdfminer.image import ImageWriter
                                image_writer = ImageWriter(tempdir)
                                filename = image_writer.export_image(item)
                                img = np.array(Image.open(Path(tempdir) / filename))
                                image_text = next(self.images_to_text([img]))
                                if image_text:
                                    text_io.write(_format_image_str.format(
                                        image_text=image_text)
                                    )
                        elif isinstance(item, LTContainer):
                            for child in item:
                                render(child)

                    render(ltpage)

            text_io = io.StringIO()
            visitor_for_all = PDFPageInterpreter(rsrcmgr,
                                                 Visitor(rsrcmgr, laparams=LAParams()))
            all_content = []
            for i, page in enumerate(pages):
                text_io.truncate(0)
                text_io.seek(0)
                visitor_for_all.process_page(page)

                content = text_io.getvalue()
                if self.mode == "paged":
                    text_io.truncate(0)
                    text_io.seek(0)
                    yield Document(page_content=content,
                                   metadata=doc_metadata | {"page": i})
                else:
                    all_content.append(content)
            if self.mode == "single":
                yield Document(
                    page_content=self.pages_delimitor.join(all_content),
                    metadata=doc_metadata)


class PyMuPDFParser(ImagesPdfParser):
    """Parse `PDF` using `PyMuPDF`."""

    # PyMuPDF is not thread safe.
    # See https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
    _lock = threading.Lock()

    from pymupdf.table import (
        DEFAULT_SNAP_TOLERANCE,
        DEFAULT_JOIN_TOLERANCE,
        DEFAULT_MIN_WORDS_VERTICAL,
        DEFAULT_MIN_WORDS_HORIZONTAL)

    _default_extract_tables_settings = {
        "clip": None,
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "vertical_lines": None,
        "horizontal_lines": None,
        "snap_tolerance": DEFAULT_SNAP_TOLERANCE,
        "snap_x_tolerance": None,
        "snap_y_tolerance": None,
        "join_tolerance": DEFAULT_JOIN_TOLERANCE,
        "join_x_tolerance": None,
        "join_y_tolerance": None,
        "edge_min_length": 3,
        "min_words_vertical": DEFAULT_MIN_WORDS_VERTICAL,
        "min_words_horizontal": DEFAULT_MIN_WORDS_HORIZONTAL,
        "intersection_tolerance": 3,
        "intersection_x_tolerance": None,
        "intersection_y_tolerance": None,
        "text_tolerance": 3,
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
        "strategy": None,  # offer abbreviation
        "add_lines": None,  # optional user-specified lines

    }

    def __init__(
            self,
            *,
            password: Optional[str] = None,
            mode: Literal["single", "paged"] = "paged",
            pages_delimitor: str = _default_page_delimitor,
            extract_images: bool = False,
            images_to_text: CONVERT_IMAGE_TO_TEXT = None,
            extract_tables: Optional[Literal["markdown"]] = None,

            extract_tables_settings: Optional[Dict[str, Any]] = None,
            text_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the parser.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "paged".
            pages_delimitor: Delimitor to use between pages.
            extract_images: Whether to extract images from PDF.
            images_to_text: Function to extract text from images.
            extract_tables_settings: Whether to extract tables from PDF.
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        try:
            import pymupdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pymupdf package not found, please install it with `pip install pymupdf`"
            )

        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
        if extract_tables and extract_tables not in ["markdown"]:
            raise ValueError("mode must be markdown")

        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.password = password  # PPR: https://github.com/pymupdf/RAG/pull/170
        self.text_kwargs = text_kwargs or {}
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_tables_settings = (
                extract_tables_settings or
                PyMuPDFParser._default_extract_tables_settings)

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""

        import pymupdf

        with PyMuPDFParser._lock:  # PPR: toto integration images
            with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
                if blob.data is None:  # type: ignore[attr-defined]
                    doc = pymupdf.open(file_path)
                else:
                    doc = pymupdf.open(stream=file_path, filetype="pdf")
                if doc.is_encrypted:
                    doc.authenticate(self.password)
                doc_metadata = self._extract_metadata(doc, blob)
                full_content = []
                for page in doc:
                    text_from_page = self._get_page_content(doc, page, blob)
                    image_from_page = ""  # PPR: todo
                    all_text = _merge_text_and_extras(image_from_page,
                                                      text_from_page)
                    if self.mode == "paged":
                        yield Document(
                            page_content=all_text,
                            metadata=(doc_metadata |
                                      {"page": page.number}),
                        )
                    else:
                        full_content.append(all_text)

                if self.mode == "single":
                    yield Document(
                        page_content=self.pages_delimitor.join(full_content),
                        metadata=doc_metadata
                    )

    def _get_page_content(
            self, doc: pymupdf.pymupdf.Document, page: pymupdf.pymupdf.Page, blob: Blob
    ) -> str:
        """
        Get the text of the page using PyMuPDF and RapidOCR and issue a warning
        if it is empty.
        """

        content = (page.get_text(**self.text_kwargs) +
                   self._extract_tables_from_page(page) +  # PPR
                   self._extract_images_from_page(doc, page)
                   )

        if not content:
            warnings.warn(
                f"Warning: Empty content on page "
                f"{page.number} of document {blob.source}"
            )

        return content

    def _extract_metadata(
            self, doc: pymupdf.pymupdf.Document, blob: Blob
    ) -> dict:
        """Extract metadata from the document and page."""
        return purge_metadata(dict(
            {
                "source": blob.source,  # type: ignore[attr-defined]
                "file_path": blob.source,  # type: ignore[attr-defined]
                "total_pages": len(doc),
            },
            **{
                k: doc.metadata[k]
                for k in doc.metadata
                if isinstance(doc.metadata[k], (str, int))
            },
        ))

    def _extract_images_from_page(
            self, doc: pymupdf.pymupdf.Document, page: pymupdf.pymupdf.Page
    ) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""
        import pymupdf

        img_list = page.get_images()
        images = []
        for img in img_list:
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            images.append(
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, -1
                )
            )
        return _format_image_str.format(
            image_text=_join_images.join(
                [text for text in self.convert_image_to_text(images)])
        )

    def _extract_tables_from_page(
            self, page: pymupdf.pymupdf.Page
    ) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if self.extract_tables is None:
            return ""
        import pymupdf

        tables_list = pymupdf.table.find_tables(page, **self.extract_tables_settings)
        if self.extract_tables == "markdown":
            return "\n\n".join([table.to_markdown() for table in tables_list])
        return ""


class PyPDFium2Parser(ImagesPdfParser):
    """Parse `PDF` with `PyPDFium2`."""

    # PyPDFium2 is not thread safe.
    # See https://pypdfium2.readthedocs.io/en/stable/python_api.html#thread-incompatibility
    _lock = threading.Lock()

    def __init__(self,
                 extract_images: bool = False,
                 *,
                 password: Optional[str] = None,
                 mode: Literal["single", "paged"] = "paged",
                 pages_delimitor: str = _default_page_delimitor,
                 images_to_text: CONVERT_IMAGE_TO_TEXT = None,

                 ) -> None:
        """Initialize a parser based on PyPDFium2.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "paged".
            pages_delimitor: Delimitor to use between pages.
            extract_images: Whether to extract images from PDF.
            images_to_text: Function to extract text from images.
        """
        try:
            import pypdfium2  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )
        super().__init__(extract_images, images_to_text)
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.password = password

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pypdfium2

        # pypdfium2 is really finicky with respect to closing things,
        # if done incorrectly creates seg faults.
        with PyPDFium2Parser._lock:  # TODO: images
            with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
                pdf_reader = pypdfium2.PdfDocument(file_path,
                                                   password=self.password,
                                                   autoclose=True)
                full_content = []

                doc_metadata = purge_metadata(pdf_reader.get_metadata_dict())
                doc_metadata["source"] = blob.source
                doc_metadata["total_pages"] = len(pdf_reader)

                try:
                    for page_number, page in enumerate(pdf_reader):
                        text_page = page.get_textpage()
                        text_from_page = "\n".join(
                            text_page.get_text_range().splitlines())  # Replace \r\n
                        text_page.close()
                        image_from_page = self._extract_images_from_page(page)
                        all_text = _merge_text_and_extras(image_from_page,
                                                          text_from_page)
                        page.close()

                        if self.mode == "paged":
                            yield Document(page_content=all_text,
                                           metadata={**doc_metadata,
                                                     **{
                                                         "page": page_number,
                                                     }})
                        else:
                            full_content.append(all_text)

                    if self.mode == "single":
                        yield Document(
                            page_content=self.pages_delimitor.join(full_content),
                            metadata=doc_metadata)
                finally:
                    pdf_reader.close()

    def _extract_images_from_page(self, page: pypdfium2._helpers.page.PdfPage) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""

        import pypdfium2.raw as pdfium_c

        images = list(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)))

        images = list(map(lambda x: x.get_bitmap().to_numpy(), images))
        return _format_image_str.format(
            image_text=_join_images.join(
                [text for text in self.convert_image_to_text(images)])
        )


class PDFPlumberParser(ImagesPdfParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(
            self,
            text_kwargs: Optional[Mapping[str, Any]] = None,
            dedupe: bool = False,
            extract_images: bool = False,
            *,
            password: Optional[str] = None,
            mode: Literal["single", "paged"] = "paged",
            pages_delimitor: str = _default_page_delimitor,
            images_to_text: CONVERT_IMAGE_TO_TEXT = None,
            extract_tables: Optional[Literal["csv", "markdown", "html"]] = None,
            extract_tables_settings: Optional[Dict[str, Any]] = None,

    ) -> None:
        """Initialize the parser.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "paged".
            extract_images: Whether to extract images from PDF.
            images_to_text: Function to extract text from images.
            extract_tables: Whether to extract tables from PDF.

            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        try:
            import pdfplumber  # noqa:F401
        except ImportError:
            raise ImportError(
                "pdfplumber package not found, please install it with `pip install pdfplumber`"
            )

        super().__init__(extract_images, images_to_text)
        self.password = password
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
        if extract_tables and extract_tables not in ["csv", "markdown", "html"]:
            raise ValueError("mode must be csv, markdown or html")
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.dedupe = dedupe
        self.text_kwargs = text_kwargs or {}
        self.extract_tables = extract_tables
        self.extract_tables_settings = extract_tables_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_y_tolerance": 5,
            "intersection_x_tolerance": 15,
        }

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            doc = pdfplumber.open(file_path,
                                  password=self.password)  # open document
            # TODO: avec ou sans tables
            from pdfplumber.utils import geometry  # import WordExctractor, TextMap
            contents = []
            tables_as_html = []
            images = []
            doc_metadata = ({
                                k: doc.metadata[k]
                                for k in doc.metadata
                                if type(doc.metadata[k]) in [str, int]
                            } |
                            {
                                "source": blob.source,  # type: ignore[attr-defined]
                                "file_path": blob.source,  # type: ignore[attr-defined]
                                "total_pages": len(doc.pages),
                            })
            doc_metadata = purge_metadata(doc_metadata)
            for page in doc.pages:
                tables_bbox: List[Tuple[
                    float, float, float, float]] = self._extract_tables_bbox_from_page(
                    page)
                tables_content = self._extract_tables_from_page(page)
                images_bbox = [geometry.obj_to_bbox(image) for image in page.images]
                image_from_page = self._extract_images_from_page(page)
                if self.dedupe:
                    text_from_page = page.dedupe_chars().extract_text(
                        **self.text_kwargs)
                else:
                    text_from_page = page.extract_text(**self.text_kwargs)

                all_text = _merge_text_and_extras(image_from_page,
                                                  text_from_page)

                if self.mode == "paged":
                    yield Document(
                        page_content=all_text,
                        metadata=(doc_metadata |
                                  {
                                      "page": page.page_number - 1,
                                  }),
                    )
                else:
                    contents.append(all_text)
                # "tables_as_html": [self._convert_table_to_html(table)
                #                    for
                #                    table in tables_content],
                # "images": images_content,
                # tables_as_html.extend([self._convert_table(table)
                #                        for
                #                        table in tables_content])
            if self.mode == "single":
                yield Document(
                    page_content=self.pages_delimitor.join(contents),
                    metadata=doc_metadata,
                )

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        return page.extract_text(**self.text_kwargs)

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
                        # PPR print(f"  saute yield sur tableau deja vu")
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

    def _extract_images_from_page(
            self,
            page: pdfplumber.page.Page) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""

        images = []
        for img in page.images:
            if img["stream"]["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    np.frombuffer(
                        img["stream"].get_data(),
                        dtype=np.uint8).reshape(
                        img["stream"]["Height"], img["stream"]["Width"], -1
                    )
                )
            elif img["stream"]["Filter"].name in _PDF_FILTER_WITH_LOSS:
                buf = np.frombuffer(
                    img["stream"].get_data(),
                    dtype=np.uint8)
                images.append(np.array(Image.open(io.BytesIO(buf))))
            else:
                warnings.warn("Unknown PDF Filter!")

        return _format_image_str.format(
            image_text=_join_images.join(
                [text for text in self.convert_image_to_text(images)])
        )

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


# %% Add on PPR
class PyMuPDF4LLMParser(ImagesPdfParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(
            self,
            *,
            password: Optional[str] = None,
            mode: Literal["single", "paged"] = "paged",
            pages_delimitor: str = _default_page_delimitor,
            to_markdown_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the parser.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "paged".
            pages_delimitor: Delimitor to use between pages.

            to_markdown_kwargs: Keyword arguments to pass to the PyMuPDF4LLM
             extraction method.
        """
        # self.password = password
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
        self.pages_delimitor = pages_delimitor
        _to_markdown_kwargs = to_markdown_kwargs or {}
        if not _to_markdown_kwargs.get("password"):
            _to_markdown_kwargs["password"] = password
        if mode == "single":
            if "page_chunks" in _to_markdown_kwargs:
                _to_markdown_kwargs.pop("page_chunks")  # FIXME: page_delimiter
        elif mode == "paged":
            _to_markdown_kwargs["page_chunks"] = True
        else:
            raise ValueError("mode must be single or paged")
        self.to_markdown_kwargs = _to_markdown_kwargs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        try:
            import pymupdf4llm  # noqa:F401
        except ImportError:
            raise ImportError(
                "pymupdf4llm package not found, please install it with `pip install pymupdf4llm`"
            )
        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            if blob.data is None:  # type: ignore[attr-defined]
                for mu_doc in pymupdf4llm.to_markdown(
                        file_path,
                        **self.to_markdown_kwargs,
                ):
                    yield Document(
                        page_content=mu_doc['text'],
                        metadata=purge_metadata(mu_doc['metadata'])
                    )
                    # PPR TODO: extraire les images. Voir PyMuPDFParser
                    # PPR TODO: extraire les tableaux ? Voir PyMuPDFParser
            else:
                raise NotImplementedError("stream not implemented")

    _map_key = {
        "page_count": "total_pages",
        "file_path": "source"
    }
    _date_key = [
        "creationdate", "moddate"
    ]


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

    # PPR
    # {"metadata":r"regex"},
    # doc_regex = r"regex"
    def __init__(
            self,
            routes: List[
                Tuple[
                    Optional[Union[re, str]],
                    Optional[Union[re, str]],
                    Optional[Union[re, str]],
                    BaseBlobParser]],
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
        for producer, creator, page, parser in routes:
            if isinstance(producer, str):
                producer = re.compile(producer)
            if isinstance(creator, str):
                creator = re.compile(creator)
            if isinstance(page, str):
                page = re.compile(page)
            new_routes.append((producer, creator, page, parser))
        self.routes = new_routes

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        from pypdf import PdfReader

        with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]

            with PdfReader(pdf_file_obj,
                           password=self.password) as reader:
                if reader.metadata:
                    producer, creator = reader.metadata.producer, reader.metadata.creator
                    page1 = reader.pages[0].extract_text()
                for re_producer, re_create, re_page, parser in self.routes:
                    is_producer = (not re_producer
                                   or re_producer.search(producer))
                    is_creator = (not re_create
                                  or re_create.search(creator))
                    is_page = (not re_page
                               or re_page.search(page1))
                    if is_producer and is_creator and is_page:
                        yield from parser.lazy_parse(blob)
