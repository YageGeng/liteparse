"""Python-friendly type wrappers around the native Rust bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextItem:
    """Individual text item extracted from a document."""

    text: str
    x: float
    y: float
    width: float
    height: float
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    confidence: Optional[float] = None
    layout_block_id: Optional[int] = None
    layout_label: Optional[str] = None


@dataclass
class LayoutBlock:
    """Detected document layout block on a page."""

    id: int
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


@dataclass
class ParsedPage:
    """A parsed page from a document."""

    page_num: int
    width: float
    height: float
    text: str
    text_items: List[TextItem] = field(default_factory=list)
    layout_blocks: List[LayoutBlock] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing a document."""

    pages: List[ParsedPage]
    text: str

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def get_page(self, page_num: int) -> Optional[ParsedPage]:
        """Get a specific page by number (1-indexed)."""
        for page in self.pages:
            if page.page_num == page_num:
                return page
        return None


@dataclass
class ScreenshotResult:
    """Result of a single page screenshot."""

    page_num: int
    width: int
    height: int
    image_bytes: bytes


@dataclass
class LiteParseConfig:
    """Resolved parser configuration."""

    ocr_language: str
    ocr_enabled: bool
    ocr_server_url: Optional[str]
    tessdata_path: Optional[str]
    max_pages: int
    target_pages: Optional[str]
    dpi: float
    output_format: str
    preserve_very_small_text: bool
    password: Optional[str]
    quiet: bool
    num_workers: int
    layout_enabled: bool
    layout_confidence_threshold: float
    layout_iou_threshold: float
    layout_image_size: int


class ParseError(Exception):
    """Exception raised when parsing fails."""

    pass
