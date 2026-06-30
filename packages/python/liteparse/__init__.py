from .parser import LiteParse, search_items
from .types import (
    ExtractedImage,
    LiteParseConfig,
    PageComplexityStats,
    ParseResult,
    ParsedPage,
    TextItem,
    WordBox,
    ScreenshotResult,
    ParseError,
)

__version__ = "2.0.0"
__all__ = [
    "LiteParse",
    "LiteParseConfig",
    "ParseResult",
    "ParsedPage",
    "TextItem",
    "WordBox",
    "ScreenshotResult",
    "PageComplexityStats",
    "ExtractedImage",
    "ParseError",
    "search_items",
]
