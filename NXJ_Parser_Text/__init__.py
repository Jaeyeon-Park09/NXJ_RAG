"""
NXJ_Parser_Text 패키지

PDF 문서 파싱 및 텍스트 추출을 위한 도구들을 제공합니다.
"""

try:
    from main_parser import PDFParser
from utils import (
        TextBlock, 
        LogManager, 
        PDFTypeDetector, 
        TableDetector, 
        TextChunker, 
        MarkdownConverter
    )
except ImportError:
    # 직접 import 시도
    from main_parser import PDFParser
    from utils import (
        TextBlock, 
        LogManager, 
        PDFTypeDetector, 
        TableDetector, 
        TextChunker, 
        MarkdownConverter
    )

__all__ = [
    "PDFParser",
    "TextBlock",
    "LogManager", 
    "PDFTypeDetector",
    "TableDetector",
    "TextChunker",
    "MarkdownConverter"
] 