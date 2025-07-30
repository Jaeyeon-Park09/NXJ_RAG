"""
PDF 파싱을 위한 데이터 클래스들
"""

from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class TextBlock:
    """텍스트 블록 정보를 저장하는 데이터 클래스"""
    text: str
    font_size: float
    font_name: str
    bbox: List[float]
    page_num: int
    block_type: str = "paragraph"  # heading, paragraph, list_item, table, table_cell
    level: int = 0  # heading level for markdown
    table_info: Optional[Dict] = None  # 표 관련 추가 정보
    is_in_table: bool = False  # 표 내부 텍스트 여부 