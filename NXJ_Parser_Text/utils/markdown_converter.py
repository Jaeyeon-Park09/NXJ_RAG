"""
Markdown 변환 모듈
"""

from typing import List, Dict, Tuple, Set
from data_classes import TextBlock


class MarkdownConverter:
    """텍스트 블록을 Markdown으로 변환하는 클래스"""
    
    def __init__(self, tables: List[Dict]):
        self.tables = tables
    
    def convert_blocks_to_markdown(self, text_blocks: List[TextBlock]) -> str:
        """텍스트 블록들을 Markdown으로 변환합니다."""
        markdown_lines = []
        processed_table_blocks = set()
        table_blocks_map = self._group_table_blocks(text_blocks)
        
        for i, block in enumerate(text_blocks):
            if id(block) in processed_table_blocks:
                continue
            
            # 페이지 정보를 포함한 블록 처리
            if block.block_type == "heading":
                markdown_lines.extend(self._convert_heading(block))
            elif block.block_type == "list_item":
                markdown_lines.append(f"- {block.text}")
            elif block.block_type == "table_cell":
                table_markdown, processed_ids = self._convert_table_block(block, table_blocks_map)
                markdown_lines.extend(table_markdown)
                processed_table_blocks.update(processed_ids)
            else:
                markdown_lines.extend([block.text, ""])
        
        return "\n".join(markdown_lines)
    
    def _group_table_blocks(self, text_blocks: List[TextBlock]) -> Dict:
        """표별로 블록들을 그룹화합니다."""
        table_blocks_map = {i: [] for i in range(len(self.tables))}
        
        for block in text_blocks:
            if not block.is_in_table:
                continue
            
            for table_idx, table in enumerate(self.tables):
                if self._block_in_specific_table(block.bbox, table):
                    table_blocks_map[table_idx].append(block)
                    break
        
        return table_blocks_map
    
    def _block_in_specific_table(self, block_bbox: List[float], table: Dict) -> bool:
        """블록이 특정 표에 속하는지 확인합니다."""
        table_bbox = table["bbox"]
        return (block_bbox[0] >= table_bbox[0] and block_bbox[1] >= table_bbox[1] and
                block_bbox[2] <= table_bbox[2] and block_bbox[3] <= table_bbox[3])
    
    def _convert_heading(self, block: TextBlock) -> List[str]:
        """제목 블록을 변환합니다."""
        level = min(block.level, 6)
        # 페이지 정보를 주석으로 추가
        page_comment = f"<!-- PAGE_{block.page_num} -->"
        return [page_comment, f"{'#' * level} {block.text}", ""]
    
    def _convert_table_block(self, block: TextBlock, table_blocks_map: Dict) -> Tuple[List[str], Set]:
        """표 블록을 변환합니다."""
        for table_idx, table in enumerate(self.tables):
            if self._block_in_specific_table(block.bbox, table):
                table_blocks = table_blocks_map[table_idx]
                markdown_lines = self._create_table_markdown(table, table_blocks)
                processed_ids = {id(tb) for tb in table_blocks}
                return markdown_lines, processed_ids
        
        return [f"**{block.text}**", ""], {id(block)}
    
    def _create_table_markdown(self, table: Dict, table_blocks: List[TextBlock]) -> List[str]:
        """표를 Markdown 형식으로 생성합니다."""
        if not table_blocks:
            return []
        
        markdown_lines = ["**[표 시작]**", ""]
        
        # 행별 그룹화
        rows_by_y = {}
        for block in table_blocks:
            y_coord = round(block.bbox[1] / 5) * 5
            if y_coord not in rows_by_y:
                rows_by_y[y_coord] = []
            rows_by_y[y_coord].append(block)
        
        # Markdown 표 생성
        sorted_rows = sorted(rows_by_y.keys())
        for i, y_coord in enumerate(sorted_rows):
            row_blocks = sorted(rows_by_y[y_coord], key=lambda x: x.bbox[0])
            row_texts = [block.text.replace("|", "\\|") for block in row_blocks]
            
            if len(row_texts) > 1:
                markdown_lines.append("| " + " | ".join(row_texts) + " |")
                if i == 0:  # 헤더 구분선
                    markdown_lines.append("| " + " | ".join(["---"] * len(row_texts)) + " |")
            else:
                markdown_lines.append(f"**{row_texts[0]}**")
        
        markdown_lines.extend(["", "**[표 끝]**", ""])
        return markdown_lines 