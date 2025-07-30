"""
표 구조 감지 모듈
"""

from typing import Dict, List


class TableDetector:
    """표 구조 감지 클래스"""
    
    def __init__(self):
        self.detected_tables: List[Dict] = []
    
    def detect_tables_in_page(self, blocks: List[Dict]) -> List[Dict]:
        """페이지에서 표 구조를 감지합니다."""
        text_blocks_with_pos = self._extract_positioned_blocks(blocks)
        
        if len(text_blocks_with_pos) < 4:
            return []
        
        return self._find_table_candidates(text_blocks_with_pos)
    
    def _extract_positioned_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """위치 정보가 있는 텍스트 블록들을 추출합니다."""
        text_blocks = []
        for block in blocks:
            if "lines" not in block:
                continue
                
            bbox = block["bbox"]
            text_content = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text_content += span["text"] + " "
            
            if text_content.strip():
                text_blocks.append({
                    "bbox": bbox,
                    "text": text_content.strip(),
                    "block": block
                })
        
        return text_blocks
    
    def _find_table_candidates(self, blocks: List[Dict]) -> List[Dict]:
        """블록들에서 표 후보를 찾습니다."""
        tables = []
        
        # Y 좌표로 행 그룹화
        rows = self._group_blocks_by_rows(blocks)
        table_rows = self._filter_table_rows(rows)
        
        if len(table_rows) >= 3:
            table = {
                "bbox": self._calculate_table_bbox(table_rows),
                "rows": table_rows,
                "num_rows": len(table_rows),
                "num_cols": max(len(row) for row in table_rows)
            }
            tables.append(table)
        
        return tables
    
    def _group_blocks_by_rows(self, blocks: List[Dict]) -> Dict:
        """Y 좌표로 블록들을 행별로 그룹화합니다."""
        rows = {}
        for block in blocks:
            y_coord = round(block["bbox"][1] / 5) * 5
            if y_coord not in rows:
                rows[y_coord] = []
            rows[y_coord].append(block)
        return rows
    
    def _filter_table_rows(self, rows: Dict) -> List[List[Dict]]:
        """표 형태의 행들만 필터링합니다."""
        table_rows = []
        for y_coord in sorted(rows.keys()):
            row_blocks = sorted(rows[y_coord], key=lambda x: x["bbox"][0])
            if len(row_blocks) >= 2:
                table_rows.append(row_blocks)
        return table_rows
    
    def _calculate_table_bbox(self, table_rows: List[List[Dict]]) -> List[float]:
        """표의 전체 경계박스를 계산합니다."""
        min_x = min(block["bbox"][0] for row in table_rows for block in row)
        min_y = min(block["bbox"][1] for row in table_rows for block in row)
        max_x = max(block["bbox"][2] for row in table_rows for block in row)
        max_y = max(block["bbox"][3] for row in table_rows for block in row)
        return [min_x, min_y, max_x, max_y]
    
    def is_block_in_table(self, block_bbox: List[float]) -> bool:
        """블록이 감지된 표 중 하나에 속하는지 확인합니다."""
        for table in self.detected_tables:
            table_bbox = table["bbox"]
            if (block_bbox[0] >= table_bbox[0] and block_bbox[1] >= table_bbox[1] and
                block_bbox[2] <= table_bbox[2] and block_bbox[3] <= table_bbox[3]):
                return True
        return False 