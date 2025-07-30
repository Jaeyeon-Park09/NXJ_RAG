"""
PDF 파서 메인 모듈 - 모듈화된 구조
"""

import fitz  # PyMuPDF
import json
import re
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict
import logging
from pathlib import Path

# 로컬 유틸리티 모듈 import
from utils import (
    TextBlock, LogManager, PDFTypeDetector, 
    TableDetector, TextChunker, MarkdownConverter
)


class PDFParser:
    """리팩토링된 PDF 파서 메인 클래스"""
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self._reset_state()
    
    def _reset_state(self):
        """파싱 상태를 초기화합니다."""
        self.text_blocks: List[TextBlock] = []
        self.font_sizes: Dict[float, int] = {}
        self.heading_sizes: List[float] = []
        self.is_scanned_pdf: bool = False
        self.table_detector = TableDetector()
        self.text_chunker = TextChunker(window_size=3)
        self.chunks: List[Dict] = []
        self.current_file_name = ""
        self.page_mapping: Dict[str, int] = {}  # 텍스트 -> 페이지 번호 매핑
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 파일을 파싱합니다."""
        # 상태 초기화
        self._reset_state()
        self.current_file_name = Path(pdf_path).name
        logging.info(f"파싱 시작: {self.current_file_name}")
        
        start_time = time.time()
        
        try:
            # 1단계: PDF 열기 및 타입 감지
            self.log_manager.log_progress(1, 7, f"PDF 열기: {self.current_file_name}")
            doc = self._open_and_validate_pdf(pdf_path)
            
            # 2단계: 타입 감지
            self.log_manager.log_progress(2, 7, "PDF 타입 감지 중...")
            pdf_type = PDFTypeDetector.detect_pdf_type(doc)
            logging.info(f"PDF 타입: {pdf_type}")
            
            if pdf_type == "scanned":
                self.is_scanned_pdf = True
                logging.warning(f"스캔본 PDF 감지됨: {pdf_path} - 파싱을 건너뜁니다.")
                doc.close()
                return self._create_empty_result(pdf_path)
            
            # 3단계: 텍스트 블록 추출
            self.log_manager.log_progress(3, 7, "텍스트 블록 추출 중...")
            self.text_blocks = self._extract_text_blocks(doc)
            
            # 4단계: 문서 구조 분석
            self.log_manager.log_progress(4, 7, "문서 구조 분석 중...")
            self._analyze_document_structure()
            
            # 5단계: Markdown 변환
            self.log_manager.log_progress(5, 7, "Markdown 변환 중...")
            markdown_converter = MarkdownConverter(self.table_detector.detected_tables)
            markdown_content = markdown_converter.convert_blocks_to_markdown(self.text_blocks)
            
            # 6단계: 텍스트 청킹
            self.log_manager.log_progress(6, 7, "텍스트 청킹 중...")
            self.chunks = self._perform_chunking(markdown_content)
            
            # 7단계: 결과 생성
            self.log_manager.log_progress(7, 7, "결과 생성 완료")
            result = self._create_result(pdf_path, markdown_content)
            
            elapsed_time = time.time() - start_time
            logging.info(f"파싱 완료: {self.current_file_name} ({elapsed_time:.2f}초)")
            
            doc.close()
            return result
            
        except Exception as e:
            logging.error(f"파싱 오류: {self.current_file_name} - {str(e)}")
            raise
    
    def _open_and_validate_pdf(self, pdf_path: str) -> fitz.Document:
        """PDF 파일을 열고 유효성을 검사합니다."""
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                raise ValueError("빈 PDF 파일")
            logging.debug(f"PDF 열기 성공: {doc.page_count}페이지")
            return doc
        except Exception as e:
            raise ValueError(f"PDF 파일 열기 실패: {str(e)}")
    
    def _create_empty_result(self, pdf_path: str) -> Dict[str, Any]:
        """스캔본 PDF용 빈 결과를 생성합니다."""
        return {
            "file_name": Path(pdf_path).name,
            "total_pages": 0,
            "pdf_type": "scanned",
            "font_analysis": {"font_sizes_frequency": {}, "heading_sizes": []},
            "table_analysis": {"total_tables": 0, "tables_info": []},
            "text_blocks": [],
            "markdown_content": "",
            "chunks": [],
            "chunking_analysis": {
                "total_chunks": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
                "total_characters": 0,
                "total_words": 0,
                "window_size": 3
            },
            "statistics": {
                "total_text_blocks": 0,
                "heading_count": 0,
                "paragraph_count": 0,
                "list_item_count": 0,
                "table_cell_count": 0,
                "tables_detected": 0,
                "chunks_generated": 0
            }
        }
    
    def _extract_text_blocks(self, doc: fitz.Document) -> List[TextBlock]:
        """PDF에서 텍스트 블록들을 추출합니다."""
        text_blocks = []
        total_pages = len(doc)
        
        logging.debug(f"텍스트 블록 추출 시작: {total_pages}페이지")
        
        for page_num in range(total_pages):
            page_progress = f"페이지 {page_num + 1}/{total_pages} 처리 중"
            logging.debug(page_progress)
            
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")
            
            # 페이지별 표 감지
            page_tables = self.table_detector.detect_tables_in_page(blocks["blocks"])
            self.table_detector.detected_tables.extend(page_tables)
            
            # 텍스트 블록 추출
            page_blocks = self._extract_page_blocks(blocks["blocks"], page_num, page_tables)
            text_blocks.extend(page_blocks)
        
        logging.info(f"추출 완료: {len(text_blocks)}개 블록, {len(self.table_detector.detected_tables)}개 표")
        return text_blocks
    
    def _extract_page_blocks(self, blocks: List[Dict], page_num: int, page_tables: List[Dict]) -> List[TextBlock]:
        """페이지에서 텍스트 블록들을 추출합니다."""
        page_blocks = []
        
        for block in blocks:
            if "lines" not in block:  # 이미지 블록 제외
                continue
            
            is_in_table = self.table_detector.is_block_in_table(block["bbox"])
            
            for line in block["lines"]:
                text_block = self._create_text_block_from_line(line, block["bbox"], page_num + 1, is_in_table)
                if text_block:
                    page_blocks.append(text_block)
                    self._update_font_statistics(text_block.font_size)
                    # 페이지 매핑 정보 추가 (첫 번째 텍스트 블록만)
                    if not self.page_mapping or text_block.text not in self.page_mapping:
                        self.page_mapping[text_block.text] = page_num + 1
        
        return page_blocks
    
    def _create_text_block_from_line(self, line: Dict, bbox: List[float], page_num: int, is_in_table: bool) -> Optional[TextBlock]:
        """라인에서 텍스트 블록을 생성합니다."""
        line_text = ""
        font_size = 0
        font_name = ""
        
        for span in line["spans"]:
            line_text += span["text"]
            font_size = span["size"]
            font_name = span["font"]
        
        line_text = line_text.strip()
        if not line_text:
            return None
        
        return TextBlock(
            text=line_text,
            font_size=font_size,
            font_name=font_name,
            bbox=bbox,
            page_num=page_num,
            is_in_table=is_in_table
        )
    
    def _update_font_statistics(self, font_size: float):
        """폰트 크기 통계를 업데이트합니다."""
        self.font_sizes[font_size] = self.font_sizes.get(font_size, 0) + 1
    
    def _perform_chunking(self, markdown_content: str) -> List[Dict]:
        """Markdown 텍스트를 청킹합니다."""
        if not markdown_content.strip():
            logging.debug("청킹 건너뛰기: 빈 텍스트")
            return []
        
        chunks = self.text_chunker.chunk_text(
            markdown_content, 
            self.current_file_name, 
            self.page_mapping
        )
        logging.info(f"청킹 완료: {len(chunks)}개 청크 생성 (페이지 매핑: {len(self.page_mapping)}개)")
        return chunks
    
    def _analyze_document_structure(self) -> None:
        """문서 구조를 분석하여 텍스트 블록 타입을 설정합니다."""
        if not self.font_sizes:
            logging.debug("폰트 정보 없음 - 구조 분석 생략")
            return
        
        # 제목 폰트 크기 결정
        self._determine_heading_sizes()
        
        # 블록 타입 설정
        self._assign_block_types()
        
        logging.debug(f"구조 분석 완료: 제목 크기 {len(self.heading_sizes)}개")
    
    def _determine_heading_sizes(self) -> None:
        """제목으로 사용될 폰트 크기들을 결정합니다."""
        body_font_size = max(self.font_sizes, key=self.font_sizes.get)
        
        potential_headings = [
            size for size in self.font_sizes.keys() 
            if size > body_font_size and self.font_sizes[size] > 1
        ]
        
        self.heading_sizes = sorted(potential_headings, reverse=True)
        logging.debug(f"본문 폰트: {body_font_size}, 제목 폰트: {self.heading_sizes}")
    
    def _assign_block_types(self) -> None:
        """텍스트 블록들의 타입을 설정합니다."""
        for block in self.text_blocks:
            if block.is_in_table:
                block.block_type = "table_cell"
            elif block.font_size in self.heading_sizes:
                block.block_type = "heading"
                block.level = self.heading_sizes.index(block.font_size) + 1
            elif self._is_list_item(block.text):
                block.block_type = "list_item"
            else:
                block.block_type = "paragraph"
    
    def _is_list_item(self, text: str) -> bool:
        """텍스트가 리스트 항목인지 판단합니다."""
        # 한국어 리스트 패턴들
        patterns = [
            r'^[0-9]+\.',           # 1. 2. 3.
            r'^[가-힣]\.',          # 가. 나. 다.
            r'^\([0-9]+\)',         # (1) (2) (3)
            r'^\([가-힣]\)',        # (가) (나) (다)
            r'^[①-⑳]',             # ① ② ③
            r'^[㉠-㉯]',             # ㉠ ㉡ ㉢
            r'^[-•▪▫◦]',           # 불릿 포인트
            r'^○',                  # ○
            r'^●',                  # ●
            r'^◆',                  # ◆
            r'^◇',                  # ◇
        ]
        
        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        return False
    
    def _create_result(self, pdf_path: str, markdown_content: str) -> Dict[str, Any]:
        """파싱 결과 딕셔너리를 생성합니다."""
        return {
            "file_name": Path(pdf_path).name,
            "total_pages": self.text_blocks[-1].page_num if self.text_blocks else 0,
            "pdf_type": "scanned" if self.is_scanned_pdf else "normal",
            "font_analysis": {
                "font_sizes_frequency": self.font_sizes,
                "heading_sizes": self.heading_sizes
            },
            "table_analysis": {
                "total_tables": len(self.table_detector.detected_tables),
                "tables_info": [
                    {
                        "bbox": table["bbox"],
                        "num_rows": table["num_rows"],
                        "num_cols": table["num_cols"]
                    } for table in self.table_detector.detected_tables
                ]
            },
            "text_blocks": [asdict(block) for block in self.text_blocks],
            "markdown_content": markdown_content,
            "chunks": self.chunks,
            "chunking_analysis": self.text_chunker.get_chunking_stats(self.chunks),
            "statistics": self._calculate_statistics()
        }
    
    def _calculate_statistics(self) -> Dict[str, int]:
        """문서 통계를 계산합니다."""
        return {
            "total_text_blocks": len(self.text_blocks),
            "heading_count": len([b for b in self.text_blocks if b.block_type == "heading"]),
            "paragraph_count": len([b for b in self.text_blocks if b.block_type == "paragraph"]),
            "list_item_count": len([b for b in self.text_blocks if b.block_type == "list_item"]),
            "table_cell_count": len([b for b in self.text_blocks if b.block_type == "table_cell"]),
            "tables_detected": len(self.table_detector.detected_tables),
            "chunks_generated": len(self.chunks)
        }
    
    def save_results(self, pdf_path: str, output_dir: str = "output") -> Tuple[str, str]:
        """PDF 파싱 결과를 JSON으로 저장합니다."""
        os.makedirs(output_dir, exist_ok=True)
        
        result = self.parse_pdf(pdf_path)
        
        # 안전한 파일명 생성
        base_name = Path(pdf_path).stem
        safe_name = re.sub(r'[^\w가-힣\-_]', '_', base_name)
        
        # 파일 저장
        json_path = Path(output_dir) / f"{safe_name}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return str(json_path)


def main():
    """리팩토링된 메인 실행 함수"""
    pdf_directory = Path("pdf_files")
    output_directory = Path("output")
    
    # 로그 매니저 초기화
    log_manager = LogManager()
    
    # PDF 파일 목록 가져오기
    pdf_files = list(pdf_directory.glob("*.pdf"))
    
    if not pdf_files:
        logging.error(f"PDF 파일을 찾을 수 없습니다: {pdf_directory}")
        return
    
    logging.info(f"총 {len(pdf_files)}개 PDF 파일 발견")
    print(f"\n🚀 PDF 파싱 시작 - {len(pdf_files)}개 파일")
    print("=" * 50)
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    # 각 PDF 파일 처리
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 [{i}/{len(pdf_files)}] {pdf_file.name}")
        print("-" * 40)
        
        try:
            # 파일별 파서 인스턴스 생성
            parser = PDFParser(log_manager)
            json_path = parser.save_results(str(pdf_file), str(output_directory))
            
            if parser.is_scanned_pdf:
                print("  ⚠️  스캔본 PDF - 파싱 제외")
                logging.warning(f"스캔본 PDF 제외: {pdf_file.name}")
            else:
                print(f"  ✅ JSON: {Path(json_path).name}")
                
                # 통계 정보 출력
                stats = parser._calculate_statistics()
                print(f"  📊 블록: {stats['total_text_blocks']}개")
                print(f"  📊 표: {stats['tables_detected']}개")
                print(f"  🧩 청크: {stats['chunks_generated']}개")
                
                success_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"  ❌ 오류: {str(e)}")
            logging.error(f"파일 처리 실패: {pdf_file.name} - {str(e)}")
    
    # 최종 결과 출력
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("🎯 최종 결과")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {error_count}개")
    print(f"  ⏱️  총 소요시간: {total_time:.2f}초")
    print(f"  📁 로그 파일: {log_manager.log_file}")
    
    logging.info(f"=== 파싱 완료 - 성공: {success_count}, 실패: {error_count}, 시간: {total_time:.2f}초 ===")


if __name__ == "__main__":
    main() 