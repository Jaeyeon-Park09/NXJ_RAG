"""
PDF 타입 감지 모듈
"""

import fitz
import logging


class PDFTypeDetector:
    """PDF 타입 감지 클래스"""
    
    @staticmethod
    def detect_pdf_type(doc: fitz.Document) -> str:
        """PDF 타입을 감지합니다 (일반/스캔본/이미지포함)"""
        total_text_length = 0
        total_blocks = 0
        image_blocks = 0
        
        sample_pages = min(3, len(doc))
        logging.debug(f"PDF 타입 감지 중... (샘플 페이지: {sample_pages}개)")
        
        for page_num in range(sample_pages):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                total_blocks += 1
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            total_text_length += len(span["text"].strip())
                else:
                    image_blocks += 1
        
        if total_text_length < 100 and image_blocks > total_blocks * 0.5:
            return "scanned"
        elif image_blocks > 0:
            return "mixed"
        else:
            return "text" 