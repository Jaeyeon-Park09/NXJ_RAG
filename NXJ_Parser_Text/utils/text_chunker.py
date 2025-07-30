"""
텍스트 청킹 모듈
"""

import logging
from typing import List, Dict, Any

# LlamaIndex imports for chunking
try:
    from llama_index.core.node_parser import SentenceWindowNodeParser
    from llama_index.core import Document
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    logging.warning("LlamaIndex가 설치되지 않았습니다. 청킹 기능을 사용하려면 'pip install llama-index'를 실행하세요.")


class TextChunker:
    """LlamaIndex를 사용한 텍스트 청킹 클래스"""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.chunker = None
        
        if LLAMAINDEX_AVAILABLE:
            self.chunker = SentenceWindowNodeParser(
                window_size=window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )
            logging.debug(f"텍스트 청킹 초기화: window_size={window_size}")
        else:
            logging.warning("LlamaIndex를 사용할 수 없어 청킹 기능이 비활성화됩니다.")
    
    def chunk_text(self, text: str, source_file: str = "", page_mapping: Dict[str, int] = None) -> List[Dict]:
        """텍스트를 청킹합니다."""
        if not self.chunker or not text.strip():
            logging.debug("청킹 건너뛰기: chunker 없음 또는 빈 텍스트")
            return []
        
        try:
            # Document 객체 생성
            document = Document(
                text=text,
                metadata={"source": source_file}
            )
            
            # 청킹 수행
            nodes = self.chunker.get_nodes_from_documents([document])
            logging.debug(f"청킹 완료: {len(nodes)}개 청크 생성")
            
            # 청크 정보 추출
            chunks = []
            for i, node in enumerate(nodes):
                chunk_info = {
                    "chunk_id": f"chunk_{i+1:03d}",
                    "text": node.text,
                    "metadata": {
                        "source": source_file,
                        "chunk_index": i,
                        "window_size": self.window_size,
                        "char_count": len(node.text),
                        "word_count": len(node.text.split())
                    }
                }
                
                # 페이지 정보 추가
                if page_mapping:
                    chunk_info["metadata"]["page_number"] = self._get_page_number_for_chunk(node.text, page_mapping)
                
                # 윈도우 정보가 있다면 추가
                if hasattr(node, 'metadata') and node.metadata:
                    if "window" in node.metadata:
                        chunk_info["metadata"]["window_text"] = node.metadata["window"]
                    if "original_text" in node.metadata:
                        chunk_info["metadata"]["original_sentence"] = node.metadata["original_text"]
                
                chunks.append(chunk_info)
            
            return chunks
            
        except Exception as e:
            logging.error(f"청킹 처리 중 오류: {str(e)}")
            return []
    
    def _get_page_number_for_chunk(self, chunk_text: str, page_mapping: Dict[str, int]) -> int:
        """청크 텍스트에 해당하는 페이지 번호를 찾습니다."""
        # Markdown 주석에서 페이지 정보 추출 시도
        import re
        page_comment_pattern = r'<!--\s*PAGE_(\d+)\s*-->'
        page_matches = re.findall(page_comment_pattern, chunk_text)
        
        if page_matches:
            # 첫 번째 페이지 주석 사용
            return int(page_matches[0])
        
        # 청크 텍스트의 첫 번째 문장이나 키워드를 사용하여 페이지 매핑에서 찾기
        chunk_start = chunk_text[:50].strip()  # 처음 50자
        
        # 정확한 매칭 시도
        if chunk_start in page_mapping:
            return page_mapping[chunk_start]
        
        # 부분 매칭 시도
        for text_key, page_num in page_mapping.items():
            if chunk_start in text_key or text_key in chunk_start:
                return page_num
        
        # 매칭되지 않으면 기본값 (첫 페이지)
        return 1
    
    def get_chunking_stats(self, chunks: List[Dict]) -> Dict[str, Any]:
        """청킹 통계를 계산합니다."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
                "total_characters": 0,
                "total_words": 0
            }
        
        chunk_lengths = [chunk["metadata"]["char_count"] for chunk in chunks]
        word_counts = [chunk["metadata"]["word_count"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths),
            "total_words": sum(word_counts),
            "window_size": self.window_size
        } 