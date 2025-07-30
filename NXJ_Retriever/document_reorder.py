"""
Document Reordering Implementation

LangChain의 LongContextReorder를 사용하여 문서를 중요도 순으로 재정렬합니다.
"""

from typing import List
import logging

from langchain_community.document_transformers import LongContextReorder
from langchain.schema import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field

logger = logging.getLogger(__name__)


class LongContextReorderCompressor(BaseDocumentCompressor):
    """
    LongContextReorder를 BaseDocumentCompressor로 래핑하는 클래스
    """
    
    reorder: LongContextReorder = Field(default_factory=LongContextReorder)
    
    def compress_documents(
        self, 
        documents: List[Document], 
        query: str
    ) -> List[Document]:
        """
        문서를 중요도 순으로 재정렬합니다.
        
        Args:
            documents: 재정렬할 Document 객체 리스트
            query: 질의 (LongContextReorder에서는 사용하지 않음)
            
        Returns:
            List[Document]: 중요도 순으로 재정렬된 Document 객체 리스트
        """
        try:
            # LongContextReorder를 사용하여 문서 재정렬
            reordered_docs = self.reorder.transform_documents(documents)
            
            logger.info(f"문서 재정렬 완료: {len(documents)}개 문서")
            return reordered_docs
            
        except Exception as e:
            logger.error(f"문서 재정렬 중 오류 발생: {str(e)}")
            raise


def create_long_context_reorder() -> LongContextReorderCompressor:
    """
    LongContextReorderCompressor 객체를 생성합니다.
    
    Returns:
        LongContextReorderCompressor: 생성된 LongContextReorderCompressor 객체
    """
    try:
        # LongContextReorderCompressor 객체 생성
        reorder = LongContextReorderCompressor()
        
        logger.info("LongContextReorderCompressor 객체가 성공적으로 생성되었습니다")
        return reorder
        
    except Exception as e:
        logger.error(f"LongContextReorderCompressor 객체 생성 중 오류 발생: {str(e)}")
        raise


def reorder_documents(documents: List[Document]) -> List[Document]:
    """
    문서 리스트를 중요도 순으로 재정렬합니다.
    
    Args:
        documents: 재정렬할 Document 객체 리스트
        
    Returns:
        List[Document]: 중요도 순으로 재정렬된 Document 객체 리스트
    """
    try:
        # 입력 검증
        if not documents:
            logger.warning("빈 문서 리스트가 입력되었습니다")
            return []
        
        if not isinstance(documents, list):
            raise ValueError("documents는 리스트여야 합니다")
        
        # LongContextReorderCompressor 객체 생성
        reorder = create_long_context_reorder()
        
        # 문서 재정렬 (query는 빈 문자열로 전달)
        reordered_docs = reorder.compress_documents(documents, "")
        
        logger.info(f"문서 재정렬 완료: {len(documents)}개 문서")
        return reordered_docs
        
    except Exception as e:
        logger.error(f"문서 재정렬 중 오류 발생: {str(e)}")
        raise


# 사용 예시 함수 (테스트용)
def example_reorder_usage():
    """
    LongContextReorderCompressor 사용 예시
    """
    from langchain.schema import Document
    
    # 샘플 문서 생성
    sample_documents = [
        Document(page_content="첫 번째 문서의 내용입니다.", metadata={"source": "doc1"}),
        Document(page_content="두 번째 문서의 내용입니다.", metadata={"source": "doc2"}),
        Document(page_content="세 번째 문서의 내용입니다.", metadata={"source": "doc3"}),
        Document(page_content="네 번째 문서의 내용입니다.", metadata={"source": "doc4"}),
        Document(page_content="다섯 번째 문서의 내용입니다.", metadata={"source": "doc5"})
    ]
    
    try:
        print("=== LongContextReorderCompressor 사용 예시 ===")
        print(f"원본 문서 수: {len(sample_documents)}")
        
        # 문서 재정렬
        reordered_docs = reorder_documents(sample_documents)
        
        print(f"재정렬된 문서 수: {len(reordered_docs)}")
        
        # 결과 출력
        for i, doc in enumerate(reordered_docs):
            print(f"문서 {i+1}: {doc.page_content[:50]}...")
            print(f"  메타데이터: {doc.metadata}")
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_reorder_usage() 