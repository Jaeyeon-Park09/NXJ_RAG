"""
Contextual Compression Retriever Implementation

LangChain의 ContextualCompressionRetriever를 사용하여 압축 기능이 포함된 리트리버를 구현합니다.
검색된 문서를 DocumentCompressorPipeline으로 압축한 후 LLM에 전달할 수 있도록 처리합니다.
"""

import logging
from typing import List

from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import BaseRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain.schema import Document

logger = logging.getLogger(__name__)


def build_compressed_retriever(
    base_retriever: BaseRetriever,
    compressor: BaseDocumentCompressor
) -> ContextualCompressionRetriever:
    """
    압축 기능이 포함된 ContextualCompressionRetriever를 구축합니다.
    
    Args:
        base_retriever: 기본 리트리버 (EnsembleRetriever 또는 호환되는 BaseRetriever)
        compressor: 문서 압축기 (DocumentCompressorPipeline 또는 BaseDocumentCompressor)
        
    Returns:
        ContextualCompressionRetriever: 압축 기능이 포함된 리트리버
        
    Raises:
        ValueError: 입력 파라미터가 유효하지 않은 경우
    """
    try:
        # 입력 검증
        if base_retriever is None:
            raise ValueError("base_retriever는 None일 수 없습니다")
        
        if compressor is None:
            raise ValueError("compressor는 None일 수 없습니다")
        
        if not isinstance(base_retriever, BaseRetriever):
            raise ValueError("base_retriever는 BaseRetriever의 인스턴스여야 합니다")
        
        if not isinstance(compressor, BaseDocumentCompressor):
            raise ValueError("compressor는 BaseDocumentCompressor의 인스턴스여야 합니다")
        
        # ContextualCompressionRetriever 생성
        compressed_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )
        
        logger.info("ContextualCompressionRetriever가 성공적으로 구성되었습니다")
        logger.info(f"기본 리트리버 타입: {type(base_retriever).__name__}")
        logger.info(f"압축기 타입: {type(compressor).__name__}")
        
        return compressed_retriever
        
    except Exception as e:
        logger.error(f"ContextualCompressionRetriever 구축 중 오류 발생: {str(e)}")
        raise


def retrieve_with_compression(
    compressed_retriever: ContextualCompressionRetriever,
    query: str
) -> List[Document]:
    """
    압축 기능이 포함된 리트리버를 사용하여 문서를 검색합니다.
    
    Args:
        compressed_retriever: ContextualCompressionRetriever 객체
        query: 검색 질의
        
    Returns:
        List[Document]: 압축된 검색 결과 문서 리스트
    """
    try:
        # 입력 검증
        if not query or not query.strip():
            raise ValueError("질의는 비어있을 수 없습니다")
        
        # 압축 리트리버를 통한 검색 실행
        results = compressed_retriever.get_relevant_documents(query)
        
        logger.info(f"압축 리트리버 검색 완료: {len(results)}개 결과")
        return results
        
    except Exception as e:
        logger.error(f"압축 리트리버 검색 중 오류 발생: {str(e)}")
        raise


def create_full_compressed_retriever(
    ensemble_retriever,
    llm,
    reorder=None
) -> ContextualCompressionRetriever:
    """
    EnsembleRetriever와 DocumentCompressorPipeline을 결합한 완전한 압축 리트리버를 생성합니다.
    
    Args:
        ensemble_retriever: EnsembleRetriever 객체
        llm: LLM 객체 (LLMChainExtractor 생성용)
        reorder: LongContextReorder 객체 (None이면 새로 생성)
        
    Returns:
        ContextualCompressionRetriever: 완전한 압축 리트리버
    """
    try:
        from .compressor_pipeline import create_full_pipeline
        
        # DocumentCompressorPipeline 생성
        compressor_pipeline = create_full_pipeline(llm, reorder)
        
        # ContextualCompressionRetriever 구축
        compressed_retriever = build_compressed_retriever(
            base_retriever=ensemble_retriever,
            compressor=compressor_pipeline
        )
        
        return compressed_retriever
        
    except Exception as e:
        logger.error(f"완전한 압축 리트리버 생성 중 오류 발생: {str(e)}")
        raise


def validate_compressed_retriever(
    compressed_retriever: ContextualCompressionRetriever,
    test_query: str = "테스트 질문"
) -> bool:
    """
    압축 리트리버가 정상적으로 작동하는지 검증합니다.
    
    Args:
        compressed_retriever: 검증할 ContextualCompressionRetriever
        test_query: 테스트용 질문
        
    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 검색 실행
        results = compressed_retriever.get_relevant_documents(test_query)
        
        # 결과 검증
        if not isinstance(results, list):
            logger.error("검색 결과가 리스트가 아닙니다")
            return False
        
        # 결과가 없는 것도 정상적인 경우일 수 있음
        if len(results) == 0:
            logger.warning("검색 결과가 없습니다")
            return True
        
        # 첫 번째 결과의 구조 검증
        first_result = results[0]
        if not hasattr(first_result, 'page_content'):
            logger.error("검색 결과에 page_content 속성이 없습니다")
            return False
        
        if not hasattr(first_result, 'metadata'):
            logger.error("검색 결과에 metadata 속성이 없습니다")
            return False
        
        logger.info(f"압축 리트리버 검증 성공: {len(results)}개 결과 반환")
        return True
        
    except Exception as e:
        logger.error(f"압축 리트리버 검증 중 오류 발생: {str(e)}")
        return False


# 사용 예시 함수 (테스트용)
def example_compressed_retriever_usage():
    """
    ContextualCompressionRetriever 사용 예시
    """
    from langchain.schema import Document
    from langchain_community.llms import OpenAI  # 예시용
    
    try:
        print("=== ContextualCompressionRetriever 사용 예시 ===")
        
        # 샘플 데이터 (실제로는 실제 리트리버와 LLM 사용)
        print("1. EnsembleRetriever 구성 (예시)")
        print("2. DocumentCompressorPipeline 구성 (예시)")
        print("3. ContextualCompressionRetriever 구축")
        
        # 실제 사용 시 아래 코드 사용:
        # from .ensemble_retriever import build_ensemble_retriever
        # from .compressor_pipeline import create_full_pipeline
        # 
        # # EnsembleRetriever 구축
        # ensemble_retriever = build_ensemble_retriever(
        #     faiss_path="/path/to/faiss",
        #     bm25_texts=texts,
        #     bm25_metadatas=metadatas
        # )
        # 
        # # LLM 객체 (외부에서 주어진 것)
        # llm = your_llm_object
        # 
        # # 압축 리트리버 구축
        # compressed_retriever = create_full_compressed_retriever(
        #     ensemble_retriever=ensemble_retriever,
        #     llm=llm
        # )
        # 
        # # 검색 실행
        # query = "검색 질문"
        # results = retrieve_with_compression(compressed_retriever, query)
        
        print("4. 압축 리트리버 검증")
        print("5. 검색 실행")
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_compressed_retriever_usage() 