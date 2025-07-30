"""
Document Compressor Pipeline Implementation

LongContextReorder와 LLMChainExtractor를 순차적으로 실행하는 문서 압축 파이프라인을 구현합니다.
"""

import logging
from typing import List

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain.schema import Document

logger = logging.getLogger(__name__)


def build_compressor_pipeline(
    reorder: BaseDocumentCompressor,
    extractor: BaseDocumentCompressor
) -> DocumentCompressorPipeline:
    """
    LongContextReorder와 LLMChainExtractor를 순차적으로 실행하는 문서 압축 파이프라인을 구축합니다.
    
    Args:
        reorder: LongContextReorder 객체 (BaseDocumentCompressor)
        extractor: LLMChainExtractor 객체 (BaseDocumentCompressor)
        
    Returns:
        DocumentCompressorPipeline: 구성된 문서 압축 파이프라인
        
    Raises:
        ValueError: 입력 압축기가 유효하지 않은 경우
    """
    try:
        # 입력 검증
        if reorder is None:
            raise ValueError("reorder 압축기는 None일 수 없습니다")
        
        if extractor is None:
            raise ValueError("extractor 압축기는 None일 수 없습니다")
        
        if not isinstance(reorder, BaseDocumentCompressor):
            raise ValueError("reorder는 BaseDocumentCompressor의 인스턴스여야 합니다")
        
        if not isinstance(extractor, BaseDocumentCompressor):
            raise ValueError("extractor는 BaseDocumentCompressor의 인스턴스여야 합니다")
        
        # 압축기 리스트 구성 (순서: 1. LongContextReorder, 2. LLMChainExtractor)
<<<<<<< HEAD
        compressors = [reorder, extractor]
        
        # DocumentCompressorPipeline 생성
        pipeline = DocumentCompressorPipeline(compressors=compressors)
=======
        transformers = [reorder, extractor]
        
        # DocumentCompressorPipeline 생성
        pipeline = DocumentCompressorPipeline(transformers=transformers)
>>>>>>> 60b74fa (2)
        
        logger.info("문서 압축 파이프라인이 성공적으로 구성되었습니다")
        logger.info("실행 순서: 1. LongContextReorder, 2. LLMChainExtractor")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"문서 압축 파이프라인 구축 중 오류 발생: {str(e)}")
        raise


def compress_with_pipeline(
    pipeline: DocumentCompressorPipeline,
    documents: List[Document],
    query: str
) -> List[Document]:
    """
    문서 압축 파이프라인을 사용하여 문서를 압축합니다.
    
    Args:
        pipeline: DocumentCompressorPipeline 객체
        documents: 압축할 Document 객체 리스트
        query: 사용자 질의
        
    Returns:
        List[Document]: 압축된 Document 객체 리스트
    """
    try:
        # 입력 검증
        if not documents:
            logger.warning("빈 문서 리스트가 입력되었습니다")
            return []
        
        if not query or not query.strip():
            raise ValueError("질의는 비어있을 수 없습니다")
        
        # 파이프라인을 통한 문서 압축 실행
        compressed_docs = pipeline.compress_documents(documents, query)
        
        logger.info(f"파이프라인 압축 완료: {len(documents)}개 -> {len(compressed_docs)}개")
        return compressed_docs
        
    except Exception as e:
        logger.error(f"파이프라인 압축 중 오류 발생: {str(e)}")
        raise


def create_full_pipeline(
    llm,
    reorder: BaseDocumentCompressor = None
) -> DocumentCompressorPipeline:
    """
    LongContextReorder와 LLMChainExtractor를 포함한 완전한 파이프라인을 생성합니다.
    
    Args:
        llm: LLM 객체 (LLMChainExtractor 생성용)
        reorder: LongContextReorder 객체 (None이면 새로 생성)
        
    Returns:
        DocumentCompressorPipeline: 완전한 문서 압축 파이프라인
    """
    try:
        from .document_reorder import create_long_context_reorder
        from .document_compressor import build_llm_extractor
        
        # LongContextReorder 생성 (없는 경우)
        if reorder is None:
            reorder = create_long_context_reorder()
        
        # LLMChainExtractor 생성
        extractor = build_llm_extractor(llm)
        
        # 파이프라인 구축
        pipeline = build_compressor_pipeline(reorder, extractor)
        
        return pipeline
        
    except Exception as e:
        logger.error(f"완전한 파이프라인 생성 중 오류 발생: {str(e)}")
        raise


# 사용 예시 함수 (테스트용)
def example_pipeline_usage():
    """
    문서 압축 파이프라인 사용 예시
    """
    from langchain.schema import Document
    from langchain_community.llms import OpenAI  # 예시용
    
    # 샘플 문서 생성
    sample_documents = [
        Document(
            page_content="첫 번째 문서의 내용입니다. 이 문서는 기본적인 정보를 포함합니다.",
            metadata={"source": "doc1", "order": 1}
        ),
        Document(
            page_content="두 번째 문서의 내용입니다. 이 문서는 중요한 세부사항을 포함합니다.",
            metadata={"source": "doc2", "order": 2}
        ),
        Document(
            page_content="세 번째 문서의 내용입니다. 이 문서는 추가적인 정보를 제공합니다.",
            metadata={"source": "doc3", "order": 3}
        ),
        Document(
            page_content="네 번째 문서의 내용입니다. 이 문서는 마지막 정보를 포함합니다.",
            metadata={"source": "doc4", "order": 4}
        )
    ]
    
    try:
        print("=== 문서 압축 파이프라인 사용 예시 ===")
        print(f"원본 문서 수: {len(sample_documents)}")
        
        # LLM 객체 생성 (예시용 - 실제로는 외부에서 주어진 LLM 사용)
        # llm = OpenAI(temperature=0)  # 실제 사용 시 주석 해제
        
        # 질의 예시
        query = "중요한 정보를 추출해주세요"
        
        print(f"질의: {query}")
        print("파이프라인을 사용하여 문서를 압축합니다...")
        print("실행 순서: 1. LongContextReorder, 2. LLMChainExtractor")
        
        # 실제 사용 시 아래 코드 사용:
        # from .document_reorder import create_long_context_reorder
        # from .document_compressor import build_llm_extractor
        # 
        # reorder = create_long_context_reorder()
        # extractor = build_llm_extractor(llm)
        # pipeline = build_compressor_pipeline(reorder, extractor)
        # compressed_docs = compress_with_pipeline(pipeline, sample_documents, query)
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_pipeline_usage() 