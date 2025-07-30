"""
Document Compressor Implementation

LangChain의 LLMChainExtractor를 사용하여 문서 압축기를 구현합니다.
사용자 질의와 관련된 핵심 정보를 추출합니다.
"""

import logging
from typing import List

from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


def build_llm_extractor(llm: BaseLanguageModel) -> LLMChainExtractor:
    """
    LLMChainExtractor 객체를 생성합니다.
    
    Args:
        llm: 외부에서 주어진 LLM 객체 (BaseLanguageModel)
        
    Returns:
        LLMChainExtractor: 생성된 LLMChainExtractor 객체
        
    Raises:
        ValueError: LLM 객체가 유효하지 않은 경우
    """
    try:
        # 입력 검증
        if llm is None:
            raise ValueError("LLM 객체는 None일 수 없습니다")
        
        if not isinstance(llm, BaseLanguageModel):
            raise ValueError("LLM 객체는 BaseLanguageModel의 인스턴스여야 합니다")
        
        # LLMChainExtractor 객체 생성
        extractor = LLMChainExtractor.from_llm(llm)
        
        logger.info("LLMChainExtractor 객체가 성공적으로 생성되었습니다")
        return extractor
        
    except Exception as e:
        logger.error(f"LLMChainExtractor 객체 생성 중 오류 발생: {str(e)}")
        raise


def compress_documents(
    extractor: LLMChainExtractor,
    documents: List[Document],
    query: str
) -> List[Document]:
    """
    문서 리스트를 압축하여 질의와 관련된 핵심 정보를 추출합니다.
    
    Args:
        extractor: LLMChainExtractor 객체
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
        
        # 문서 압축 실행
        compressed_docs = extractor.compress_documents(documents, query)
        
        logger.info(f"문서 압축 완료: {len(documents)}개 -> {len(compressed_docs)}개")
        return compressed_docs
        
    except Exception as e:
        logger.error(f"문서 압축 중 오류 발생: {str(e)}")
        raise


def extract_relevant_content(
    llm: BaseLanguageModel,
    documents: List[Document],
    query: str
) -> List[Document]:
    """
    LLM을 사용하여 문서에서 질의와 관련된 내용을 추출합니다.
    
    Args:
        llm: 사용할 LLM 객체
        documents: 추출할 Document 객체 리스트
        query: 사용자 질의
        
    Returns:
        List[Document]: 추출된 Document 객체 리스트
    """
    try:
        # LLMChainExtractor 생성
        extractor = build_llm_extractor(llm)
        
        # 문서 압축 실행
        extracted_docs = compress_documents(extractor, documents, query)
        
        return extracted_docs
        
    except Exception as e:
        logger.error(f"관련 내용 추출 중 오류 발생: {str(e)}")
        raise


# 사용 예시 함수 (테스트용)
def example_extractor_usage():
    """
    LLMChainExtractor 사용 예시
    """
    from langchain.schema import Document
    from langchain_community.llms import OpenAI  # 예시용
    
    # 샘플 문서 생성
    sample_documents = [
        Document(
            page_content="이 문서는 법률 조항에 대한 내용입니다. 법률은 국가의 기본 규칙을 정의합니다.",
            metadata={"source": "doc1", "type": "legal"}
        ),
        Document(
            page_content="절차에 대한 설명입니다. 신청 절차는 다음과 같습니다: 1단계, 2단계, 3단계",
            metadata={"source": "doc2", "type": "procedure"}
        ),
        Document(
            page_content="규정에 대한 내용입니다. 이 규정은 모든 사용자에게 적용됩니다.",
            metadata={"source": "doc3", "type": "regulation"}
        )
    ]
    
    try:
        print("=== LLMChainExtractor 사용 예시 ===")
        print(f"원본 문서 수: {len(sample_documents)}")
        
        # LLM 객체 생성 (예시용 - 실제로는 외부에서 주어진 LLM 사용)
        # llm = OpenAI(temperature=0)  # 실제 사용 시 주석 해제
        
        # 질의 예시
        query = "법률 조항과 절차에 대해 설명해주세요"
        
        print(f"질의: {query}")
        print("LLMChainExtractor를 사용하여 관련 내용을 추출합니다...")
        
        # 실제 사용 시 아래 코드 사용:
        # extractor = build_llm_extractor(llm)
        # extracted_docs = compress_documents(extractor, sample_documents, query)
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_extractor_usage() 