"""
Retrieval QA Chain Implementation

LangChain의 RetrievalQA를 사용하여 LLM 객체와 Retriever를 조합합니다.
"stuff" 방식으로 context를 prompt에 직접 삽입하여 문서 기반 질의응답을 수행합니다.
"""

import logging
from typing import Dict, Any, List
from langchain_core.language_models import BaseLanguageModel
from langchain.chains.retrieval_qa.base import BaseRetriever, RetrievalQA

logger = logging.getLogger(__name__)


def build_qa_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever
) -> RetrievalQA:
    """
    LLM 객체와 Retriever를 조합하여 RetrievalQA 체인을 구축합니다.
    
    Args:
        llm: BaseLanguageModel 객체 (command-r:35b 기반)
        retriever: BaseRetriever 객체 (EnsembleRetriever 또는 호환되는 리트리버)
        
    Returns:
        RetrievalQA: 구성된 RetrievalQA 체인
        
    Raises:
        ValueError: 입력 파라미터가 유효하지 않은 경우
    """
    try:
        # 입력 검증
        if llm is None:
            raise ValueError("LLM 객체는 None일 수 없습니다")
        
        if retriever is None:
            raise ValueError("Retriever 객체는 None일 수 없습니다")
        
        if not isinstance(llm, BaseLanguageModel):
            raise ValueError("LLM 객체는 BaseLanguageModel의 인스턴스여야 합니다")
        
        if not isinstance(retriever, BaseRetriever):
            raise ValueError("Retriever 객체는 BaseRetriever의 인스턴스여야 합니다")
        
        # RetrievalQA 체인 생성 ("stuff" 방식)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,  # 소스 문서 반환
            verbose=False  # 로깅 비활성화
        )
        
        logger.info("RetrievalQA 체인이 성공적으로 구성되었습니다")
        logger.info(f"LLM 타입: {type(llm).__name__}")
        logger.info(f"Retriever 타입: {type(retriever).__name__}")
        logger.info("Chain type: stuff")
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"RetrievalQA 체인 구축 중 오류 발생: {str(e)}")
        raise


def run_qa_chain(
    qa_chain: RetrievalQA,
    query: str
) -> Dict[str, Any]:
    """
    RetrievalQA 체인을 실행하여 질의응답을 수행합니다.
    
    Args:
        qa_chain: RetrievalQA 체인 객체
        query: 사용자 질문
        
    Returns:
        Dict[str, Any]: 응답 결과 (answer, source_documents 포함)
    """
    try:
        # 입력 검증
        if not query or not query.strip():
            raise ValueError("질문은 비어있을 수 없습니다")
        
        # 체인 실행
        result = qa_chain({"query": query})
        
        logger.info(f"RetrievalQA 체인 실행 완료: '{query}'")
        return result
        
    except Exception as e:
        logger.error(f"RetrievalQA 체인 실행 중 오류 발생: {str(e)}")
        raise


def create_full_qa_chain(
    model_name: str = "command-r:35b",
    retriever=None
) -> RetrievalQA:
    """
    LLM과 Retriever를 포함한 완전한 RetrievalQA 체인을 생성합니다.
    
    Args:
        model_name: 사용할 모델 이름
        retriever: 외부에서 주어진 Retriever 객체 (None이면 기본 Retriever 생성)
        
    Returns:
        RetrievalQA: 완전한 RetrievalQA 체인
    """
    try:
        from .ollama_llm import build_llm
        
        # LLM 생성
        llm = build_llm(model_name)
        
        # Retriever가 없는 경우 기본 Retriever 생성
        if retriever is None:
            logger.warning("Retriever가 제공되지 않았습니다. 기본 Retriever를 생성합니다.")
            # 여기서는 기본 Retriever 생성 로직을 구현할 수 있지만,
            # 실제로는 외부에서 주어진 Retriever를 사용해야 합니다.
            raise ValueError("Retriever 객체가 필요합니다. 외부에서 제공된 Retriever를 사용하세요.")
        
        # RetrievalQA 체인 구축
        qa_chain = build_qa_chain(llm, retriever)
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"완전한 QA 체인 생성 중 오류 발생: {str(e)}")
        raise


def validate_qa_chain(qa_chain: RetrievalQA, test_query: str = "테스트 질문입니다.") -> bool:
    """
    RetrievalQA 체인이 정상적으로 작동하는지 검증합니다.
    
    Args:
        qa_chain: 검증할 RetrievalQA 체인
        test_query: 테스트용 질문
        
    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 체인 실행
        result = qa_chain({"query": test_query})
        
        # 결과 검증
        if not isinstance(result, dict):
            logger.error("체인 결과가 딕셔너리가 아닙니다")
            return False
        
        if "result" not in result:
            logger.error("체인 결과에 'result' 키가 없습니다")
            return False
        
        answer = result["result"]
        if not answer or not answer.strip():
            logger.warning("체인 응답이 비어있습니다")
            return False
        
        logger.info(f"RetrievalQA 체인 검증 성공: {len(answer)}자 응답")
        return True
        
    except Exception as e:
        logger.error(f"RetrievalQA 체인 검증 중 오류 발생: {str(e)}")
        return False


def get_qa_chain_info(qa_chain: RetrievalQA) -> Dict[str, Any]:
    """
    RetrievalQA 체인 객체의 정보를 반환합니다.
    
    Args:
        qa_chain: 정보를 확인할 RetrievalQA 체인
        
    Returns:
        Dict[str, Any]: 체인 정보 딕셔너리
    """
    try:
        info = {
            "chain_type": type(qa_chain).__name__,
            "llm_type": type(qa_chain.llm).__name__,
            "retriever_type": type(qa_chain.retriever).__name__,
            "chain_type_name": getattr(qa_chain, 'chain_type', 'Unknown'),
            "return_source_documents": getattr(qa_chain, 'return_source_documents', False),
        }
        
        return info
        
    except Exception as e:
        logger.error(f"QA 체인 정보 수집 중 오류 발생: {str(e)}")
        return {"error": str(e)}


def format_qa_response(result: Dict[str, Any]) -> str:
    """
    RetrievalQA 체인의 결과를 포맷팅하여 반환합니다.
    
    Args:
        result: RetrievalQA 체인의 결과 딕셔너리
        
    Returns:
        str: 포맷팅된 응답 문자열
    """
    try:
        if not isinstance(result, dict):
            return str(result)
        
        # 답변 추출
        answer = result.get("result", "")
        source_documents = result.get("source_documents", [])
        
        # 응답 구성
        response_parts = []
        
        if answer:
            response_parts.append(answer)
        
        # 소스 문서 정보 추가
        if source_documents:
            response_parts.append("\n\n참조 문서:")
            for i, doc in enumerate(source_documents, 1):
                metadata = doc.metadata
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "Unknown")
                response_parts.append(f"{i}. {source} (p.{page})")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"응답 포맷팅 중 오류 발생: {str(e)}")
        return str(result)


# 사용 예시 함수 (테스트용)
def example_qa_chain_usage():
    """
    RetrievalQA 체인 사용 예시
    """
    try:
        print("=== RetrievalQA 체인 사용 예시 ===")
        
        print("1. 외부에서 제공된 Retriever가 필요합니다.")
        print("2. LLM과 Retriever를 조합하여 QA 체인을 구성합니다.")
        print("3. 체인을 실행하여 질의응답을 수행합니다.")
        
        # 실제 사용 시 아래 코드 사용:
        # from .ollama_llm import build_llm
        # 
        # # LLM 생성
        # llm = build_llm("command-r:35b")
        # 
        # # 외부에서 제공된 Retriever 사용
        # retriever = external_retriever
        # 
        # # QA 체인 구축
        # qa_chain = build_qa_chain(llm, retriever)
        # 
        # # 체인 실행
        # query = "사용자 질문"
        # result = run_qa_chain(qa_chain, query)
        # 
        # # 결과 출력
        # formatted_response = format_qa_response(result)
        # print(formatted_response)
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_qa_chain_usage() 