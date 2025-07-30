"""
Ollama LLM Wrapper Implementation

Ollama에서 실행 중인 command-r:35b 모델을 LangChain의 LLM 객체로 wrapping합니다.
"""

import logging
from typing import Optional, Dict, Any

from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


def build_llm(model_name: str = "command-r:35b") -> BaseLanguageModel:
    """
    Ollama에서 실행 중인 모델을 LangChain의 LLM 객체로 wrapping합니다.
    
    Args:
        model_name: 사용할 모델 이름 (기본값: "command-r:35b")
        
    Returns:
        BaseLanguageModel: LangChain LLM 객체
        
    Raises:
        ValueError: 모델 이름이 유효하지 않은 경우
        ConnectionError: Ollama 서버에 연결할 수 없는 경우
    """
    try:
        # 입력 검증
        if not model_name or not model_name.strip():
            raise ValueError("모델 이름은 비어있을 수 없습니다")
        
        # Ollama LLM 객체 생성
        llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",  # Ollama 기본 URL
            timeout=120,  # 2분 타임아웃
            verbose=False  # 로깅 비활성화
        )
        
        logger.info(f"Ollama LLM 객체가 성공적으로 생성되었습니다: {model_name}")
        
        return llm
        
    except Exception as e:
        logger.error(f"Ollama LLM 객체 생성 중 오류 발생: {str(e)}")
        raise


def validate_llm_connection(llm: BaseLanguageModel) -> bool:
    """
    LLM 객체가 Ollama 서버와 정상적으로 연결되는지 검증합니다.
    
    Args:
        llm: 검증할 LLM 객체
        
    Returns:
        bool: 연결 성공 여부
    """
    try:
        # 간단한 테스트 질문으로 연결 확인
        test_prompt = "Hello, this is a test message. Please respond with 'OK'."
        
        # LLM 호출 (타임아웃 설정)
        response = llm.invoke(test_prompt)
        
        if response and len(response.strip()) > 0:
            logger.info("LLM 연결 검증 성공")
            return True
        else:
            logger.warning("LLM 응답이 비어있습니다")
            return False
            
    except Exception as e:
        logger.error(f"LLM 연결 검증 중 오류 발생: {str(e)}")
        return False


def get_llm_info(llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    LLM 객체의 정보를 반환합니다.
    
    Args:
        llm: 정보를 확인할 LLM 객체
        
    Returns:
        Dict[str, Any]: LLM 정보 딕셔너리
    """
    try:
        info = {
            "model_type": type(llm).__name__,
            "model_name": getattr(llm, 'model', 'Unknown'),
            "base_url": getattr(llm, 'base_url', 'Unknown'),
            "timeout": getattr(llm, 'timeout', 'Unknown'),
            "model_kwargs": getattr(llm, 'model_kwargs', {}),
        }
        
        return info
        
    except Exception as e:
        logger.error(f"LLM 정보 수집 중 오류 발생: {str(e)}")
        return {"error": str(e)}