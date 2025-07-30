"""
Ollama LLM Wrapper Implementation

Ollama에서 실행 중인 command-r:35b 모델을 LangChain의 LLM 객체로 wrapping합니다.
"""

import logging
from typing import Optional, Dict, Any

from langchain_community.llms import Ollama
<<<<<<< HEAD
from langchain.schema import BaseLanguageModel
=======
from langchain_core.language_models import BaseLanguageModel
>>>>>>> 60b74fa (2)

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
        
<<<<<<< HEAD
        # command-r:35b 모델의 긴 context window를 위한 설정
        model_kwargs = {
            "num_ctx": 32768,  # 32K context window
            "num_predict": 4096,  # 최대 토큰 생성 수
            "temperature": 0.1,  # 낮은 temperature로 일관성 확보
            "top_p": 0.9,  # nucleus sampling
            "repeat_penalty": 1.1,  # 반복 방지
            "stop": ["</s>", "Human:", "Assistant:"],  # 중단 토큰
        }
        
=======
>>>>>>> 60b74fa (2)
        # Ollama LLM 객체 생성
        llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",  # Ollama 기본 URL
<<<<<<< HEAD
            model_kwargs=model_kwargs,
=======
            temperature=0.1,
>>>>>>> 60b74fa (2)
            timeout=120,  # 2분 타임아웃
            verbose=False  # 로깅 비활성화
        )
        
        logger.info(f"Ollama LLM 객체가 성공적으로 생성되었습니다: {model_name}")
<<<<<<< HEAD
        logger.info(f"Context window: {model_kwargs['num_ctx']} tokens")
        logger.info(f"Max tokens: {model_kwargs['num_predict']} tokens")
=======
>>>>>>> 60b74fa (2)
        
        return llm
        
    except Exception as e:
        logger.error(f"Ollama LLM 객체 생성 중 오류 발생: {str(e)}")
        raise


<<<<<<< HEAD
def build_llm_with_custom_config(
    model_name: str = "command-r:35b",
    temperature: float = 0.1,
    context_window: int = 32768,
    max_tokens: int = 4096,
    base_url: str = "http://localhost:11434"
) -> BaseLanguageModel:
    """
    사용자 정의 설정으로 Ollama LLM 객체를 생성합니다.
    
    Args:
        model_name: 사용할 모델 이름
        temperature: 생성 다양성 조절 (0.0-1.0)
        context_window: 컨텍스트 윈도우 크기
        max_tokens: 최대 생성 토큰 수
        base_url: Ollama 서버 URL
        
    Returns:
        BaseLanguageModel: LangChain LLM 객체
    """
    try:
        # 입력 검증
        if not model_name or not model_name.strip():
            raise ValueError("모델 이름은 비어있을 수 없습니다")
        
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature는 0.0과 1.0 사이여야 합니다")
        
        if context_window <= 0:
            raise ValueError("context_window는 양수여야 합니다")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens는 양수여야 합니다")
        
        # 사용자 정의 설정으로 model_kwargs 구성
        model_kwargs = {
            "num_ctx": context_window,
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["</s>", "Human:", "Assistant:"],
        }
        
        # Ollama LLM 객체 생성
        llm = Ollama(
            model=model_name,
            base_url=base_url,
            model_kwargs=model_kwargs,
            timeout=120,
            verbose=False
        )
        
        logger.info(f"사용자 정의 설정으로 Ollama LLM 객체 생성 완료: {model_name}")
        logger.info(f"Temperature: {temperature}, Context: {context_window}, Max tokens: {max_tokens}")
        
        return llm
        
    except Exception as e:
        logger.error(f"사용자 정의 설정 LLM 객체 생성 중 오류 발생: {str(e)}")
        raise


=======
>>>>>>> 60b74fa (2)
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


<<<<<<< HEAD
# 사용 예시 함수 (테스트용)
def example_llm_usage():
    """
    Ollama LLM 사용 예시
    """
    try:
        print("=== Ollama LLM 사용 예시 ===")
        
        # 기본 LLM 객체 생성
        print("1. 기본 LLM 객체 생성 중...")
        llm = build_llm()
        
        # LLM 정보 출력
        print("2. LLM 정보:")
        info = get_llm_info(llm)
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 연결 검증
        print("3. 연결 검증 중...")
        is_connected = validate_llm_connection(llm)
        print(f"   연결 상태: {'성공' if is_connected else '실패'}")
        
        # 간단한 테스트
        if is_connected:
            print("4. 간단한 테스트 질문:")
            test_question = "What is 2 + 2? Please answer briefly."
            print(f"   질문: {test_question}")
            
            try:
                response = llm.invoke(test_question)
                print(f"   응답: {response}")
            except Exception as e:
                print(f"   응답 오류: {str(e)}")
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_llm_usage() 
=======
 
>>>>>>> 60b74fa (2)
