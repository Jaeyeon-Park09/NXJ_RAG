"""
LLM Chain Implementation

LangChain의 LLMChain을 사용하여 PromptTemplate과 LLM 객체를 결합합니다.
"stuff" 방식으로 context를 prompt에 삽입하여 문서 기반 응답을 생성합니다.
"""

import logging
from typing import Dict, Any, List


logger = logging.getLogger(__name__)


def build_llm_chain(
    llm: BaseLanguageModel,
    prompt: PromptTemplate
) -> LLMChain:
    """
    PromptTemplate과 LLM 객체를 사용하여 LLMChain을 구축합니다.
    
    Args:
        llm: BaseLanguageModel 객체 (command-r:35b 기반)
        prompt: PromptTemplate 객체 (문서 기반 응답용)
        
    Returns:
        LLMChain: 구성된 LLMChain 객체
        
    Raises:
        ValueError: 입력 파라미터가 유효하지 않은 경우
    """
    try:
        # 입력 검증
        if llm is None:
            raise ValueError("LLM 객체는 None일 수 없습니다")
        
        if prompt is None:
            raise ValueError("PromptTemplate 객체는 None일 수 없습니다")
        
        if not isinstance(llm, BaseLanguageModel):
            raise ValueError("LLM 객체는 BaseLanguageModel의 인스턴스여야 합니다")
        
        if not isinstance(prompt, PromptTemplate):
            raise ValueError("Prompt 객체는 PromptTemplate의 인스턴스여야 합니다")
        
        # LLMChain 생성 ("stuff" 방식)
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False  # 로깅 비활성화
        )
        
        logger.info("LLMChain이 성공적으로 구성되었습니다")
        logger.info(f"LLM 타입: {type(llm).__name__}")
        logger.info(f"Prompt 타입: {type(prompt).__name__}")
        
        return chain
        
    except Exception as e:
        logger.error(f"LLMChain 구축 중 오류 발생: {str(e)}")
        raise


def run_chain_with_documents(
    chain: LLMChain,
    question: str,
    documents: List[Document]
) -> str:
    """
    LLMChain을 사용하여 문서 기반 응답을 생성합니다.
    
    Args:
        chain: LLMChain 객체
        question: 사용자 질문
        documents: 참조할 Document 객체 리스트
        
    Returns:
        str: 생성된 응답
    """
    try:
        # 입력 검증
        if not question or not question.strip():
            raise ValueError("질문은 비어있을 수 없습니다")
        
        if not documents:
            logger.warning("참조할 문서가 없습니다")
            return "참조할 문서가 없어 답변을 생성할 수 없습니다."
        
        # Document 객체들을 context 문자열로 변환
        context = format_context_with_sources(documents)
        
        # 체인 실행
        response = chain.run({
            "question": question,
            "context": context
        })
        
        logger.info(f"LLMChain 실행 완료: {len(documents)}개 문서 참조")
        return response
        
    except Exception as e:
        logger.error(f"LLMChain 실행 중 오류 발생: {str(e)}")
        raise


def run_chain_with_input_dict(
    chain: LLMChain,
    input_dict: Dict[str, Any]
) -> str:
    """
    LLMChain을 입력 딕셔너리로 실행합니다.
    
    Args:
        chain: LLMChain 객체
        input_dict: 입력 딕셔너리 {"question": ..., "context": ...}
        
    Returns:
        str: 생성된 응답
    """
    try:
        # 입력 검증
        if not isinstance(input_dict, dict):
            raise ValueError("input_dict는 딕셔너리여야 합니다")
        
        required_keys = ["question", "context"]
        for key in required_keys:
            if key not in input_dict:
                raise ValueError(f"input_dict에 '{key}' 키가 필요합니다")
        
        # 체인 실행
        response = chain.run(input_dict)
        
        logger.info("LLMChain 실행 완료")
        return response
        
    except Exception as e:
        logger.error(f"LLMChain 실행 중 오류 발생: {str(e)}")
        raise


def create_full_chain(
    model_name: str = "command-r:35b",
    use_simple_prompt: bool = False
) -> LLMChain:
    """
    LLM과 PromptTemplate을 포함한 완전한 체인을 생성합니다.
    
    Args:
        model_name: 사용할 모델 이름
        use_simple_prompt: 간단한 프롬프트 사용 여부
        
    Returns:
        LLMChain: 완전한 LLMChain 객체
    """
    try:
        
        # LLM 생성
        llm = build_llm(model_name)
        
        # PromptTemplate 생성
        if use_simple_prompt:
            prompt = build_simple_report_prompt()
        else:
            prompt = build_report_prompt()
        
        # LLMChain 구축
        chain = build_llm_chain(llm, prompt)
        
        return chain
        
    except Exception as e:
        logger.error(f"완전한 체인 생성 중 오류 발생: {str(e)}")
        raise


def validate_chain(chain: LLMChain, test_question: str = "테스트 질문입니다.") -> bool:
    """
    LLMChain이 정상적으로 작동하는지 검증합니다.
    
    Args:
        chain: 검증할 LLMChain 객체
        test_question: 테스트용 질문
        
    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 간단한 테스트 입력
        test_input = {
            "question": test_question,
            "context": "이것은 테스트용 문서 내용입니다."
        }
        
        # 체인 실행
        response = chain.run(test_input)
        
        # 응답 검증
        if not response or not response.strip():
            logger.warning("체인 응답이 비어있습니다")
            return False
        
        logger.info(f"LLMChain 검증 성공: {len(response)}자 응답")
        return True
        
    except Exception as e:
        logger.error(f"LLMChain 검증 중 오류 발생: {str(e)}")
        return False


def get_chain_info(chain: LLMChain) -> Dict[str, Any]:
    """
    LLMChain 객체의 정보를 반환합니다.
    
    Args:
        chain: 정보를 확인할 LLMChain 객체
        
    Returns:
        Dict[str, Any]: 체인 정보 딕셔너리
    """
    try:
        info = {
            "chain_type": type(chain).__name__,
            "llm_type": type(chain.llm).__name__,
            "prompt_type": type(chain.prompt).__name__,
            "verbose": getattr(chain, 'verbose', False),
            "input_variables": getattr(chain.prompt, 'input_variables', []),
        }
        
        return info
        
    except Exception as e:
        logger.error(f"체인 정보 수집 중 오류 발생: {str(e)}")
        return {"error": str(e)}


# 사용 예시 함수 (테스트용)
def example_chain_usage():
    """
    LLMChain 사용 예시
    """
    from langchain.schema import Document
    
    try:
        print("=== LLMChain 사용 예시 ===")
        
        # 완전한 체인 생성
        print("1. LLMChain 생성 중...")
        chain = create_full_chain()
        
        # 체인 정보 출력
        print("2. 체인 정보:")
        info = get_chain_info(chain)
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 체인 검증
        print("3. 체인 검증 중...")
        is_valid = validate_chain(chain)
        print(f"   검증 결과: {'성공' if is_valid else '실패'}")
        
        # 샘플 문서 생성
        sample_documents = [
            Document(
                page_content="휠체어 사용자를 위한 건물 설계 시 접근성을 고려해야 합니다.",
                metadata={"source": "접근성가이드.pdf", "page": 5}
            ),
            Document(
                page_content="휠체어 전용 경사로의 경사는 1:12 이하여야 합니다.",
                metadata={"source": "접근성가이드.pdf", "page": 7}
            )
        ]
        
        # 체인 실행
        if is_valid:
            print("4. 체인 실행:")
            question = "휠체어 사용자를 위한 건물 설계 시 고려사항은 무엇인가요?"
            print(f"   질문: {question}")
            
            try:
                response = run_chain_with_documents(chain, question, sample_documents)
                print(f"   응답: {response[:200]}...")
            except Exception as e:
                print(f"   실행 오류: {str(e)}")
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_chain_usage() 