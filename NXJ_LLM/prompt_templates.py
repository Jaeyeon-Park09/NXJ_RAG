"""
Prompt Template Implementation

LangChain의 PromptTemplate을 사용하여 문서 기반 응답 생성을 위한 프롬프트를 구현합니다.
사용자 질문에 대해 context를 바탕으로 자연스럽고 유익한 응답을 생성하며, 출처를 명시합니다.
"""

import logging
from typing import List, Dict, Any
from langchain_core.prompts.prompt import PromptTemplate
from langchain.schema import Document

logger = logging.getLogger(__name__)


def build_report_prompt() -> PromptTemplate:
    """
    문서 기반 응답 생성을 위한 프롬프트 템플릿을 구축합니다.
    
    Returns:
        PromptTemplate: 구성된 프롬프트 템플릿
    """
    try:
        # 프롬프트 템플릿 정의
        template = """당신은 전문적이고 유익한 답변을 제공하는 AI 어시스턴트입니다. 
주어진 문서 내용을 바탕으로 사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.

다음 지침을 따라 답변해주세요:

1. **답변 형식**: 질문의 성격에 따라 가장 적절한 형식으로 답변하세요
   - 설명이 필요한 경우: 단락형으로 자세히 설명
   - 비교나 분석이 필요한 경우: 항목형으로 정리
   - 요약이 필요한 경우: 요약형으로 간결하게 정리

2. **내용의 정확성**: 제공된 문서 내용에 기반하여 정확한 정보만을 포함하세요
   - 문서에 없는 내용은 추측하지 마세요
   - 불확실한 정보는 "문서에 명시되지 않았습니다"라고 표시하세요

3. **자연스러운 표현**: 읽기 쉽고 이해하기 쉬운 자연스러운 한국어로 답변하세요
   - 전문 용어는 필요시 설명을 추가하세요
   - 문장은 명확하고 간결하게 작성하세요

4. **출처 명시**: 답변 마지막에 반드시 참조한 문서의 출처를 명시하세요
   - 각 문서의 파일명과 페이지 번호를 포함하세요
   - 동일 문서의 여러 페이지는 하나로 묶어서 표시하세요
   - 참조 문서가 없는 경우 출처는 생략하세요

출처 표시 형식:
출처:
- {문서명1} (p.{페이지1}, {페이지2})
- {문서명2} (p.{페이지3})

사용자 질문: {question}

참조 문서 내용:
{context}

답변:"""

        # PromptTemplate 생성
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template=template
        )
        
        logger.info("문서 기반 응답 프롬프트 템플릿이 성공적으로 생성되었습니다")
        return prompt_template
        
    except Exception as e:
        logger.error(f"프롬프트 템플릿 생성 중 오류 발생: {str(e)}")
        raise


def build_simple_report_prompt() -> PromptTemplate:
    """
    간단한 문서 기반 응답을 위한 프롬프트 템플릿을 구축합니다.
    
    Returns:
        PromptTemplate: 간단한 프롬프트 템플릿
    """
    try:
        template = """다음 문서 내용을 바탕으로 질문에 답변해주세요.

질문: {question}

문서 내용:
{context}

답변을 작성할 때 다음 사항을 지켜주세요:
1. 문서 내용에 기반하여 정확하게 답변하세요
2. 자연스러운 한국어로 작성하세요
3. 질문에 가장 적절한 형식(단락형, 항목형, 요약형)으로 답변하세요
4. 답변 마지막에 참조한 문서의 출처를 명시하세요

출처 형식:
출처:
- {문서명} (p.{페이지번호})

답변:"""

        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template=template
        )
        
        logger.info("간단한 문서 기반 응답 프롬프트 템플릿이 생성되었습니다")
        return prompt_template
        
    except Exception as e:
        logger.error(f"간단한 프롬프트 템플릿 생성 중 오류 발생: {str(e)}")
        raise


def extract_sources_from_documents(documents: List[Document]) -> str:
    """
    Document 객체 리스트에서 출처 정보를 추출하여 문자열로 반환합니다.
    
    Args:
        documents: Document 객체 리스트
        
    Returns:
        str: 출처 정보 문자열
    """
    try:
        if not documents:
            return ""
        
        # 출처 정보 수집
        source_info = {}
        
        for doc in documents:
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            
            if source not in source_info:
                source_info[source] = []
            
            if page != "Unknown":
                source_info[source].append(str(page))
        
        # 출처 문자열 생성
        if not source_info:
            return ""
        
        source_lines = ["출처:"]
        
        for source, pages in source_info.items():
            if pages:
                # 페이지 번호 정렬 및 중복 제거
                unique_pages = sorted(list(set(pages)), key=lambda x: int(x) if x.isdigit() else 0)
                page_str = ", ".join(unique_pages)
                source_lines.append(f"- {source} (p.{page_str})")
            else:
                source_lines.append(f"- {source}")
        
        return "\n".join(source_lines)
        
    except Exception as e:
        logger.error(f"출처 정보 추출 중 오류 발생: {str(e)}")
        return ""


def format_context_with_sources(documents: List[Document]) -> str:
    """
    Document 객체 리스트를 context 문자열로 변환하고 출처 정보를 포함합니다.
    
    Args:
        documents: Document 객체 리스트
        
    Returns:
        str: 출처 정보가 포함된 context 문자열
    """
    try:
        if not documents:
            return "참조할 문서가 없습니다."
        
        # 문서 내용 수집
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            
            context_parts.append(f"[문서 {i}] {source} (p.{page})")
            context_parts.append(content)
            context_parts.append("")  # 빈 줄 추가
        
        context_text = "\n".join(context_parts)
        
        # 출처 정보 추가
        sources = extract_sources_from_documents(documents)
        if sources:
            context_text += f"\n\n{sources}"
        
        return context_text
        
    except Exception as e:
        logger.error(f"Context 포맷팅 중 오류 발생: {str(e)}")
        return "문서 처리 중 오류가 발생했습니다."


def create_prompt_with_documents(
    prompt_template: PromptTemplate,
    question: str,
    documents: List[Document]
) -> str:
    """
    프롬프트 템플릿과 문서를 사용하여 완성된 프롬프트를 생성합니다.
    
    Args:
        prompt_template: PromptTemplate 객체
        question: 사용자 질문
        documents: Document 객체 리스트
        
    Returns:
        str: 완성된 프롬프트 문자열
    """
    try:
        # Context 포맷팅
        context = format_context_with_sources(documents)
        
        # 프롬프트 생성
        formatted_prompt = prompt_template.format(
            question=question,
            context=context
        )
        
        return formatted_prompt
        
    except Exception as e:
        logger.error(f"프롬프트 생성 중 오류 발생: {str(e)}")
        raise


# 사용 예시 함수 (테스트용)
def example_prompt_usage():
    """
    프롬프트 템플릿 사용 예시
    """
    from langchain.schema import Document
    
    try:
        print("=== 프롬프트 템플릿 사용 예시 ===")
        
        # 샘플 문서 생성
        sample_documents = [
            Document(
                page_content="휠체어 사용자는 건물 내부에서 이동할 때 반드시 휠체어 전용 경사로를 이용해야 합니다.",
                metadata={"source": "휠체어가이드라인.pdf", "page": 13}
            ),
            Document(
                page_content="휠체어 전용 경사로의 경사는 1:12 이하여야 하며, 너비는 최소 1.2m 이상이어야 합니다.",
                metadata={"source": "휠체어가이드라인.pdf", "page": 15}
            ),
            Document(
                page_content="건물 출입구에는 휠체어 사용자를 위한 자동문 설치가 권장됩니다.",
                metadata={"source": "건축물접근성.pdf", "page": 8}
            )
        ]
        
        # 프롬프트 템플릿 생성
        print("1. 프롬프트 템플릿 생성")
        prompt_template = build_report_prompt()
        
        # 샘플 질문
        question = "휠체어 사용자를 위한 건물 설계 시 고려사항은 무엇인가요?"
        
        print(f"2. 질문: {question}")
        
        # 완성된 프롬프트 생성
        print("3. 완성된 프롬프트 생성")
        formatted_prompt = create_prompt_with_documents(
            prompt_template, question, sample_documents
        )
        
        print("4. 생성된 프롬프트:")
        print("-" * 50)
        print(formatted_prompt)
        print("-" * 50)
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_prompt_usage() 