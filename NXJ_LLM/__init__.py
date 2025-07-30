"""
NXJ_LLM Package

Ollama 기반 LLM 모델을 LangChain으로 wrapping하는 패키지
"""

# LLM 관련 함수들
from .ollama_llm import (
    build_llm,
    validate_llm_connection,
    get_llm_info
)

# 프롬프트 템플릿 관련 함수들
from .prompt_templates import (
    build_report_prompt,
    build_simple_report_prompt,
    extract_sources_from_documents,
    format_context_with_sources,
    create_prompt_with_documents
)

# LLM 체인 관련 함수들
from .llm_chain import (
    build_llm_chain,
    run_chain_with_documents,
    run_chain_with_input_dict,
    create_full_chain,
    validate_chain,
    get_chain_info
)

# RetrievalQA 체인 관련 함수들
from .retrieval_qa_chain import (
    build_qa_chain,
    run_qa_chain,
    create_full_qa_chain,
    validate_qa_chain,
    get_qa_chain_info,
    format_qa_response
)

__version__ = "1.0.0"

__all__ = [
    # LLM 관련
    "build_llm",
    "validate_llm_connection",
    "get_llm_info",
    
    # 프롬프트 템플릿 관련
    "build_report_prompt",
    "build_simple_report_prompt",
    "extract_sources_from_documents",
    "format_context_with_sources",
    "create_prompt_with_documents",
    
    # LLM 체인 관련
    "build_llm_chain",
    "run_chain_with_documents",
    "run_chain_with_input_dict",
    "create_full_chain",
    "validate_chain",
    "get_chain_info",
    
    # RetrievalQA 체인 관련
    "build_qa_chain",
    "run_qa_chain",
    "create_full_qa_chain",
    "validate_qa_chain",
    "get_qa_chain_info",
    "format_qa_response"
] 