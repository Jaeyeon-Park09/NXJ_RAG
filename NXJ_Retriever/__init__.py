"""
NXJ_Retriever Package

LangChain 기반의 Ensemble Retriever 구현
FAISS와 BM25를 결합한 강건한 문서 검색 시스템
"""

# Ensemble Retriever 관련 함수들
from .ensemble_retriever import (
    build_ensemble_retriever,
    format_query_for_embedding,
    load_faiss_index,
    create_bm25_retriever
)

# 문서 재정렬 관련 함수들
from .document_reorder import (
    create_long_context_reorder,
    reorder_documents
)

# 문서 압축 관련 함수들
from .document_compressor import (
    build_llm_extractor,
    compress_documents,
    extract_relevant_content
)

# 압축 파이프라인 관련 함수들
from .compressor_pipeline import (
    build_compressor_pipeline,
    compress_with_pipeline,
    create_full_pipeline
)

# Contextual Compression Retriever 관련 함수들
from .contextual_compression_retriever import (
    build_compressed_retriever,
    retrieve_with_compression,
    create_full_compressed_retriever,
    validate_compressed_retriever
)

# 유틸리티 함수들
from .utils import (
    load_metadata_for_bm25,
    load_metadata_sample,
    validate_ensemble_retriever,
    get_retriever_stats
)

__version__ = "1.0.0"

__all__ = [
    # Ensemble Retriever 관련
    "build_ensemble_retriever",
    "format_query_for_embedding",
    "load_faiss_index",
    "create_bm25_retriever",
    
    # 문서 재정렬 관련
    "create_long_context_reorder",
    "reorder_documents",
    
    # 문서 압축 관련
    "build_llm_extractor",
    "compress_documents",
    "extract_relevant_content",
    
    # 압축 파이프라인 관련
    "build_compressor_pipeline",
    "compress_with_pipeline",
    "create_full_pipeline",
    
    # Contextual Compression Retriever 관련
    "build_compressed_retriever",
    "retrieve_with_compression",
    "create_full_compressed_retriever",
    "validate_compressed_retriever",
    
    # 유틸리티 함수들
    "load_metadata_for_bm25",
    "load_metadata_sample",
    "validate_ensemble_retriever",
    "get_retriever_stats"
] 