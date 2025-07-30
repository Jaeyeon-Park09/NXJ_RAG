"""
Ensemble Retriever 유틸리티 함수들

메타데이터 처리 및 BM25 리트리버 구성을 위한 헬퍼 함수들
"""

import json
import os
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def load_metadata_for_bm25(metadata_path: str, max_docs: int = None) -> Tuple[List[str], List[dict]]:
    """
    메타데이터 파일에서 BM25 리트리버용 텍스트와 메타데이터를 추출합니다.
<<<<<<< HEAD
=======
    JSON 배열 형식과 JSONL 형식 모두 지원합니다.
>>>>>>> 60b74fa (2)
    
    Args:
        metadata_path: 메타데이터 JSON 파일 경로
        max_docs: 로드할 최대 문서 수 (None이면 전체 로드)
        
    Returns:
        Tuple[List[str], List[dict]]: (텍스트 리스트, 메타데이터 리스트)
    """
    try:
        texts = []
        metadatas = []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
<<<<<<< HEAD
=======
            content = f.read().strip()
            
            # JSON 배열 형식인지 확인
            if content.startswith('[') and content.endswith(']'):
                try:
                    data_list = json.loads(content)
                    if isinstance(data_list, list):
                        # JSON 배열 형식 처리
                        for i, data in enumerate(data_list):
                            if max_docs and i >= max_docs:
                                break
                                
                            if isinstance(data, dict):
                                # 텍스트 추출 (page_content 또는 text 필드에서)
                                text = data.get('page_content') or data.get('text', '')
                                if text:
                                    texts.append(text)
                                    
                                    # 전체 메타데이터 포함 (page_content/text도 포함)
                                    metadatas.append(data)
                        
                        logger.info(f"JSON 배열 형식으로 BM25용 데이터 로드 완료: {len(texts)}개 문서")
                        return texts, metadatas
                        
                except json.JSONDecodeError:
                    logger.warning("JSON 배열 파싱 실패, JSONL 형식으로 시도")
            
            # JSONL 형식으로 처리
            f.seek(0)  # 파일 포인터를 처음으로 되돌림
>>>>>>> 60b74fa (2)
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # 텍스트 추출 (page_content 또는 text 필드에서)
                    text = data.get('page_content') or data.get('text', '')
                    if text:
                        texts.append(text)
                        
<<<<<<< HEAD
                        # 메타데이터에서 page_content/text 제외
                        metadata = {k: v for k, v in data.items() 
                                  if k not in ['page_content', 'text']}
                        metadatas.append(metadata)
=======
                        # 전체 메타데이터 포함 (page_content/text도 포함)
                        metadatas.append(data)
>>>>>>> 60b74fa (2)
                        
                except json.JSONDecodeError:
                    logger.warning(f"라인 {i+1}에서 JSON 파싱 오류 발생, 건너뜀")
                    continue
        
        logger.info(f"BM25용 데이터 로드 완료: {len(texts)}개 문서")
        return texts, metadatas
        
    except Exception as e:
        logger.error(f"메타데이터 로드 중 오류 발생: {str(e)}")
        raise


def load_metadata_sample(metadata_path: str, sample_size: int = 1000) -> Tuple[List[str], List[dict]]:
    """
    메타데이터 파일에서 샘플 데이터를 로드합니다 (테스트용).
    
    Args:
        metadata_path: 메타데이터 JSON 파일 경로
        sample_size: 샘플 크기
        
    Returns:
        Tuple[List[str], List[dict]]: (텍스트 리스트, 메타데이터 리스트)
    """
    return load_metadata_for_bm25(metadata_path, max_docs=sample_size)


def validate_ensemble_retriever(ensemble_retriever, test_query: str = "테스트 질문") -> bool:
    """
    앙상블 리트리버가 정상적으로 작동하는지 검증합니다.
    
    Args:
        ensemble_retriever: 검증할 앙상블 리트리버
        test_query: 테스트용 질문
        
    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 테스트 질문을 임베딩 형식으로 변환
        formatted_query = f"query: {test_query}"
        
        # 검색 실행
        results = ensemble_retriever.get_relevant_documents(formatted_query)
        
        # 결과 검증
        if not isinstance(results, list):
            logger.error("검색 결과가 리스트가 아닙니다")
            return False
        
        if len(results) == 0:
            logger.warning("검색 결과가 없습니다")
            return True  # 결과가 없는 것도 정상적인 경우일 수 있음
        
        # 첫 번째 결과의 구조 검증
        first_result = results[0]
        if not hasattr(first_result, 'page_content'):
            logger.error("검색 결과에 page_content 속성이 없습니다")
            return False
        
        if not hasattr(first_result, 'metadata'):
            logger.error("검색 결과에 metadata 속성이 없습니다")
            return False
        
        logger.info(f"앙상블 리트리버 검증 성공: {len(results)}개 결과 반환")
        return True
        
    except Exception as e:
        logger.error(f"앙상블 리트리버 검증 중 오류 발생: {str(e)}")
        return False


def get_retriever_stats(ensemble_retriever) -> Dict[str, Any]:
    """
    앙상블 리트리버의 통계 정보를 반환합니다.
    
    Args:
        ensemble_retriever: 통계를 확인할 앙상블 리트리버
        
    Returns:
        Dict[str, Any]: 리트리버 통계 정보
    """
    try:
        stats = {
            "retriever_type": "EnsembleRetriever",
            "num_retrievers": len(ensemble_retriever.retrievers),
            "weights": ensemble_retriever.weights,
            "retriever_names": []
        }
        
        for i, retriever in enumerate(ensemble_retriever.retrievers):
            retriever_name = type(retriever).__name__
            stats["retriever_names"].append(retriever_name)
            
            # 각 리트리버별 추가 정보
<<<<<<< HEAD
            if hasattr(retriever, 'index') and retriever.index is not None:
                stats[f"retriever_{i}_index_size"] = retriever.index.ntotal
            elif hasattr(retriever, 'docstore') and retriever.docstore is not None:
=======
            if hasattr(retriever, 'vectorstore') and retriever.vectorstore is not None:
                # FAISS vectorstore의 경우
                if hasattr(retriever.vectorstore, 'index') and retriever.vectorstore.index is not None:
                    stats[f"retriever_{i}_index_size"] = retriever.vectorstore.index.ntotal
            elif hasattr(retriever, 'docstore') and retriever.docstore is not None:
                # BM25 리트리버의 경우
>>>>>>> 60b74fa (2)
                stats[f"retriever_{i}_docstore_size"] = len(retriever.docstore._dict)
        
        return stats
        
    except Exception as e:
        logger.error(f"리트리버 통계 수집 중 오류 발생: {str(e)}")
        return {"error": str(e)} 