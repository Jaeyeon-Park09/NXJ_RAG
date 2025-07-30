"""
Utility Functions for NXJ_Retriever

NXJ_Retriever 패키지에서 사용하는 유틸리티 함수들을 제공합니다.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def load_metadata_for_bm25(metadata_path: str, max_docs: int = None) -> Tuple[List[str], List[dict]]:
    """
    메타데이터 JSON 파일을 로드하여 BM25 리트리버용 데이터를 생성합니다.
    
    Args:
        metadata_path: 메타데이터 JSON 파일 경로
        max_docs: 최대 문서 수 (None이면 전체 로드)
        
    Returns:
        Tuple[List[str], List[dict]]: (텍스트 리스트, 메타데이터 리스트)
        
    Raises:
        FileNotFoundError: 메타데이터 파일을 찾을 수 없는 경우
        ValueError: JSON 파일 형식이 잘못된 경우
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        
        # JSON 파일 로드
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 데이터 형식 검증
        if not isinstance(data, list):
            raise ValueError("메타데이터 파일은 JSON 배열 형식이어야 합니다")
        
        # 최대 문서 수 제한
        if max_docs is not None:
            data = data[:max_docs]
        
        # 텍스트와 메타데이터 분리
        texts = []
        metadatas = []
        
        for i, item in enumerate(data):
            try:
                if isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
                    # text 필드를 제외한 나머지를 메타데이터로 사용
                    metadata = {k: v for k, v in item.items() if k != 'text'}
                    metadatas.append(metadata)
                else:
                    logger.warning(f"라인 {i+1}에서 'text' 필드를 찾을 수 없음, 건너뜀")
                    continue
            except Exception as e:
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
            
            # 각 리트리버별 추가 정보 (가능한 경우)
            try:
                if hasattr(retriever, 'docstore') and hasattr(retriever.docstore, '_dict'):
                    stats[f"retriever_{i}_docstore_size"] = len(retriever.docstore._dict)
                elif hasattr(retriever, 'vectorstore'):
                    stats[f"retriever_{i}_vectorstore_type"] = type(retriever.vectorstore).__name__
            except Exception as e:
                logger.warning(f"리트리버 {i}의 추가 정보 수집 실패: {str(e)}")
        
        return stats
        
    except Exception as e:
        logger.error(f"리트리버 통계 수집 중 오류 발생: {str(e)}")
        return {"error": str(e)}


# 사용 예시 함수 (테스트용)
def example_utils_usage():
    """
    유틸리티 함수 사용 예시
    """
    # 메타데이터 파일 경로
    metadata_path = "/home/james4u1/NXJ_RAG/NXJ_Embed/emb/metadata.json"
    
    try:
        print("=== 유틸리티 함수 사용 예시 ===")
        
        # 메타데이터 로드
        if os.path.exists(metadata_path):
            texts, metadatas = load_metadata_sample(metadata_path, sample_size=5)
            print(f"로드된 문서 수: {len(texts)}")
            print(f"첫 번째 문서: {texts[0][:100]}...")
            print(f"첫 번째 메타데이터: {metadatas[0]}")
        else:
            print("메타데이터 파일이 존재하지 않습니다")
        
        print("=== 예시 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_utils_usage() 