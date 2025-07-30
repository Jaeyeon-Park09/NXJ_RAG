"""
Ensemble Retriever 테스트 스크립트

실제 데이터를 사용하여 앙상블 리트리버를 테스트합니다.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble_retriever import build_ensemble_retriever, format_query_for_embedding
from utils import load_metadata_sample, validate_ensemble_retriever, get_retriever_stats

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ensemble_retriever():
    """
    앙상블 리트리버를 테스트합니다.
    """
    try:
        # 경로 설정
        faiss_path = "/home/james4u1/NXJ_RAG/NXJ_Embed/emb"
        metadata_path = os.path.join(faiss_path, "metadata.json")
        
        logger.info("앙상블 리트리버 테스트 시작")
        logger.info(f"FAISS 경로: {faiss_path}")
        logger.info(f"메타데이터 경로: {metadata_path}")
        
        # 메타데이터에서 샘플 데이터 로드 (처음 1000개 문서)
        logger.info("메타데이터에서 샘플 데이터 로드 중...")
        bm25_texts, bm25_metadatas = load_metadata_sample(metadata_path, sample_size=1000)
        
        logger.info(f"로드된 문서 수: {len(bm25_texts)}")
        
        # 앙상블 리트리버 구축
        logger.info("앙상블 리트리버 구축 중...")
        ensemble_retriever = build_ensemble_retriever(
            faiss_path=faiss_path,
            bm25_texts=bm25_texts,
            bm25_metadatas=bm25_metadatas
        )
        
        # 리트리버 통계 출력
        stats = get_retriever_stats(ensemble_retriever)
        logger.info("리트리버 통계:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # 리트리버 검증
        logger.info("앙상블 리트리버 검증 중...")
        is_valid = validate_ensemble_retriever(ensemble_retriever)
        
        if not is_valid:
            logger.error("앙상블 리트리버 검증 실패")
            return False
        
        # 테스트 질문들
        test_queries = [
            "문서 내용",
            "법률 조항",
            "규정",
            "절차",
            "신청",
            "처리",
            "관련",
            "정보"
        ]
        
        # 각 테스트 질문으로 검색 실행
        logger.info("테스트 질문으로 검색 실행...")
        for query in test_queries:
            logger.info(f"\n=== 질문: {query} ===")
            
            # 질문을 임베딩 형식으로 변환
            formatted_query = format_query_for_embedding(query)
            
            # 검색 실행
            results = ensemble_retriever.get_relevant_documents(formatted_query)
            
            logger.info(f"검색 결과 수: {len(results)}")
            
            # 상위 3개 결과 출력
            for i, doc in enumerate(results[:3]):
                logger.info(f"결과 {i+1}:")
                logger.info(f"  내용: {doc.page_content[:200]}...")
                logger.info(f"  메타데이터: {doc.metadata}")
                logger.info("")
        
        logger.info("앙상블 리트리버 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        return False


def test_individual_retrievers():
    """
    개별 리트리버들을 테스트합니다.
    """
    try:
        logger.info("개별 리트리버 테스트 시작")
        
        # 경로 설정
        faiss_path = "/home/james4u1/NXJ_RAG/NXJ_Embed/emb"
        metadata_path = os.path.join(faiss_path, "metadata.json")
        
        # 메타데이터에서 샘플 데이터 로드
        bm25_texts, bm25_metadatas = load_metadata_sample(metadata_path, sample_size=500)
        
        # 앙상블 리트리버 구축
        ensemble_retriever = build_ensemble_retriever(
            faiss_path=faiss_path,
            bm25_texts=bm25_texts,
            bm25_metadatas=bm25_metadatas
        )
        
        # 개별 리트리버 테스트
        test_query = "문서 내용"
        formatted_query = format_query_for_embedding(test_query)
        
        logger.info(f"테스트 질문: {test_query}")
        
        # FAISS 리트리버만 테스트
        faiss_retriever = ensemble_retriever.retrievers[0]
        faiss_results = faiss_retriever.get_relevant_documents(formatted_query)
        logger.info(f"FAISS 결과 수: {len(faiss_results)}")
        
        # BM25 리트리버만 테스트
        bm25_retriever = ensemble_retriever.retrievers[1]
        bm25_results = bm25_retriever.get_relevant_documents(test_query)  # BM25는 원본 질문 사용
        logger.info(f"BM25 결과 수: {len(bm25_results)}")
        
        # 앙상블 결과
        ensemble_results = ensemble_retriever.get_relevant_documents(formatted_query)
        logger.info(f"앙상블 결과 수: {len(ensemble_results)}")
        
        logger.info("개별 리트리버 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"개별 리트리버 테스트 중 오류 발생: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Ensemble Retriever 테스트 시작")
    logger.info("=" * 50)
    
    # 메인 테스트 실행
    success = test_ensemble_retriever()
    
    if success:
        logger.info("메인 테스트 성공")
        
        # 개별 리트리버 테스트 실행
        individual_success = test_individual_retrievers()
        
        if individual_success:
            logger.info("개별 리트리버 테스트 성공")
        else:
            logger.error("개별 리트리버 테스트 실패")
    else:
        logger.error("메인 테스트 실패")
    
    logger.info("=" * 50)
    logger.info("테스트 완료")
    logger.info("=" * 50) 