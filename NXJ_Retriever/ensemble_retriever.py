"""
Ensemble Retriever Implementation

FAISS와 BM25 리트리버를 결합한 앙상블 검색 시스템
LangChain 기반으로 구현
"""

import os
import json
from typing import List, Dict, Any
import logging

<<<<<<< HEAD
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import FAISSRetriever, BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import BaseEmbedding
=======
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

>>>>>>> 60b74fa (2)
from langchain.schema import Document

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


<<<<<<< HEAD
def load_faiss_index(faiss_path: str) -> FAISSRetriever:
    """
    FAISS 인덱스를 로드하여 FAISSRetriever를 생성합니다.
=======
def load_faiss_index(faiss_path: str):
    """
    FAISS 인덱스를 로드하여 FAISS vectorstore를 생성합니다.
>>>>>>> 60b74fa (2)
    
    Args:
        faiss_path: FAISS 인덱스가 저장된 디렉토리 경로
        
    Returns:
<<<<<<< HEAD
        FAISSRetriever: 로드된 FAISS 리트리버
=======
        FAISS: 로드된 FAISS vectorstore
>>>>>>> 60b74fa (2)
    """
    try:
        # FAISS 인덱스 파일 경로 확인
        faiss_index_path = os.path.join(faiss_path, "faiss_index.bin")
        metadata_path = os.path.join(faiss_path, "metadata.json")
        
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {faiss_index_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        
        # 임베딩 모델 로드
        embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
<<<<<<< HEAD
        # FAISS 인덱스 로드
        import faiss
        faiss_index = faiss.read_index(faiss_index_path)
        
        # 메타데이터 로드
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # FAISSRetriever 생성
        faiss_retriever = FAISSRetriever(
            index=faiss_index,
            embedding_function=embedding_model,
            metadata=metadata
        )
        
        logger.info(f"FAISS 리트리버가 성공적으로 로드되었습니다: {faiss_path}")
        return faiss_retriever
=======
        # FAISS vectorstore 로드 (faiss_index.bin 파일 사용)
        faiss_vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
        
        logger.info(f"FAISS vectorstore가 성공적으로 로드되었습니다: {faiss_path}")
        return faiss_vectorstore
>>>>>>> 60b74fa (2)
        
    except Exception as e:
        logger.error(f"FAISS 인덱스 로드 중 오류 발생: {str(e)}")
        raise


def create_bm25_retriever(texts: List[str], metadatas: List[dict]) -> BM25Retriever:
    """
    BM25 리트리버를 생성합니다.
    
    Args:
        texts: 검색할 텍스트 리스트
        metadatas: 각 텍스트에 해당하는 메타데이터 리스트
        
    Returns:
        BM25Retriever: 생성된 BM25 리트리버
    """
    try:
        # Document 객체 리스트 생성
        documents = []
        for text, metadata in zip(texts, metadatas):
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        # BM25Retriever 생성
        bm25_retriever = BM25Retriever.from_documents(documents)
        
        logger.info(f"BM25 리트리버가 성공적으로 생성되었습니다. 문서 수: {len(documents)}")
        return bm25_retriever
        
    except Exception as e:
        logger.error(f"BM25 리트리버 생성 중 오류 발생: {str(e)}")
        raise


def build_ensemble_retriever(
    faiss_path: str,
    bm25_texts: List[str],
<<<<<<< HEAD
    bm25_metadatas: List[dict],
    embedding_model: BaseEmbedding = None
=======
    bm25_metadatas: List[dict]
>>>>>>> 60b74fa (2)
) -> EnsembleRetriever:
    """
    FAISS와 BM25 리트리버를 결합한 앙상블 리트리버를 구축합니다.
    
    Args:
        faiss_path: FAISS 인덱스가 저장된 디렉토리 경로
        bm25_texts: BM25 리트리버용 텍스트 리스트
        bm25_metadatas: BM25 리트리버용 메타데이터 리스트
<<<<<<< HEAD
        embedding_model: 임베딩 모델 (기본값: None, 내장 모델 사용)
=======

>>>>>>> 60b74fa (2)
        
    Returns:
        EnsembleRetriever: 구성된 앙상블 리트리버
        
    Raises:
        ValueError: 입력 파라미터가 유효하지 않은 경우
        FileNotFoundError: FAISS 인덱스 파일을 찾을 수 없는 경우
    """
    try:
        # 입력 검증
        if not faiss_path or not os.path.exists(faiss_path):
            raise ValueError(f"유효하지 않은 FAISS 경로: {faiss_path}")
        
        if not bm25_texts or not bm25_metadatas:
            raise ValueError("BM25 텍스트와 메타데이터는 비어있을 수 없습니다")
        
        if len(bm25_texts) != len(bm25_metadatas):
            raise ValueError("BM25 텍스트와 메타데이터의 길이가 일치하지 않습니다")
        
<<<<<<< HEAD
        # FAISS 리트리버 로드
        faiss_retriever = load_faiss_index(faiss_path)
=======
        # FAISS vectorstore 로드
        faiss_vectorstore = load_faiss_index(faiss_path)
        
        # FAISS 리트리버 생성
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 10})
>>>>>>> 60b74fa (2)
        
        # BM25 리트리버 생성
        bm25_retriever = create_bm25_retriever(bm25_texts, bm25_metadatas)
        
        # 앙상블 리트리버 구성
        # 가중치: FAISS 60%, BM25 40%
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
        
        logger.info("앙상블 리트리버가 성공적으로 구성되었습니다")
        logger.info(f"FAISS 가중치: 0.6, BM25 가중치: 0.4")
        
        return ensemble_retriever
        
    except Exception as e:
        logger.error(f"앙상블 리트리버 구축 중 오류 발생: {str(e)}")
        raise


def format_query_for_embedding(query: str) -> str:
    """
    사용자 질문을 임베딩 모델에 적합한 형식으로 변환합니다.
    
    Args:
        query: 사용자 질문
        
    Returns:
        str: "query: {질문}" 형식으로 변환된 질문
    """
    return f"query: {query}"


# 사용 예시 함수
def example_usage():
    """
    앙상블 리트리버 사용 예시
    """
    # FAISS 경로
    faiss_path = "/home/james4u1/NXJ_RAG/NXJ_Embed/emb"
    
    # BM25용 샘플 데이터 (실제로는 실제 문서 데이터를 사용해야 함)
    sample_texts = [
        "샘플 문서 1의 내용입니다.",
        "샘플 문서 2의 내용입니다.",
        "샘플 문서 3의 내용입니다."
    ]
    
    sample_metadatas = [
        {"source": "doc1", "page": 1},
        {"source": "doc2", "page": 2},
        {"source": "doc3", "page": 3}
    ]
    
    try:
        # 앙상블 리트리버 구축
        ensemble_retriever = build_ensemble_retriever(
            faiss_path=faiss_path,
            bm25_texts=sample_texts,
            bm25_metadatas=sample_metadatas
        )
        
        # 검색 테스트
        query = "샘플 문서"
        formatted_query = format_query_for_embedding(query)
        
        results = ensemble_retriever.get_relevant_documents(formatted_query)
        
        print(f"질문: {query}")
        print(f"검색 결과 수: {len(results)}")
        for i, doc in enumerate(results[:3]):  # 상위 3개 결과만 출력
            print(f"결과 {i+1}: {doc.page_content[:100]}...")
            print(f"메타데이터: {doc.metadata}")
            print()
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    example_usage() 