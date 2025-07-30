#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Tuple
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/james4u1/NXJ_RAG/NXJ_Embed/embedding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingTool:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        """
        임베딩 툴 초기화
        
        Args:
            model_name: 사용할 임베딩 모델명
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        logger.info(f"임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("임베딩 모델 로딩 완료")
    
    def load_json_files(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        JSON 파일들을 로드하고 텍스트 데이터 추출
        
        Args:
            input_dir: JSON 파일들이 있는 디렉토리 경로
            
        Returns:
            추출된 텍스트 데이터 리스트
        """
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        logger.info(f"발견된 JSON 파일 수: {len(json_files)}")
        
        all_texts = []
        
        for json_file in tqdm(json_files, desc="JSON 파일 로딩"):
            file_path = os.path.join(input_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # JSON 데이터에서 텍스트 추출
                texts = self._extract_texts_from_json(data, json_file)
                all_texts.extend(texts)
                
                logger.info(f"{json_file}: {len(texts)}개 텍스트 청크 추출")
                
            except Exception as e:
                logger.error(f"파일 로딩 실패 {json_file}: {str(e)}")
                continue
        
        logger.info(f"총 {len(all_texts)}개 텍스트 청크 추출 완료")
        return all_texts
    
    def _extract_texts_from_json(self, data: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
        """
        JSON 데이터에서 텍스트 청크들을 추출
        
        Args:
            data: JSON 데이터
            filename: 원본 파일명
            
        Returns:
            텍스트 청크 리스트
        """
        texts = []
        
        def extract_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    extract_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    extract_recursive(item, current_path)
            elif isinstance(obj, str) and obj.strip():
                # 텍스트가 너무 짧으면 건너뛰기
                if len(obj.strip()) < 8:
                    return
                
                texts.append({
                    'text': obj.strip(),
                    'source_file': filename,
                    'path': path,
                    'length': len(obj.strip())
                })
        
        extract_recursive(data)
        return texts
    
    def create_embeddings(self, texts: List[Dict[str, Any]], batch_size: int = 64) -> np.ndarray:
        """
        텍스트들을 임베딩으로 변환
        
        Args:
            texts: 텍스트 데이터 리스트
            batch_size: 배치 크기
            
        Returns:
            임베딩 벡터 배열
        """
        text_list = [item['text'] for item in texts]
        
        logger.info(f"임베딩 생성 시작: {len(text_list)}개 텍스트")
        
        embeddings = []
        
        for i in tqdm(range(0, len(text_list), batch_size), desc="임베딩 생성"):
            batch_texts = text_list[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        logger.info(f"임베딩 생성 완료: {embeddings.shape}")
        
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "IVFFlat"):
        """
        FAISS 인덱스 생성 (CPU 버전)
        
        Args:
            embeddings: 임베딩 벡터 배열
            index_type: 인덱스 타입 (예: "IVFFlat", "Flat")
            
        Returns:
            FAISS 인덱스 (CPU)
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS가 설치되지 않았습니다. pip install faiss-cpu를 설치하세요.")
            raise
         
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)

        if index_type == "IVFFlat":
            nlist = min(100, len(embeddings) // 30)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            if len(embeddings) >= nlist:
                index.train(embeddings)

            index.add(embeddings)

        else:
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

        logger.info(f"FAISS 인덱스 생성 완료 (CPU): {index.ntotal}개 벡터")
        return index

    
    def save_index_and_metadata(self, index, texts: List[Dict[str, Any]], 
                               output_dir: str):
        """
        인덱스와 메타데이터 저장
        
        Args:
            index: FAISS 인덱스
            texts: 텍스트 메타데이터
            output_dir: 출력 디렉토리
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS가 설치되지 않았습니다. pip install faiss-cpu를 설치하세요.")
            raise
            
        # FAISS 인덱스 저장
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        logger.info(f"FAISS 인덱스 저장: {index_path}")
        
        # 메타데이터 저장
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        logger.info(f"메타데이터 저장: {metadata_path}")
        
        # 통계 정보 저장
        stats = {
            "total_documents": len(texts),
            "embedding_dimension": index.d,
            "index_type": type(index).__name__,
            "model_name": self.model_name
        }
        
        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"통계 정보 저장: {stats_path}")
    
    def process(self, input_dir: str, output_dir: str, batch_size: int = 64):
        """
        전체 임베딩 프로세스 실행
        
        Args:
            input_dir: 입력 JSON 파일 디렉토리
            output_dir: 출력 디렉토리
            batch_size: 배치 크기
        """
        logger.info("임베딩 프로세스 시작")
        
        # 1. JSON 파일들 로드
        texts = self.load_json_files(input_dir)
        
        if not texts:
            logger.error("처리할 텍스트가 없습니다.")
            return
        
        # 2. 임베딩 생성
        embeddings = self.create_embeddings(texts, batch_size)
        
        # 3. FAISS 인덱스 생성
        index = self.create_faiss_index(embeddings)
        
        # 4. 결과 저장
        self.save_index_and_metadata(index, texts, output_dir)
        
        logger.info("임베딩 프로세스 완료")

def main():
    parser = argparse.ArgumentParser(description="문서 임베딩 툴")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/james4u1/NXJ_RAG/NXJ_Parser_Text/output",
                       help="입력 JSON 파일 디렉토리")
    parser.add_argument("--output_dir", type=str,
                       default="/home/james4u1/NXJ_RAG/NXJ_Embed/emb",
                       help="출력 디렉토리")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="배치 크기")
    parser.add_argument("--model_name", type=str,
                       default="intfloat/multilingual-e5-base",
                       help="임베딩 모델명")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 임베딩 툴 실행
    tool = EmbeddingTool(args.model_name)
    tool.process(args.input_dir, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main() 