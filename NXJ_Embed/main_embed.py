#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
안정적인 임베딩 툴 실행 스크립트 (메모리 효율적)
"""

import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from typing import List, Dict, Any
import gc
import argparse
import psutil

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

class StableEmbeddingTool:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        """
        고성능 임베딩 툴 초기화 (20GiB 메모리 최적화)
        
        Args:
            model_name: 사용할 임베딩 모델명
        """
        self.model_name = model_name
        self.model = None
        
        logger.info(f"임베딩 모델 로딩 중: {model_name}")
        
        # 성능 최적화를 위한 설정
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용 디바이스: {device}")
        
        # SentenceTransformer 성능 최적화 설정
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=None  # 캐시 폴더 비활성화로 메모리 절약
        )
        
        # 모델을 eval 모드로 설정 (추론 최적화)
        self.model.eval()
        
        # 배치 크기 자동 조정 (20GiB 메모리 제한 고려)
        self.default_batch_size = 128  # CPU/GPU 모두 128로 통일
            
        logger.info(f"임베딩 모델 로딩 완료 (기본 배치 크기: {self.default_batch_size})")
        
        # 메모리 사용량 초기화
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"초기 메모리 사용량: {self.initial_memory:.1f} MB")
    
    def _check_memory_usage(self, stage: str = ""):
        """
        메모리 사용량을 확인하고 경고를 출력
        
        Args:
            stage: 현재 단계 설명
        """
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.initial_memory
        
        logger.info(f"메모리 사용량 {stage}: {current_memory:.1f} MB (증가: {memory_increase:.1f} MB)")
        
        # 16GB (안전 마진) 경고
        if current_memory > 16 * 1024:  # 16GB
            logger.warning(f"메모리 사용량이 높습니다: {current_memory:.1f} MB")
            
        # 18GB 임계값 경고
        if current_memory > 18 * 1024:  # 18GB
            logger.error(f"메모리 사용량이 위험 수준입니다: {current_memory:.1f} MB")
            logger.info("가비지 컬렉션을 강제로 실행합니다.")
            gc.collect()
    
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
        self._check_memory_usage("JSON 파일 로딩 후")
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
                
                text_info = {
                    "text": obj.strip(),
                    "source": filename,
                    "path": path
                }
                texts.append(text_info)
        
        extract_recursive(data)
        return texts
    
    def save_embeddings_incremental(self, texts: List[Dict[str, Any]], output_dir: str, batch_size: int = None):
        """
        임베딩을 고성능으로 생성하고 저장 (20GiB 메모리 활용)
        
        Args:
            texts: 텍스트 데이터 리스트
            output_dir: 출력 디렉토리
            batch_size: 배치 크기 (None이면 자동 조정)
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS가 설치되지 않았습니다. pip install faiss-cpu를 설치하세요.")
            raise
        
        text_list = [item['text'] for item in texts]
        total_texts = len(text_list)
        
        # 배치 크기 자동 조정
        if batch_size is None:
            batch_size = self.default_batch_size
        
        logger.info(f"고성능 임베딩 저장 시작: {total_texts}개 텍스트 (배치 크기: {batch_size})")
        self._check_memory_usage("임베딩 시작 전")
        
        # 첫 번째 배치로 차원 확인
        first_batch = text_list[:min(batch_size, total_texts)]
        first_embeddings = self.model.encode(first_batch, convert_to_numpy=True, show_progress_bar=True)
        dimension = first_embeddings.shape[1]
        
        # FAISS 인덱스 초기화 (더 빠른 인덱스 사용)
        faiss.normalize_L2(first_embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(first_embeddings)
        
        logger.info(f"FAISS 인덱스 초기화 완료 (차원: {dimension})")
        
        # 나머지 배치들 처리 (더 큰 배치로 처리)
        for i in tqdm(range(batch_size, total_texts, batch_size), desc="임베딩 저장"):
            end_idx = min(i + batch_size, total_texts)
            batch_texts = text_list[i:end_idx]
            
            # 임베딩 생성 (진행률 표시 포함)
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(batch_embeddings)
            
            # 인덱스에 추가
            index.add(batch_embeddings)
            
            # 메모리 정리 (더 적극적으로)
            del batch_embeddings
            if i % (batch_size * 5) == 0:  # 5배치마다 가비지 컬렉션
                gc.collect()
                self._check_memory_usage(f"배치 {i//batch_size} 처리 후")
            
            # 진행 상황 로깅 (더 자주)
            if (i // batch_size) % 25 == 0:
                logger.info(f"진행률: {i}/{total_texts} ({i/total_texts*100:.1f}%)")
        
        logger.info(f"FAISS 인덱스 생성 완료: {index.ntotal}개 벡터")
        self._check_memory_usage("FAISS 인덱스 생성 완료 후")
        
        # 결과 저장
        self._save_final_results(index, texts, output_dir)
    
    def save_embeddings_fast(self, texts: List[Dict[str, Any]], output_dir: str, batch_size: int = None):
        """
        최대 성능으로 임베딩 생성 및 저장 (20GiB 메모리 최대 활용)
        
        Args:
            texts: 텍스트 데이터 리스트
            output_dir: 출력 디렉토리
            batch_size: 배치 크기 (None이면 자동 조정)
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS가 설치되지 않았습니다. pip install faiss-cpu를 설치하세요.")
            raise
        
        text_list = [item['text'] for item in texts]
        total_texts = len(text_list)
        
        # 배치 크기 자동 조정 (20GiB 메모리 제한 고려)
        if batch_size is None:
            batch_size = min(self.default_batch_size * 2, 256)  # 최대 256으로 제한 (fast 모드)
        
        logger.info(f"최대 성능 임베딩 저장 시작: {total_texts}개 텍스트 (배치 크기: {batch_size})")
        self._check_memory_usage("fast 모드 시작 전")
        
        # 메모리 안전성을 위한 조건 강화
        max_safe_batch = 512  # 20GiB 기준 안전한 최대 배치 크기
        if total_texts <= batch_size and total_texts <= max_safe_batch:  # 더 엄격한 조건
            logger.info("작은 데이터셋: 모든 임베딩을 한 번에 생성합니다.")
            all_embeddings = self.model.encode(text_list, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(all_embeddings)
            
            # FAISS 인덱스 생성
            dimension = all_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(all_embeddings)
            
            logger.info(f"FAISS 인덱스 생성 완료: {index.ntotal}개 벡터")
            self._check_memory_usage("작은 데이터셋 처리 완료 후")
            
        else:
            # 큰 데이터셋은 큰 배치로 처리
            logger.info("큰 데이터셋: 큰 배치로 처리합니다.")
            
            # 첫 번째 배치로 차원 확인
            first_batch = text_list[:min(batch_size, total_texts)]
            first_embeddings = self.model.encode(first_batch, convert_to_numpy=True, show_progress_bar=True)
            dimension = first_embeddings.shape[1]
            
            # FAISS 인덱스 초기화
            faiss.normalize_L2(first_embeddings)
            index = faiss.IndexFlatIP(dimension)
            index.add(first_embeddings)
            
            logger.info(f"FAISS 인덱스 초기화 완료 (차원: {dimension})")
            
            # 나머지 배치들 처리 (더 큰 배치)
            for i in tqdm(range(batch_size, total_texts, batch_size), desc="임베딩 저장"):
                end_idx = min(i + batch_size, total_texts)
                batch_texts = text_list[i:end_idx]
                
                # 임베딩 생성
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                faiss.normalize_L2(batch_embeddings)
                
                # 인덱스에 추가
                index.add(batch_embeddings)
                
                # 메모리 정리 (덜 자주)
                del batch_embeddings
                if i % (batch_size * 20) == 0:  # 20배치마다 가비지 컬렉션
                    gc.collect()
                
                # 진행 상황 로깅
                if (i // batch_size) % 25 == 0:
                    logger.info(f"진행률: {i}/{total_texts} ({i/total_texts*100:.1f}%)")
        
        # 결과 저장
        self._check_memory_usage("저장 시작 전")
        self._save_final_results(index, texts, output_dir)
    
    def _save_final_results(self, index, texts: List[Dict[str, Any]], output_dir: str):
        """
        최종 결과 저장
        
        Args:
            index: FAISS 인덱스
            texts: 텍스트 메타데이터
            output_dir: 출력 디렉토리
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS가 설치되지 않았습니다.")
            raise
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # FAISS 인덱스 저장
        index_path = os.path.join(output_dir, "faiss_index.bin")
        logger.info(f"FAISS 인덱스 저장 중: {index_path}")
        self._check_memory_usage("FAISS 인덱스 저장 전")
        faiss.write_index(index, index_path)
        logger.info(f"FAISS 인덱스 저장 완료")
        self._check_memory_usage("FAISS 인덱스 저장 후")
        
        # 메타데이터 저장
        metadata_path = os.path.join(output_dir, "metadata.json")
        logger.info(f"메타데이터 저장 중: {metadata_path}")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        logger.info(f"메타데이터 저장 완료")
        
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
        
        logger.info("모든 저장 작업 완료!")

def main():
    parser = argparse.ArgumentParser(description="안정적인 문서 임베딩 툴")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/james4u1/NXJ_RAG/NXJ_Parser_Text/output",
                       help="입력 JSON 파일 디렉토리")
    parser.add_argument("--output_dir", type=str,
                       default="/home/james4u1/NXJ_RAG/NXJ_Embed/emb",
                       help="출력 디렉토리")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="배치 크기 (None이면 자동 조정, 20GiB 메모리 활용)")
    parser.add_argument("--fast_mode", action="store_true",
                       help="최대 성능 모드 (더 큰 배치 크기 사용)")
    parser.add_argument("--model_name", type=str,
                       default="intfloat/multilingual-e5-base",
                       help="임베딩 모델명")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("안정적인 문서 임베딩 툴 실행")
    print("=" * 60)
    print(f"입력 디렉토리: {args.input_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"모델: {args.model_name}")
    print(f"배치 크기: {args.batch_size}")
    print(f"최대 성능 모드: {args.fast_mode}")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 임베딩 툴 실행
    try:
        tool = StableEmbeddingTool(args.model_name)
        
        # JSON 파일들 로드
        texts = tool.load_json_files(args.input_dir)
        
        if not texts:
            logger.error("처리할 텍스트가 없습니다.")
            return
        
        # 임베딩 생성 및 저장 (성능 모드에 따라 선택)
        if args.fast_mode:
            # 최대 성능 모드
            tool.save_embeddings_fast(texts, args.output_dir, args.batch_size)
        else:
            # 고성능 모드
            tool.save_embeddings_incremental(texts, args.output_dir, args.batch_size)
        
        print("\n임베딩 프로세스가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        print(f"\n오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 