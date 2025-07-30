#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
from typing import List, Dict, Any, Tuple

class SearchTool:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", 
                 index_dir: str = "/home/james4u1/NXJ_RAG/NXJ_Embed/emb"):
        """
        검색 툴 초기화
        
        Args:
            model_name: 임베딩 모델명
            index_dir: FAISS 인덱스가 저장된 디렉토리
        """
        self.model_name = model_name
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.metadata = None
        
        # 모델 로드
        print(f"임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 인덱스와 메타데이터 로드
        self.load_index_and_metadata()
    
    def load_index_and_metadata(self):
        """FAISS 인덱스와 메타데이터 로드"""
        try:
            import faiss
        except ImportError:
            print("FAISS가 설치되지 않았습니다. pip install faiss-cpu 또는 faiss-gpu를 설치하세요.")
            raise
            
        # FAISS 인덱스 로드
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"FAISS 인덱스 로드 완료: {self.index.ntotal}개 벡터")
        
        # 메타데이터 로드
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"메타데이터 로드 완료: {len(self.metadata)}개 문서")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        쿼리 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        try:
            import faiss
        except ImportError:
            print("FAISS가 설치되지 않았습니다. pip install faiss-cpu 또는 faiss-gpu를 설치하세요.")
            raise
            
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # 정규화
        faiss.normalize_L2(query_embedding)
        
        # 검색 실행
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10  # IVF 인덱스의 경우
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'text': self.metadata[idx]['text'],
                    'source_file': self.metadata[idx]['source_file'],
                    'path': self.metadata[idx]['path'],
                    'length': self.metadata[idx]['length']
                }
                results.append(result)
        
        return results
    
    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        배치 쿼리 검색
        
        Args:
            queries: 검색 쿼리 리스트
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트의 리스트
        """
        try:
            import faiss
        except ImportError:
            print("FAISS가 설치되지 않았습니다. pip install faiss-cpu 또는 faiss-gpu를 설치하세요.")
            raise
            
        # 쿼리 임베딩 생성
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        
        # 정규화
        faiss.normalize_L2(query_embeddings)
        
        # 검색 실행
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10
        
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # 결과 구성
        all_results = []
        for query_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_results = []
            for i, (score, idx) in enumerate(zip(query_scores, query_indices)):
                if idx < len(self.metadata):
                    result = {
                        'rank': i + 1,
                        'score': float(score),
                        'text': self.metadata[idx]['text'],
                        'source_file': self.metadata[idx]['source_file'],
                        'path': self.metadata[idx]['path'],
                        'length': self.metadata[idx]['length']
                    }
                    query_results.append(result)
            all_results.append(query_results)
        
        return all_results
    
    def print_results(self, results: List[Dict[str, Any]], query: str = ""):
        """
        검색 결과 출력
        
        Args:
            results: 검색 결과
            query: 검색 쿼리
        """
        if query:
            print(f"\n검색 쿼리: {query}")
            print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"\n[{result['rank']}] 점수: {result['score']:.4f}")
            print(f"파일: {result['source_file']}")
            print(f"경로: {result['path']}")
            print(f"텍스트 길이: {result['length']}자")
            print(f"내용: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
            print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="문서 검색 툴")
    parser.add_argument("--query", type=str, required=True,
                       help="검색 쿼리")
    parser.add_argument("--top_k", type=int, default=10,
                       help="반환할 결과 수")
    parser.add_argument("--index_dir", type=str,
                       default="/home/james4u1/NXJ_RAG/NXJ_Embed/emb",
                       help="FAISS 인덱스 디렉토리")
    parser.add_argument("--model_name", type=str,
                       default="intfloat/multilingual-e5-base",
                       help="임베딩 모델명")
    
    args = parser.parse_args()
    
    try:
        # 검색 툴 초기화
        search_tool = SearchTool(args.model_name, args.index_dir)
        
        # 검색 실행
        results = search_tool.search(args.query, args.top_k)
        
        # 결과 출력
        search_tool.print_results(results, args.query)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    query = input("검색어를 입력하세요: ")
    search_tool = SearchTool()
    results = search_tool.search(query)
    search_tool.print_results(results, query) 