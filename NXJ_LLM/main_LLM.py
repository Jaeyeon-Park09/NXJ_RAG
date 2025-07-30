"""
LLM RAG System Main Pipeline

NXJ_Retriever와 NXJ_LLM을 결합한 완전한 LLM RAG 시스템의 메인 파이프라인입니다.
F5로 실행하면 전체 시스템이 가동됩니다.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# NXJ_Retriever와 NXJ_LLM 패키지 import
sys.path.append("/home/james4u1/NXJ_RAG/NXJ_Retriever")
sys.path.append("/home/james4u1/NXJ_RAG/NXJ_LLM")

# NXJ_Retriever 패키지 import
sys.path.insert(0, "/home/james4u1/NXJ_RAG")
from NXJ_Retriever import (
    build_ensemble_retriever,
    create_long_context_reorder,
    build_llm_extractor,
    build_compressor_pipeline,
    build_compressed_retriever,
    load_metadata_sample
)

# NXJ_LLM 패키지 import
from NXJ_LLM import (
    build_llm,
    build_report_prompt,
    build_qa_chain,
    run_qa_chain,
    format_qa_response
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMRAGSystem:
    """
    완전한 LLM RAG 시스템 클래스
    """
    
    def __init__(self, model_name: str = "command-r:35b"):
        """
        LLM RAG 시스템을 초기화합니다.
        
        Args:
            model_name: 사용할 LLM 모델 이름
        """
        self.model_name = model_name
        self.llm = None
        self.ensemble_retriever = None
        self.compressed_retriever = None
        self.qa_chain = None
        self.is_initialized = False
        
        logger.info(f"LLM RAG 시스템 초기화 시작: {model_name}")
    
    def initialize_system(self, sample_size: int = 1000):
        """
        시스템의 모든 컴포넌트를 초기화합니다.
        
        Args:
            sample_size: BM25용 샘플 문서 수
        """
        try:
            logger.info("=== LLM RAG 시스템 초기화 시작 ===")
            
            # 1. LLM 초기화
            logger.info("1. LLM 초기화 중...")
            self.llm = build_llm(self.model_name)
            logger.info(f"   LLM 초기화 완료: {self.model_name}")
            
            # 2. Ensemble Retriever 초기화
            logger.info("2. Ensemble Retriever 초기화 중...")
            faiss_path = "/home/james4u1/NXJ_RAG/NXJ_Embed/emb"
            metadata_path = os.path.join(faiss_path, "metadata.json")
            
            # BM25용 데이터 로드
            bm25_texts, bm25_metadatas = load_metadata_sample(
                metadata_path, sample_size=sample_size
            )
            
            # Ensemble Retriever 구축
            self.ensemble_retriever = build_ensemble_retriever(
                faiss_path=faiss_path,
                bm25_texts=bm25_texts,
                bm25_metadatas=bm25_metadatas
            )
            logger.info(f"   Ensemble Retriever 초기화 완료: {len(bm25_texts)}개 문서")
            
            # 3. 문서 압축 파이프라인 구성
            logger.info("3. 문서 압축 파이프라인 구성 중...")
            
            # LongContextReorder 생성
            reorder = create_long_context_reorder()
            
            # LLMChainExtractor 생성
            extractor = build_llm_extractor(self.llm)
            
            # DocumentCompressorPipeline 구성
            compressor_pipeline = build_compressor_pipeline(reorder, extractor)
            
            # ContextualCompressionRetriever 구성
            self.compressed_retriever = build_compressed_retriever(
                base_retriever=self.ensemble_retriever,
                compressor=compressor_pipeline
            )
            logger.info("   문서 압축 파이프라인 구성 완료")
            
            # 4. RetrievalQA 체인 구성
            logger.info("4. RetrievalQA 체인 구성 중...")
            self.qa_chain = build_qa_chain(
                llm=self.llm,
                retriever=self.compressed_retriever
            )
            logger.info("   RetrievalQA 체인 구성 완료")
            
            self.is_initialized = True
            logger.info("=== LLM RAG 시스템 초기화 완료 ===")
            
        except Exception as e:
            logger.error(f"시스템 초기화 중 오류 발생: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변을 생성합니다.
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict[str, Any]: 답변 결과
        """
        try:
            if not self.is_initialized:
                raise ValueError("시스템이 초기화되지 않았습니다. initialize_system()을 먼저 호출하세요.")
            
            if not question or not question.strip():
                raise ValueError("질문은 비어있을 수 없습니다.")
            
            logger.info(f"질문 처리 시작: '{question}'")
            
            # RetrievalQA 체인 실행
            result = run_qa_chain(self.qa_chain, question)
            
            # 결과 포맷팅
            formatted_response = format_qa_response(result)
            
            logger.info("질문 처리 완료")
            
            return {
                "question": question,
                "answer": result.get("result", ""),
                "source_documents": result.get("source_documents", []),
                "formatted_response": formatted_response
            }
            
        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생: {str(e)}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        시스템 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 시스템 정보
        """
        try:
            info = {
                "model_name": self.model_name,
                "is_initialized": self.is_initialized,
                "llm_type": type(self.llm).__name__ if self.llm else None,
                "ensemble_retriever_type": type(self.ensemble_retriever).__name__ if self.ensemble_retriever else None,
                "compressed_retriever_type": type(self.compressed_retriever).__name__ if self.compressed_retriever else None,
                "qa_chain_type": type(self.qa_chain).__name__ if self.qa_chain else None,
            }
            
            return info
            
        except Exception as e:
            logger.error(f"시스템 정보 수집 중 오류 발생: {str(e)}")
            return {"error": str(e)}


def run_interactive_mode():
    """
    대화형 모드로 시스템을 실행합니다.
    """
    try:
        print("=" * 60)
        print("LLM RAG 시스템 시작")
        print("=" * 60)
        
        # 시스템 초기화
        print("시스템 초기화 중... (시간이 걸릴 수 있습니다)")
        rag_system = LLMRAGSystem()

        
        # 시스템 정보 출력
        print("\n시스템 정보:")
        info = rag_system.get_system_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n안녕하세요❗ 의료제품 인허가를 지원하기 위해 개발된 인허스턴트 NXJ_LLM입니다☺️ \n종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("-" * 60)
        
        # 대화형 루프
        while True:
            try:
                # 사용자 입력 받기
                question = input("\n질문을 입력하세요: ").strip()
                
                # 종료 조건 확인
                if question.lower() in ['quit', 'exit', '종료']:
                    print("시스템을 종료합니다.")
                    break
                
                if not question:
                    print("질문을 입력해주세요.")
                    continue
                
                # 질문 처리
                print("\n답변 생성 중...")
                result = rag_system.ask_question(question)
                
                # 결과 출력
                print("\n" + "=" * 40)
                print("답변:")
                print(result["formatted_response"])
                print("=" * 40)
                
            except KeyboardInterrupt:
                print("\n\n시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {str(e)}")
                continue
        
    except Exception as e:
        print(f"시스템 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    """
    메인 실행 함수
    F5로 실행하면 전체 LLM RAG 시스템이 가동됩니다.
    """
    try:
        print("NXJ_LLM RAG 시스템을 시작합니다...")
        print("의료 인허가 어시스턴트 모드를 시작합니다...")
        run_interactive_mode()
            
    except Exception as e:
        print(f"시스템 시작 중 오류 발생: {str(e)}")
        print("프로그램을 종료합니다.") 