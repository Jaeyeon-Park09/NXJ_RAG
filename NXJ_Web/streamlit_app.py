"""
의약품 인허가 가이드 - Streamlit 웹 인터페이스

사용자가 웹 상에서 질문을 입력하면, 구성된 RAG 파이프라인을 통해 응답을 생성합니다.
"""

import streamlit as st
import sys
import os
from typing import Optional, Dict, Any, List

# NXJ_RAG 워크스페이스 내 모듈들 import

# 상위 디렉토리들을 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 각 모듈 디렉토리를 Python 경로에 추가
nxj_llm_path = os.path.join(parent_dir, 'NXJ_LLM')
nxj_retriever_path = os.path.join(parent_dir, 'NXJ_Retriever')
nxj_embed_path = os.path.join(parent_dir, 'NXJ_Embed')
nxj_parser_path = os.path.join(parent_dir, 'NXJ_Parser_Text')

sys.path.extend([nxj_llm_path, nxj_retriever_path, nxj_embed_path, nxj_parser_path])

# 필요한 모듈들 직접 import
try:
    # NXJ_LLM 모듈들
    from llm_wrapper import build_llm
    from prompt_templates import build_report_prompt
    from utils.retrieval_qa_chain_builder import (
        build_qa_chain, 
        build_default_qa_chain, 
        query_with_qa_chain,
        build_qa_chain_with_nxj_retriever_full,
        build_qa_chain_with_advanced_retrieval
    )
    from enhanced_qa import format_enhanced_response
    
except ImportError as e:
    st.error(f"모듈 import 오류: {str(e)}")
    st.error("모든 의존성 패키지가 설치되었는지 확인해주세요.")
    st.stop()


def initialize_rag_pipeline():
    """
    NXJ_Retriever의 모든 기능을 활용한 RAG 파이프라인을 초기화합니다.
    
    Returns:
        구성된 qa_chain 객체
    """
    try:
        # 1. LLM 구성
        llm = build_llm(
            model_name="command-r:35b",
            temperature=0.1
        )
        
        # 2. NXJ_Retriever의 모든 기능을 활용한 QA 체인 구성
        qa_chain = build_qa_chain_with_nxj_retriever_full(
            llm=llm,
            faiss_path="/home/james4u1/NXJ_RAG/NXJ_Embed/emb",
            bm25_texts=[
                "의약품 인허가 신청 절차에 대한 가이드라인입니다.",
                "임상시험 데이터 제출 요구사항을 설명합니다.",
                "안전성 및 유효성 평가 기준을 제시합니다."
            ],
            bm25_metadatas=[
                {"source": "guide1.pdf", "page": 1},
                {"source": "guide2.pdf", "page": 2},
                {"source": "guide3.pdf", "page": 3}
            ],
            retriever_type="complete",  # 완전한 ContextualCompressionRetriever 사용
            embedding_device="cpu",
            model_name="command-r:35b",
            use_reorder=True,      # LongContextReorder 사용
            use_extractor=True,    # LLMChainExtractor 사용
            ensemble_weights=[0.6, 0.4]  # FAISS 60%, BM25 40%
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"RAG 파이프라인 초기화 실패: {str(e)}")
        return None


def initialize_rag_pipeline_advanced(
    retriever_type: str = "complete",
    use_reorder: bool = True,
    use_extractor: bool = True,
    ensemble_weights: List[float] = None,
    model_name: str = "command-r:35b",
    temperature: float = 0.1
):
    """
    고급 설정을 사용하여 NXJ_Retriever의 모든 기능을 활용한 RAG 파이프라인을 초기화합니다.
    
    Args:
        retriever_type: retriever 타입
        use_reorder: LongContextReorder 사용 여부
        use_extractor: LLMChainExtractor 사용 여부
        ensemble_weights: Ensemble 가중치
        model_name: LLM 모델명
        temperature: Temperature 값
    
    Returns:
        구성된 qa_chain 객체
    """
    try:
        # 1. LLM 구성
        llm = build_llm(
            model_name=model_name,
            temperature=temperature
        )
        
        # 2. NXJ_Retriever의 모든 기능을 활용한 QA 체인 구성
        qa_chain = build_qa_chain_with_nxj_retriever_full(
            llm=llm,
            faiss_path="/home/james4u1/NXJ_RAG/NXJ_Embed/emb",
            bm25_texts=[
                "의약품 인허가 신청 절차에 대한 가이드라인입니다.",
                "임상시험 데이터 제출 요구사항을 설명합니다.",
                "안전성 및 유효성 평가 기준을 제시합니다."
            ],
            bm25_metadatas=[
                {"source": "guide1.pdf", "page": 1},
                {"source": "guide2.pdf", "page": 2},
                {"source": "guide3.pdf", "page": 3}
            ],
            retriever_type=retriever_type,
            embedding_device="cpu",
            model_name=model_name,
            use_reorder=use_reorder,
            use_extractor=use_extractor,
            ensemble_weights=ensemble_weights
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"고급 RAG 파이프라인 초기화 실패: {str(e)}")
        return None


def process_question(qa_chain, question: str) -> Optional[Dict[str, Any]]:
    """
    질문을 처리하여 응답을 생성합니다.
    
    Args:
        qa_chain: RetrievalQA 체인 객체
        question: 사용자 질문
    
    Returns:
        응답 결과 딕셔너리 또는 None
    """
    try:
        # 질문이 비어있는지 확인
        if not question or not question.strip():
            return None
        
        # RAG 파이프라인을 통한 질의응답 수행
        result = query_with_qa_chain(
            qa_chain=qa_chain,
            question=question.strip(),
            k=4
        )
        
        return result
        
    except Exception as e:
        st.error(f"질문 처리 중 오류 발생: {str(e)}")
        return None


def display_response(result: Dict[str, Any]):
    """
    응답 결과를 화면에 표시합니다.
    
    Args:
        result: 질의응답 결과 딕셔너리
    """
    if not result:
        return
    
    # 질문 표시
    st.subheader("질문")
    st.write(result.get('query', 'N/A'))
    
    # 답변 표시
    st.subheader("답변")
    answer = result.get('result', 'N/A')
    
    # 답변을 단락으로 분리하여 표시
    if answer and answer != 'N/A':
        # 줄바꿈을 기준으로 단락 분리
        paragraphs = answer.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                st.write(paragraph.strip())
                st.write("")  # 단락 간 간격
    else:
        st.write("답변을 생성할 수 없습니다.")
    
    # 출처 문서 표시
    if 'source_documents' in result and result['source_documents']:
        st.subheader("참고 문서")
        for i, doc in enumerate(result['source_documents'], 1):
            with st.expander(f"문서 {i}"):
                st.write(f"**내용:** {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.write(f"**메타데이터:** {doc.metadata}")


def main():
    """
    Streamlit 애플리케이션의 메인 함수
    """
    # 페이지 설정
    st.set_page_config(
        page_title="NXJ RAG 시스템",
        page_icon="🔍",
        layout="wide"
    )
    
    # 페이지 제목
    st.title("🔍 NXJ RAG 시스템")
    st.markdown("---")
    
    # 사이드바에 고급 설정 추가
    with st.sidebar:
        st.header("🔧 고급 설정")
        
        # Retriever 타입 선택
        retriever_type = st.selectbox(
            "Retriever 타입",
            options=[
                "complete",           # 완전한 ContextualCompressionRetriever
                "ensemble",           # 기본 Ensemble Retriever
                "with_reorder",       # 재정렬 기능 포함
                "with_extractor",     # 추출 기능 포함
                "with_pipeline",      # 압축 파이프라인 포함
                "custom"              # 사용자 정의
            ],
            index=0,
            help="사용할 retriever의 타입을 선택하세요"
        )
        
        # 압축 기능 설정 (custom 타입일 때만)
        if retriever_type == "custom":
            use_reorder = st.checkbox("LongContextReorder 사용", value=True)
            use_extractor = st.checkbox("LLMChainExtractor 사용", value=True)
        else:
            use_reorder = True
            use_extractor = True
        
        # Ensemble 가중치 설정
        st.subheader("Ensemble 가중치")
        faiss_weight = st.slider("FAISS 가중치", 0.0, 1.0, 0.6, 0.1)
        bm25_weight = 1.0 - faiss_weight
        st.write(f"BM25 가중치: {bm25_weight:.1f}")
        
        # 모델 설정
        st.subheader("모델 설정")
        model_name = st.selectbox(
            "LLM 모델",
            options=["command-r:35b", "llama3.2:3b", "qwen2.5:7b"],
            index=0
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        # 파이프라인 재구성 버튼
        if st.button("파이프라인 재구성", type="secondary"):
            st.session_state.rebuild_pipeline = True
        
        st.markdown("---")
        st.header("사용법")
        st.markdown("""
        1. 고급 설정에서 원하는 retriever 타입을 선택하세요
        2. 질문을 입력하고 '질문 제출' 버튼을 클릭하세요
        3. AI가 NXJ_Retriever의 모든 기능을 활용하여 답변을 생성합니다
        """)
        
        st.header("예시 질문")
        st.markdown("""
        - 의약품 인허가 신청 절차는 어떻게 되나요?
        - 임상시험 데이터는 어떤 형식으로 제출해야 하나요?
        - 안전성 평가 기준은 무엇인가요?
        """)
    
    # RAG 파이프라인 초기화 또는 재구성
    if 'qa_chain' not in st.session_state or st.session_state.get('rebuild_pipeline', False):
        with st.spinner("RAG 파이프라인을 구성하고 있습니다..."):
            qa_chain = initialize_rag_pipeline_advanced(
                retriever_type=retriever_type,
                use_reorder=use_reorder,
                use_extractor=use_extractor,
                ensemble_weights=[faiss_weight, bm25_weight],
                model_name=model_name,
                temperature=temperature
            )
            st.session_state.qa_chain = qa_chain
            st.session_state.rebuild_pipeline = False
    else:
        qa_chain = st.session_state.qa_chain
    
    if qa_chain is None:
        st.error("RAG 파이프라인 초기화에 실패했습니다.")
        return
    
    st.success("✅ NXJ_Retriever의 모든 기능을 활용한 RAG 파이프라인이 구성되었습니다!")
    
    # 질문 입력 섹션
    st.header("질문 입력")
    
    # 텍스트 입력창
    question = st.text_area(
        "질문을 입력하세요:",
        placeholder="예: 의약품 인허가 신청 절차는 어떻게 되나요?",
        height=100,
        max_chars=1000
    )
    
    # 제출 버튼
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button(
            "질문 제출",
            type="primary",
            use_container_width=True
        )
    
    # 질문 처리 및 응답 생성
    if submit_button:
        if not question or not question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            # 로딩 스피너 표시
            with st.spinner("질문을 처리하는 중..."):
                result = process_question(qa_chain, question)
            
            if result:
                # 응답 표시
                st.markdown("---")
                display_response(result)
            else:
                st.error("응답을 생성할 수 없습니다. 다시 시도해주세요.")
    
    # 하단 정보
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>NXJ RAG 시스템 - NXJ_Retriever의 모든 기능을 활용한 고성능 질의응답 시스템</p>
            <p>Powered by Ensemble Retriever + Contextual Compression</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 