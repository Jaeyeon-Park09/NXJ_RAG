# NXJ_RAG: 의료제품 인허가 지원 RAG 시스템

## 📋 프로젝트 개요

NXJ_RAG는 의료제품 인허가 지원을 위한 LLM 챗봇입니다. LangChain을 기반으로 구축된 RAG(Retrieval-Augmented Generation) 시스템으로, 문서 검색과 LLM을 결합하여 정확하고 신뢰할 수 있는 답변을 제공합니다.

## 🏗️ 시스템 아키텍처

```
NXJ_RAG/
├── NXJ_Parser_Text/   # PDF 파싱 툴 (텍스트 한정)
├── NXJ_Embed/         # 문서 임베딩 및 벡터 저장소
├── NXJ_Retriever/     # 문서 검색 및 압축 시스템
├── NXJ_LLM/           # LLM 통합 및 QA 체인
└── NXJ_Web/           # Streamlit 웹 인터페이스
```

### 전체 RAG 파이프라인

```
사용자 질문 → Ensemble Retriever → Document Compression → LLM → 답변
                ↓
        [Sparse + Dense] → [LongContextReorder + LLMChainExtractor] → [Ollama LLM]
```

## 🔧 주요 컴포넌트

### 1. NXJ_Parser_Text: PDF 파싱 시스템

**기능:**
- PDF 문서 텍스트 추출
- 텍스트 전처리 및 정제
- 메타데이터 추출 (페이지 번호, 파일명 등)
- 다국어 지원 (한국어, 영어)

**주요 특징:**
- **PyPDF2** 기반 PDF 파싱
- **정규표현식**을 통한 텍스트 정제
- **SentenceWindowNodeParser**를 통한 문맥 보존 청킹
- **메타데이터 보존**으로 출처 추적

**SentenceWindowNodeParser 원리:**
- **문장 단위 분할**: 자연스러운 문장 경계에서 문서를 분할
- **컨텍스트 윈도우**: 각 노드(=문장) 주변에 이전/다음 문장들을 포함하여 문맥 보존
- **중복 제거**: 윈도우 간 중복되는 문장들을 효율적으로 처리
- **문맥 연속성**: 문장 간의 논리적 연결성을 유지하여 검색 품질 향상

**SentenceWindowNodeParser 하이퍼파라미터:**
- **window_size**: 3 (각 노드 전후로 3개 문장씩 포함)
- **window_metadata_key**: "window" (윈도우 정보 저장 키)
- **original_text_metadata_key**: "original_text" (원본 텍스트 저장 키)
- **sentence_splitter**: 한국어/영어 문장 분할기

### 2. NXJ_Embed: 문서 임베딩 시스템

**사용 모델:**
- **임베딩 모델**: `intfloat/multilingual-e5-base`
- **언어**: 다국어 지원 (한국어, 영어)
- **차원**: 768차원

**주요 하이퍼파라미터:**
- **SentenceWindowNodeParser**: window_size=2, 문맥 보존 청킹
- **임베딩 모델**: intfloat/multilingual-e5-base
- **Device**: CPU (GPU 지원 가능)
- **Normalize**: True
- **Chunk Overlap**: 문장 단위 자동 처리

**기능:**
- PDF 문서 텍스트 추출
- SentenceWindowNodeParser를 통한 문맥 보존 청킹
- 임베딩 생성 및 벡터화
- FAISS 벡터 인덱스 구축
- 메타데이터 저장 (윈도우 정보 포함)

### 3. NXJ_Retriever: 문서 검색 시스템

#### 3.1 Ensemble Retriever

**구성 요소:**
- **FAISS Retriever**: 벡터 유사도 검색 - 의미적 유사성 기반 검색에 효과적
- **BM25 Retriever**: 키워드 기반 검색 - 키워드 기반 검색에 효과적

**하이퍼파라미터:**
- **FAISS 가중치**: 0.6
- **BM25 가중치**: 0.4
- **Top-k**: 10개 문서

#### 3.2 Document Compression Pipeline

**구성 요소:**
1. **LongContextReorder**: 문서 중요도 순 재정렬
2. **LLMChainExtractor**: 관련 내용 추출

**압축 파이프라인 순서:**
```
문서 → LongContextReorder → LLMChainExtractor → 압축된 문서
```

#### 3.3 Contextual Compression Retriever

**기능:**
- 검색된 문서를 질문에 맞게 압축
- 관련성 높은 내용만 추출
- 컨텍스트 품질 향상

### 4. NXJ_LLM: LLM 통합 시스템

**사용 모델:**
- **기본 모델**: `command-r:35b` (Ollama)

**주요 하이퍼파라미터:**
- **Temperature**: 0.1 (일관성 있는 답변)
- **Max Tokens**: 4096
- **Top-p**: 0.9
- **Repeat Penalty**: 1.1

**QA 체인 설정:**
- **Chain Type**: "stuff" (문서를 프롬프트에 직접 삽입)
- **Return Source Documents**: True
- **Chain Verbose**: False

### 5. NXJ_Web: 웹 인터페이스

**기술 스택:**
- **Framework**: Streamlit
- **UI**: 반응형 웹 인터페이스
- **실시간 처리**: 비동기 질의응답

**기능:**
- 대화형 질의응답
- 고급 설정 (Retriever 타입, 가중치 조정)
- 결과 시각화
- 소스 문서 표시



## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/Jaeyeon-Park09/NXJ_RAG.git
cd NXJ_RAG

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. Ollama 설정

```bash
# Ollama 설치 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드
ollama pull command-r:35b
```

### 3. 실행

#### 대화형 모드
```bash
cd NXJ_LLM
python main_LLM.py
```

#### 웹 인터페이스
```bash
cd NXJ_Web
streamlit run streamlit_app.py
```


## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해 주세요.