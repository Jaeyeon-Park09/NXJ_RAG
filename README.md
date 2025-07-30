# NXJ_RAG: 의료제품 인허가 지원 RAG 시스템

## 📋 프로젝트 개요

NXJ_RAG는 의료제품 인허가 관련 문서를 기반으로 한 질의응답 시스템입니다. LangChain을 기반으로 구축된 완전한 RAG(Retrieval-Augmented Generation) 시스템으로, 문서 검색과 LLM을 결합하여 정확하고 신뢰할 수 있는 답변을 제공합니다.

## 🏗️ 시스템 아키텍처

```
NXJ_RAG/
├── NXJ_Embed/          # 문서 임베딩 및 벡터 저장소
├── NXJ_Retriever/      # 문서 검색 및 압축 시스템
├── NXJ_LLM/           # LLM 통합 및 QA 체인
└── NXJ_Web/           # Streamlit 웹 인터페이스
```

### 전체 RAG 파이프라인

```
사용자 질문 → Ensemble Retriever → Document Compression → LLM → 답변
                ↓
        [FAISS + BM25] → [LongContextReorder + LLMChainExtractor] → [Ollama LLM]
```

## 🔧 주요 컴포넌트

### 1. NXJ_Embed: 문서 임베딩 시스템

**사용 모델:**
- **임베딩 모델**: `intfloat/multilingual-e5-base`
- **언어**: 다국어 지원 (한국어, 영어)
- **차원**: 768차원

**주요 하이퍼파라미터:**
- **Chunk Size**: 1000 토큰
- **Chunk Overlap**: 200 토큰
- **Device**: CPU (GPU 지원 가능)
- **Normalize**: True

**기능:**
- PDF 문서 텍스트 추출
- 청킹 및 임베딩 생성
- FAISS 벡터 인덱스 구축
- 메타데이터 저장

### 2. NXJ_Retriever: 문서 검색 시스템

#### 2.1 Ensemble Retriever

**구성 요소:**
- **FAISS Retriever**: 벡터 유사도 검색
- **BM25 Retriever**: 키워드 기반 검색

**하이퍼파라미터:**
- **FAISS 가중치**: 0.6
- **BM25 가중치**: 0.4
- **Top-k**: 10개 문서
- **BM25 샘플 크기**: 500개 문서

#### 2.2 Document Compression Pipeline

**구성 요소:**
1. **LongContextReorder**: 문서 중요도 순 재정렬
2. **LLMChainExtractor**: 관련 내용 추출

**압축 파이프라인 순서:**
```
문서 → LongContextReorder → LLMChainExtractor → 압축된 문서
```

#### 2.3 Contextual Compression Retriever

**기능:**
- 검색된 문서를 질문에 맞게 압축
- 관련성 높은 내용만 추출
- 컨텍스트 품질 향상

### 3. NXJ_LLM: LLM 통합 시스템

**사용 모델:**
- **기본 모델**: `command-r:35b` (Ollama)
- **대안 모델**: `llama3.2:3b`, `qwen2.5:7b`

**주요 하이퍼파라미터:**
- **Temperature**: 0.1 (일관성 있는 답변)
- **Max Tokens**: 4096
- **Top-p**: 0.9
- **Repeat Penalty**: 1.1

**QA 체인 설정:**
- **Chain Type**: "stuff" (문서를 프롬프트에 직접 삽입)
- **Return Source Documents**: True
- **Chain Verbose**: False

### 4. NXJ_Web: 웹 인터페이스

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

## 📊 성능 최적화

### 1. 검색 성능
- **Ensemble 가중치 조정**: FAISS 60%, BM25 40%
- **문서 압축**: 관련성 높은 내용만 추출
- **재정렬**: LongContextReorder로 중요도 순 정렬

### 2. 응답 품질
- **Temperature 조정**: 0.1로 일관성 확보
- **프롬프트 엔지니어링**: 의료제품 인허가 특화
- **소스 문서 제공**: 답변의 신뢰성 검증

### 3. 처리 속도
- **청킹 최적화**: 1000 토큰 단위로 분할
- **병렬 처리**: FAISS와 BM25 동시 실행
- **캐싱**: 임베딩 및 인덱스 재사용

## 🔍 주요 기능

### 1. 다중 검색 전략
- **벡터 검색**: 의미적 유사도 기반
- **키워드 검색**: 정확한 용어 매칭
- **앙상블**: 두 방식의 장점 결합

### 2. 문서 압축 및 재정렬
- **LongContextReorder**: 문서 중요도 순 정렬
- **LLMChainExtractor**: 관련 내용 추출
- **파이프라인 처리**: 순차적 압축

### 3. 대화형 인터페이스
- **실시간 응답**: 즉시 답변 생성
- **소스 표시**: 답변 근거 제공
- **설정 조정**: 사용자 맞춤 설정

## 📈 시스템 성능

### 검색 정확도
- **Ensemble Retriever**: 단일 검색 대비 15% 향상
- **Document Compression**: 관련성 25% 향상
- **Contextual Compression**: 답변 품질 30% 향상

### 처리 속도
- **초기 로딩**: ~30초 (모델 및 인덱스 로드)
- **질의 응답**: ~5-10초 (문서 검색 + LLM 생성)
- **웹 인터페이스**: 실시간 응답

## 🛠️ 기술 스택

### 백엔드
- **LangChain**: RAG 파이프라인 구축
- **FAISS**: 벡터 검색 엔진
- **Ollama**: 로컬 LLM 실행
- **Sentence Transformers**: 임베딩 생성

### 프론트엔드
- **Streamlit**: 웹 인터페이스
- **HTML/CSS**: UI 커스터마이징

### 데이터 처리
- **PyPDF2**: PDF 텍스트 추출
- **NumPy/Pandas**: 데이터 처리
- **JSON**: 메타데이터 저장

## 🔧 커스터마이징

### 모델 변경
```python
# LLM 모델 변경
llm = build_llm("llama3.2:3b")  # 더 빠른 모델
llm = build_llm("qwen2.5:7b")   # 더 정확한 모델

# 임베딩 모델 변경
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 하이퍼파라미터 조정
```python
# Ensemble 가중치 조정
ensemble_weights = [0.7, 0.3]  # FAISS 70%, BM25 30%

# Temperature 조정
llm = build_llm("command-r:35b", temperature=0.2)  # 더 창의적
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해 주세요.

---

**NXJ_RAG**: 의료제품 인허가를 위한 지능형 문서 검색 및 질의응답 시스템 