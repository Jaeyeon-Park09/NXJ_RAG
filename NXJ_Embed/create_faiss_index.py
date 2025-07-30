"""
FAISS 인덱스 생성 스크립트

기존 metadata.json 파일을 읽어서 FAISS 인덱스를 생성하고
index.faiss와 index.pkl 파일을 만듭니다.
"""

import json
import os
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_faiss_index():
    """
    metadata.json 파일을 읽어서 FAISS 인덱스를 생성합니다.
    """
    try:
        # 경로 설정
        emb_dir = "emb"
        metadata_path = os.path.join(emb_dir, "metadata.json")
        
        # 메타데이터 파일 존재 확인
        if not os.path.exists(metadata_path):
            print(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
            return
        
        # 임베딩 모델 로드
        print("임베딩 모델 로드 중...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 메타데이터 파일 읽기
        print("메타데이터 파일 읽는 중...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # JSON 배열 형식인지 확인
            if content.startswith('[') and content.endswith(']'):
                data_list = json.loads(content)
            else:
                # JSONL 형식으로 처리
                data_list = []
                f.seek(0)
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        data_list.append(data)
                    except json.JSONDecodeError:
                        continue
        
        # Document 객체 리스트 생성
        print(f"문서 변환 중... (총 {len(data_list)}개)")
        docs = []
        for data in tqdm(data_list, desc="문서 변환"):
            if isinstance(data, dict):
                # 텍스트 추출
                text = data.get('page_content') or data.get('text', '')
                if text:
                    # 메타데이터에서 page_content/text 제외
                    metadata = {k: v for k, v in data.items() 
                              if k not in ['page_content', 'text']}
                    doc = Document(page_content=text, metadata=metadata)
                    docs.append(doc)
        
        print(f"변환 완료: {len(docs)}개 문서")
        
        if len(docs) == 0:
            print("변환된 문서가 없습니다.")
            return
        
        # FAISS 인덱스 생성
        print("FAISS 인덱스 생성 중...")
        db = FAISS.from_documents(docs, embedding_model)
        
        # 저장
        print(f"FAISS 인덱스를 {emb_dir}에 저장 중...")
        db.save_local(emb_dir)
        
        print("FAISS 인덱스 생성 완료!")
        print(f"- index.faiss: {os.path.join(emb_dir, 'index.faiss')}")
        print(f"- index.pkl: {os.path.join(emb_dir, 'index.pkl')}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    create_faiss_index() 