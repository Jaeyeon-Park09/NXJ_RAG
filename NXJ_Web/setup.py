"""
NXJ_RAG 프로젝트 설치 스크립트

이 스크립트는 NXJ_RAG 워크스페이스의 모든 모듈을 개발 모드로 설치합니다.
"""

from setuptools import setup, find_packages
import os

# 프로젝트 루트 디렉토리 (NXJ_Web의 상위 디렉토리)
project_root = os.path.dirname(os.path.abspath(__file__))

setup(
    name="nxj-rag",
    version="0.1.0",
    description="NXJ RAG (Retrieval-Augmented Generation) 시스템",
    author="NXJ Team",
    packages=find_packages(include=[
        "NXJ_LLM*",
        "NXJ_Retriever*", 
        "NXJ_Embed*",
        "NXJ_Parser_Text*"
    ]),
    install_requires=[
        "streamlit>=1.28.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.5",
        "ollama>=0.1.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "torch>=1.13.0",
        "rank-bm25>=0.2.2",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "json5>=0.9.0",
        "PyMuPDF==1.23.9",
        "python-dateutil==2.8.2",
        "regex==2023.10.3",
        "llama-index==0.9.48",
        "llama-index-core==0.10.57"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json", "*.yaml", "*.yml"]
    }
) 