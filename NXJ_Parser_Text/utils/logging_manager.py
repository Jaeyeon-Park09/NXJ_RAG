"""
로깅 및 진행상황 관리 모듈
"""

import logging
import time
from pathlib import Path


class LogManager:
    """로깅 및 진행상황 관리 클래스"""
    
    def __init__(self):
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.run_number = self._get_next_run_number()
        self.log_file = self.logs_dir / f"parsing_run_{self.run_number:03d}.log"
        self._setup_logging()
        
    def _get_next_run_number(self) -> int:
        """다음 실행 번호를 가져옵니다."""
        existing_logs = list(self.logs_dir.glob("parsing_run_*.log"))
        if not existing_logs:
            return 1
        
        numbers = []
        for log_file in existing_logs:
            try:
                num = int(log_file.stem.split('_')[-1])
                numbers.append(num)
            except ValueError:
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def _setup_logging(self):
        """로깅 설정을 초기화합니다."""
        # 기존 핸들러 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 파일 핸들러
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # 루트 로거 설정
        logging.root.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        
        # 시작 로그
        logging.info(f"=== PDF 파싱 실행 #{self.run_number} 시작 ===")
        logging.info(f"로그 파일: {self.log_file}")
    
    def log_progress(self, current: int, total: int, message: str):
        """진행상황을 로그로 기록합니다."""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_bar = self._create_progress_bar(percentage)
        log_msg = f"[{current}/{total}] {progress_bar} {percentage:.1f}% - {message}"
        
        print(f"\r{log_msg}", end='', flush=True)
        logging.debug(log_msg)
        
        if current == total:
            print()  # 완료 시 새 줄
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """진행률 바를 생성합니다."""
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]" 