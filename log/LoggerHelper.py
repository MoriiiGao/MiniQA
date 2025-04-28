import logging
import os
from datetime import datetime
from typing import Optional


class LoggerHelper:
    def __init__(self,
                 name: str = "AppLogger",
                 log_dir: str = "logs",
                 log_file: Optional[str] = None,
                 level: int = logging.INFO):
        """
        初始化日志类
        :param name: 日志器名称（可用于多模块区分）
        :param log_dir: 日志保存目录
        :param log_file: 日志文件名（默认按时间生成）
        :param level: 日志等级，默认 logging.INFO
        """
        os.makedirs(log_dir, exist_ok=True)
        if log_file is None:
            log_file = datetime.now().strftime("%Y%m%d_%H%M%S.log")
        log_path = os.path.join(log_dir, log_file)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(console_handler)

            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(file_handler)

        self.logger.info(f"✅ 日志初始化完成 | 路径: {log_path}")

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

    def log_dict(self, title: str, data: dict):
        self.logger.info(f"📌 {title}")
        for k, v in data.items():
            self.logger.info(f"    {k}: {v}")
