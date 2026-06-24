"""Logging configuration."""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config.settings import SystemConfig, LOGS_DIR


# ============================================================================
# 自定義日誌格式化器（帶顏色）
# ============================================================================
class ColoredFormatter(logging.Formatter):
    """
    為控制臺日誌輸出添加ANSI顏色標記
    
    顏色方案：
    - DEBUG:    灰色
    - INFO:     綠色
    - WARNING:  黃色
    - ERROR:    紅色
    - CRITICAL: 紅底白字
    """
    
    # ANSI顏色碼
    COLORS = {
        'DEBUG':    '\033[90m',    # 灰色
        'INFO':     '\033[92m',    # 綠色
        'WARNING':  '\033[93m',    # 黃色
        'ERROR':    '\033[91m',    # 紅色
        'CRITICAL': '\033[41m\033[97m',  # 紅底白字
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # 克隆record以避免汙染原始對象
        record_copy = logging.LogRecord(
            record.name, record.levelno, record.pathname,
            record.lineno, record.msg, record.args,
            record.exc_info, record.funcName
        )
        record_copy.__dict__.update(record.__dict__)
        
        color = self.COLORS.get(record.levelname, '')
        if color:
            record_copy.levelname = f"{color}{record.levelname}{self.RESET}"
            record_copy.msg = f"{color}{record.msg}{self.RESET}"
            record_copy.name = f"{color}{record.name}{self.RESET}"
        
        return super().format(record_copy)


# ============================================================================
# 日誌管理器
# ============================================================================
class LogManager:
    """
    集中管理所有日誌器
    
    設計原則：
    - 交易審計日誌：記錄每筆下單、成交、撤單，不可刪除，用於合規審查
    - 系統運行日誌：記錄系統狀態、數據拉取、模型運行等
    - 風控日誌：記錄所有風控觸發事件
    """
    
    _instance: Optional['LogManager'] = None
    _initialized: bool = False
    
    def __new__(cls, config: Optional[SystemConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[SystemConfig] = None):
        # 只初始化一次（單例模式）
        if LogManager._initialized:
            return
        LogManager._initialized = True
        
        self.config = config or SystemConfig()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 初始化各類日誌器
        self._setup_root_logger()
        self._setup_trade_logger()     # 交易審計日誌
        self._setup_system_logger()    # 系統運行日誌
        self._setup_risk_logger()      # 風控日誌
    
    def _setup_root_logger(self) -> None:
        """配置根日誌器（捕獲未分類的日誌）"""
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        
        # 清除已有的handler（防止重複）
        root.handlers.clear()
        
        # 控制臺handler（INFO級別以上）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_handler.setFormatter(ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root.addHandler(console_handler)
        
        # 禁止第三方庫的DEBUG日誌噪音
        for noisy_lib in ['urllib3', 'matplotlib', 'PIL', 'asyncio', 'ib_insync']:
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)
    
    def _setup_file_handler(self, log_name: str) -> logging.FileHandler:
        """
        創建帶輪轉的文件handler
        
        參數:
            log_name: 日誌文件名前綴（不含擴展名）
        
        返回:
            配置好的RotatingFileHandler
        """
        log_path = LOGS_DIR / f"{log_name}.log"
        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=self.config.log_max_size_mb * 1024 * 1024,
            backupCount=self.config.log_backup_count,
            encoding='utf-8'
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        return handler
    
    def _setup_trade_logger(self) -> None:
        """
        交易審計日誌器
        
        記錄內容：每筆信號、下單、改單、撤單、成交、拒絕等
        安全要求：不可刪除、不可修改，用於合規審查
        """
        self.trade_logger = logging.getLogger('trade_audit')
        self.trade_logger.setLevel(logging.DEBUG)
        self.trade_logger.propagate = False  # 不向root傳播
        self.trade_logger.addHandler(
            self._setup_file_handler('trade_audit')
        )
    
    def _setup_system_logger(self) -> None:
        """
        系統運行日誌器
        
        記錄內容：數據拉取進度、模型訓練狀態、API調用耗時、錯誤堆棧等
        """
        self.system_logger = logging.getLogger('system')
        self.system_logger.setLevel(logging.DEBUG)
        self.system_logger.propagate = False
        self.system_logger.addHandler(
            self._setup_file_handler('system')
        )
        
        # 性能子日誌器：記錄各模塊耗時
        self.perf_logger = logging.getLogger('system.perf')
        if self.config.enable_perf_logging:
            self.perf_logger.setLevel(logging.DEBUG)
    
    def _setup_risk_logger(self) -> None:
        """
        風控日誌器
        
        記錄內容：所有風控規則觸發事件（事前拒絕、事中減倉、事後告警）
        """
        self.risk_logger = logging.getLogger('risk')
        self.risk_logger.setLevel(logging.DEBUG)
        self.risk_logger.propagate = False
        self.risk_logger.addHandler(
            self._setup_file_handler('risk')
        )
    
    @classmethod
    def get_trade_logger(cls) -> logging.Logger:
        """獲取交易審計日誌器"""
        instance = cls._instance or cls()
        return instance.trade_logger
    
    @classmethod
    def get_system_logger(cls) -> logging.Logger:
        """獲取系統運行日誌器"""
        instance = cls._instance or cls()
        return instance.system_logger
    
    @classmethod
    def get_risk_logger(cls) -> logging.Logger:
        """獲取風控日誌器"""
        instance = cls._instance or cls()
        return instance.risk_logger


def setup_logging(config: Optional[SystemConfig] = None) -> LogManager:
    """
    初始化全局日誌系統（程序入口調用一次即可）
    
    參數:
        config: 系統配置對象，為None時使用默認配置
    
    返回:
        LogManager實例
    """
    return LogManager(config)
