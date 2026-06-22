"""
美股量化交易系统 - 日志配置模块

提供统一的日志配置，支持：
- 控制台输出（带颜色区分日志级别）
- 文件输出（按日期轮转）
- 结构化日志（JSON格式可选，便于接入ELK等日志平台）
- 分级管理（交易日志、系统日志、审计日志分离）
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config.settings import SystemConfig, LOGS_DIR


# ============================================================================
# 自定义日志格式化器（带颜色）
# ============================================================================
class ColoredFormatter(logging.Formatter):
    """
    为控制台日志输出添加ANSI颜色标记
    
    颜色方案：
    - DEBUG:    灰色
    - INFO:     绿色
    - WARNING:  黄色
    - ERROR:    红色
    - CRITICAL: 红底白字
    """
    
    # ANSI颜色码
    COLORS = {
        'DEBUG':    '\033[90m',    # 灰色
        'INFO':     '\033[92m',    # 绿色
        'WARNING':  '\033[93m',    # 黄色
        'ERROR':    '\033[91m',    # 红色
        'CRITICAL': '\033[41m\033[97m',  # 红底白字
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # 克隆record以避免污染原始对象
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
# 日志管理器
# ============================================================================
class LogManager:
    """
    集中管理所有日志器
    
    设计原则：
    - 交易审计日志：记录每笔下单、成交、撤单，不可删除，用于合规审查
    - 系统运行日志：记录系统状态、数据拉取、模型运行等
    - 风控日志：记录所有风控触发事件
    """
    
    _instance: Optional['LogManager'] = None
    _initialized: bool = False
    
    def __new__(cls, config: Optional[SystemConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[SystemConfig] = None):
        # 只初始化一次（单例模式）
        if LogManager._initialized:
            return
        LogManager._initialized = True
        
        self.config = config or SystemConfig()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 初始化各类日志器
        self._setup_root_logger()
        self._setup_trade_logger()     # 交易审计日志
        self._setup_system_logger()    # 系统运行日志
        self._setup_risk_logger()      # 风控日志
    
    def _setup_root_logger(self) -> None:
        """配置根日志器（捕获未分类的日志）"""
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        
        # 清除已有的handler（防止重复）
        root.handlers.clear()
        
        # 控制台handler（INFO级别以上）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_handler.setFormatter(ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root.addHandler(console_handler)
        
        # 禁止第三方库的DEBUG日志噪音
        for noisy_lib in ['urllib3', 'matplotlib', 'PIL', 'asyncio', 'ib_insync']:
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)
    
    def _setup_file_handler(self, log_name: str) -> logging.FileHandler:
        """
        创建带轮转的文件handler
        
        参数:
            log_name: 日志文件名前缀（不含扩展名）
        
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
        交易审计日志器
        
        记录内容：每笔信号、下单、改单、撤单、成交、拒绝等
        安全要求：不可删除、不可修改，用于合规审查
        """
        self.trade_logger = logging.getLogger('trade_audit')
        self.trade_logger.setLevel(logging.DEBUG)
        self.trade_logger.propagate = False  # 不向root传播
        self.trade_logger.addHandler(
            self._setup_file_handler('trade_audit')
        )
    
    def _setup_system_logger(self) -> None:
        """
        系统运行日志器
        
        记录内容：数据拉取进度、模型训练状态、API调用耗时、错误堆栈等
        """
        self.system_logger = logging.getLogger('system')
        self.system_logger.setLevel(logging.DEBUG)
        self.system_logger.propagate = False
        self.system_logger.addHandler(
            self._setup_file_handler('system')
        )
        
        # 性能子日志器：记录各模块耗时
        self.perf_logger = logging.getLogger('system.perf')
        if self.config.enable_perf_logging:
            self.perf_logger.setLevel(logging.DEBUG)
    
    def _setup_risk_logger(self) -> None:
        """
        风控日志器
        
        记录内容：所有风控规则触发事件（事前拒绝、事中减仓、事后告警）
        """
        self.risk_logger = logging.getLogger('risk')
        self.risk_logger.setLevel(logging.DEBUG)
        self.risk_logger.propagate = False
        self.risk_logger.addHandler(
            self._setup_file_handler('risk')
        )
    
    @classmethod
    def get_trade_logger(cls) -> logging.Logger:
        """获取交易审计日志器"""
        instance = cls._instance or cls()
        return instance.trade_logger
    
    @classmethod
    def get_system_logger(cls) -> logging.Logger:
        """获取系统运行日志器"""
        instance = cls._instance or cls()
        return instance.system_logger
    
    @classmethod
    def get_risk_logger(cls) -> logging.Logger:
        """获取风控日志器"""
        instance = cls._instance or cls()
        return instance.risk_logger


def setup_logging(config: Optional[SystemConfig] = None) -> LogManager:
    """
    初始化全局日志系统（程序入口调用一次即可）
    
    参数:
        config: 系统配置对象，为None时使用默认配置
    
    返回:
        LogManager实例
    """
    return LogManager(config)
