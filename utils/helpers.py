"""Utility helper functions."""

import re
import time
import logging
from typing import Any, Iterator, Optional, List, TypeVar
from datetime import datetime, date, timedelta
from contextlib import contextmanager
from functools import wraps

import numpy as np
import pandas as pd

from utils.constants import EXCHANGE_HOLIDAYS
from utils.exceptions import ConfigurationError

T = TypeVar('T')
logger = logging.getLogger(__name__)


# ============================================================================
# 性能計時器
# ============================================================================
class Timer:
    """
    高性能計時器，支持作為上下文管理器或裝飾器
    
    用法1 - 上下文管理器:
        with Timer("數據拉取"):
            fetch_data()
    
    用法2 - 裝飾器:
        @Timer("模型訓練")
        def train_model():
            pass
    """
    
    def __init__(self, name: str = "", log_level: int = logging.DEBUG):
        """
        參數:
            name: 任務名稱，用於日誌輸出
            log_level: 日誌級別
        """
        self.name = name
        self.log_level = log_level
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
    
    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.elapsed < 1.0:
            msg = f"{self.name} 完成，耗時 {self.elapsed*1000:.1f}ms"
        elif self.elapsed < 60:
            msg = f"{self.name} 完成，耗時 {self.elapsed:.2f}s"
        else:
            msg = f"{self.name} 完成，耗時 {self.elapsed/60:.1f}min"
        logger.log(self.log_level, msg)
    
    @classmethod
    def decorator(cls, name: str = ""):
        """作為裝飾器使用"""
        def wrapper(func):
            timer_name = name or func.__name__
            @wraps(func)
            def inner(*args, **kwargs):
                with cls(timer_name):
                    return func(*args, **kwargs)
            return inner
        return wrapper


# ============================================================================
# 數學工具函數
# ============================================================================
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零錯誤
    
    時間複雜度: O(1)
    空間複雜度: O(1)
    
    參數:
        numerator: 分子
        denominator: 分母
        default: 分母為零時的默認返回值
    
    返回:
        除法結果或默認值
    """
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, float('nan'))
        return result.fillna(default)
    if isinstance(denominator, (int, float)) and denominator == 0:
        return default
    return numerator / denominator


def rolling_window(arr: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """
    生成滾動窗口視圖（零拷貝），用於高效計算移動統計量
    
    時間複雜度: O(1) - 僅返回視圖，不複製數據
    空間複雜度: O(1) - 返回的是原數組的視圖
    
    參數:
        arr: 輸入的一維numpy數組
        window: 窗口大小
        stride: 步長
    
    返回:
        形狀為 (n_windows, window) 的二維數組視圖
    """
    n = arr.shape[0]
    if n < window:
        raise ValueError(f"數組長度 {n} 小於窗口大小 {window}")
    
    shape = ((n - window) // stride + 1, window)
    strides = (arr.strides[0] * stride, arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def clip_values(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Winsorize截尾處理：將超出上下界的值替換為邊界值
    
    時間複雜度: O(n)
    空間複雜度: O(n)
    
    參數:
        values: 輸入數組
        lower: 下界分位數（如0.01表示截掉最低1%）
        upper: 上界分位數（如0.99表示截掉最高1%）
    
    返回:
        截尾處理後的數組
    """
    lo = np.quantile(values, lower)
    hi = np.quantile(values, upper)
    return np.clip(values, lo, hi)


# ============================================================================
# 周末與假期判斷
# ============================================================================
# 緩存已計算的假期集合，避免重複構建
_holiday_set_cache: Optional[set] = None
_holiday_cache_year: Optional[int] = None


def _get_holidays_for_year(year: int) -> set:
    """獲取指定年份的假期集合（帶緩存）"""
    global _holiday_set_cache, _holiday_cache_year
    if _holiday_cache_year == year and _holiday_set_cache is not None:
        return _holiday_set_cache
    
    _holiday_set_cache = set()
    for y, holidays in EXCHANGE_HOLIDAYS.items():
        if y == year:
            _holiday_set_cache.update(holidays)
    if not _holiday_set_cache and year > max(EXCHANGE_HOLIDAYS.keys()):
        import warnings
        warnings.warn(f"假期數據不包含 {year} 年，僅使用周末判斷交易日")
    _holiday_cache_year = year
    return _holiday_set_cache


def is_trading_day(d: date) -> bool:
    """
    判斷指定日期是否為美股交易日
    
    規則：
    1. 周六、周日不是交易日
    2. 交易所假期不是交易日
    
    時間複雜度: O(1) - 假期集合查找
    空間複雜度: O(1) - 假期緩存
    
    參數:
        d: 待判斷的日期
    
    返回:
        是否為交易日
    """
    # 周六(5)或周日(6)不交易
    if d.weekday() >= 5:
        return False
    
    # 檢查交易所假期
    holidays = _get_holidays_for_year(d.year)
    if d in holidays:
        return False
    
    return True


def next_trading_day(d: date, offset: int = 1) -> date:
    """
    獲取下一個（或第N個後的）交易日
    
    時間複雜度: O(offset + 跳過天數)，通常offset很小，接近O(1)
    
    參數:
        d: 起始日期
        offset: 偏移量（正數為向後，負數為向前）
    
    返回:
        目標交易日
    """
    step = 1 if offset > 0 else -1
    count = 0
    current = d
    while count < abs(offset):
        current += timedelta(days=step)
        if is_trading_day(current):
            count += 1
    return current


def previous_trading_day(d: date) -> date:
    """獲取前一個交易日"""
    return next_trading_day(d, offset=-1)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    判斷當前（或指定時間）美股是否在正常交易時段內
    
    注意：此函數僅判斷正常盤 (9:30-16:00 ET)
    如需判斷盤前/盤後，使用 is_extended_hours()
    
    參數:
        dt: 待判斷的datetime（UTC時區），None則使用當前時間
    
    返回:
        是否在正常交易時段
    """
    from utils.constants import MarketHours
    
    if dt is None:
        dt = datetime.now(datetime.timezone.utc)
    
    # 使用 zoneinfo 正確轉換美東時間（自動處理 EST/EDT）
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    et_tz = ZoneInfo('America/New_York')
    et_dt = dt.astimezone(et_tz)
    et_time = et_dt.time()
    local_date = et_dt.date()
    
    if not is_trading_day(local_date):
        return False
    
    return MarketHours.REGULAR_START <= et_time <= MarketHours.REGULAR_END


# ============================================================================
# 格式化工具
# ============================================================================
def format_currency(amount: float, precision: int = 2) -> str:
    """
    格式化金額為美元格式
    
    參數:
        amount: 金額
        precision: 小數位數
    
    返回:
        格式化字符串，如 "$1,234.56"
    """
    if abs(amount) >= 1e6:
        return f"${amount:,.{precision}f}"
    return f"${amount:,.{precision}f}"


def format_pct(value: float, precision: int = 2) -> str:
    """
    格式化百分比
    
    參數:
        value: 比例值（如0.0523表示5.23%）
        precision: 小數位數
    
    返回:
        格式化字符串，如 "+5.23%"
    """
    pct = value * 100
    sign = '+' if pct > 0 else ''
    return f"{sign}{pct:.{precision}f}%"


TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')


def validate_ticker(ticker: str) -> bool:
    """
    驗證股票代碼格式
    
    美股ticker規則：
    - 1-5個大寫字母
    - 不含數字和特殊字符（部分ETF除外）
    
    時間複雜度: O(n), n為ticker長度（通常≤5）
    
    參數:
        ticker: 股票代碼
    
    返回:
        是否為有效格式
    """
    return bool(TICKER_PATTERN.match(ticker))


# ============================================================================
# 數據框工具函數
# ============================================================================
def align_dataframes(*dfs: pd.DataFrame, on: str = 'date') -> List[pd.DataFrame]:
    """
    按指定列對齊多個DataFrame，確保所有DataFrame具有相同的索引
    
    時間複雜度: O(n*k*log(n))，n為行數，k為DataFrame數量
    空間複雜度: O(n*k)
    
    參數:
        dfs: 需要對齊的DataFrame列表
        on: 對齊依據的列名
    
    返回:
        對齊後的DataFrame列表
    """
    if not dfs:
        return []
    
    # 找到所有DataFrame共有的鍵值
    common_keys = dfs[0][on]
    for df in dfs[1:]:
        common_keys = pd.merge(
            common_keys.to_frame(), df[[on]], on=on, how='inner'
        )[on]
    
    # 按共有鍵過濾
    aligned = [df[df[on].isin(common_keys)].sort_values(on).reset_index(drop=True) 
               for df in dfs]
    return aligned
