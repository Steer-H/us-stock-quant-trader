"""
美股量化交易系统 - 通用工具函数

提供各模块共享的基础工具，包括：
- 性能计时器（上下文管理器）
- 数学工具（安全除法、滚动窗口）
- 时间工具（交易日判断、假期判断）
- 格式化工具（货币、百分比）
- 数据验证工具
"""

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
# 性能计时器
# ============================================================================
class Timer:
    """
    高性能计时器，支持作为上下文管理器或装饰器
    
    用法1 - 上下文管理器:
        with Timer("数据拉取"):
            fetch_data()
    
    用法2 - 装饰器:
        @Timer("模型训练")
        def train_model():
            pass
    """
    
    def __init__(self, name: str = "", log_level: int = logging.DEBUG):
        """
        参数:
            name: 任务名称，用于日志输出
            log_level: 日志级别
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
            msg = f"{self.name} 完成，耗时 {self.elapsed*1000:.1f}ms"
        elif self.elapsed < 60:
            msg = f"{self.name} 完成，耗时 {self.elapsed:.2f}s"
        else:
            msg = f"{self.name} 完成，耗时 {self.elapsed/60:.1f}min"
        logger.log(self.log_level, msg)
    
    @classmethod
    def decorator(cls, name: str = ""):
        """作为装饰器使用"""
        def wrapper(func):
            timer_name = name or func.__name__
            @wraps(func)
            def inner(*args, **kwargs):
                with cls(timer_name):
                    return func(*args, **kwargs)
            return inner
        return wrapper


# ============================================================================
# 数学工具函数
# ============================================================================
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    时间复杂度: O(1)
    空间复杂度: O(1)
    
    参数:
        numerator: 分子
        denominator: 分母
        default: 分母为零时的默认返回值
    
    返回:
        除法结果或默认值
    """
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, float('nan'))
        return result.fillna(default)
    if isinstance(denominator, (int, float)) and denominator == 0:
        return default
    return numerator / denominator


def rolling_window(arr: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """
    生成滚动窗口视图（零拷贝），用于高效计算移动统计量
    
    时间复杂度: O(1) - 仅返回视图，不复制数据
    空间复杂度: O(1) - 返回的是原数组的视图
    
    参数:
        arr: 输入的一维numpy数组
        window: 窗口大小
        stride: 步长
    
    返回:
        形状为 (n_windows, window) 的二维数组视图
    """
    n = arr.shape[0]
    if n < window:
        raise ValueError(f"数组长度 {n} 小于窗口大小 {window}")
    
    shape = ((n - window) // stride + 1, window)
    strides = (arr.strides[0] * stride, arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def clip_values(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Winsorize截尾处理：将超出上下界的值替换为边界值
    
    时间复杂度: O(n)
    空间复杂度: O(n)
    
    参数:
        values: 输入数组
        lower: 下界分位数（如0.01表示截掉最低1%）
        upper: 上界分位数（如0.99表示截掉最高1%）
    
    返回:
        截尾处理后的数组
    """
    lo = np.quantile(values, lower)
    hi = np.quantile(values, upper)
    return np.clip(values, lo, hi)


# ============================================================================
# 周末与假期判断
# ============================================================================
# 缓存已计算的假期集合，避免重复构建
_holiday_set_cache: Optional[set] = None
_holiday_cache_year: Optional[int] = None


def _get_holidays_for_year(year: int) -> set:
    """获取指定年份的假期集合（带缓存）"""
    global _holiday_set_cache, _holiday_cache_year
    if _holiday_cache_year == year and _holiday_set_cache is not None:
        return _holiday_set_cache
    
    _holiday_set_cache = set()
    for y, holidays in EXCHANGE_HOLIDAYS.items():
        if y == year:
            _holiday_set_cache.update(holidays)
    if not _holiday_set_cache and year > max(EXCHANGE_HOLIDAYS.keys()):
        import warnings
        warnings.warn(f"假期数据不包含 {year} 年，仅使用周末判断交易日")
    _holiday_cache_year = year
    return _holiday_set_cache


def is_trading_day(d: date) -> bool:
    """
    判断指定日期是否为美股交易日
    
    规则：
    1. 周六、周日不是交易日
    2. 交易所假期不是交易日
    
    时间复杂度: O(1) - 假期集合查找
    空间复杂度: O(1) - 假期缓存
    
    参数:
        d: 待判断的日期
    
    返回:
        是否为交易日
    """
    # 周六(5)或周日(6)不交易
    if d.weekday() >= 5:
        return False
    
    # 检查交易所假期
    holidays = _get_holidays_for_year(d.year)
    if d in holidays:
        return False
    
    return True


def next_trading_day(d: date, offset: int = 1) -> date:
    """
    获取下一个（或第N个后的）交易日
    
    时间复杂度: O(offset + 跳过天数)，通常offset很小，接近O(1)
    
    参数:
        d: 起始日期
        offset: 偏移量（正数为向后，负数为向前）
    
    返回:
        目标交易日
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
    """获取前一个交易日"""
    return next_trading_day(d, offset=-1)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    判断当前（或指定时间）美股是否在正常交易时段内
    
    注意：此函数仅判断正常盘 (9:30-16:00 ET)
    如需判断盘前/盘后，使用 is_extended_hours()
    
    参数:
        dt: 待判断的datetime（UTC时区），None则使用当前时间
    
    返回:
        是否在正常交易时段
    """
    from utils.constants import MarketHours
    
    if dt is None:
        dt = datetime.now(datetime.timezone.utc)
    
    # 使用 zoneinfo 正确转换美东时间（自动处理 EST/EDT）
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
    格式化金额为美元格式
    
    参数:
        amount: 金额
        precision: 小数位数
    
    返回:
        格式化字符串，如 "$1,234.56"
    """
    if abs(amount) >= 1e6:
        return f"${amount:,.{precision}f}"
    return f"${amount:,.{precision}f}"


def format_pct(value: float, precision: int = 2) -> str:
    """
    格式化百分比
    
    参数:
        value: 比例值（如0.0523表示5.23%）
        precision: 小数位数
    
    返回:
        格式化字符串，如 "+5.23%"
    """
    pct = value * 100
    sign = '+' if pct > 0 else ''
    return f"{sign}{pct:.{precision}f}%"


TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')


def validate_ticker(ticker: str) -> bool:
    """
    验证股票代码格式
    
    美股ticker规则：
    - 1-5个大写字母
    - 不含数字和特殊字符（部分ETF除外）
    
    时间复杂度: O(n), n为ticker长度（通常≤5）
    
    参数:
        ticker: 股票代码
    
    返回:
        是否为有效格式
    """
    return bool(TICKER_PATTERN.match(ticker))


# ============================================================================
# 数据框工具函数
# ============================================================================
def align_dataframes(*dfs: pd.DataFrame, on: str = 'date') -> List[pd.DataFrame]:
    """
    按指定列对齐多个DataFrame，确保所有DataFrame具有相同的索引
    
    时间复杂度: O(n*k*log(n))，n为行数，k为DataFrame数量
    空间复杂度: O(n*k)
    
    参数:
        dfs: 需要对齐的DataFrame列表
        on: 对齐依据的列名
    
    返回:
        对齐后的DataFrame列表
    """
    if not dfs:
        return []
    
    # 找到所有DataFrame共有的键值
    common_keys = dfs[0][on]
    for df in dfs[1:]:
        common_keys = pd.merge(
            common_keys.to_frame(), df[[on]], on=on, how='inner'
        )[on]
    
    # 按共有键过滤
    aligned = [df[df[on].isin(common_keys)].sort_values(on).reset_index(drop=True) 
               for df in dfs]
    return aligned
