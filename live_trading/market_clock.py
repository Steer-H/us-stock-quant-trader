"""
在线模拟交易系统 - 市场时钟模块

实时追踪美股交易时段，提供：
- 当前市场状态检测（盘前/正常盘/盘后/休市）
- 距离开市/闭市的精确倒计时
- 支持夏令时(EDT/EST)自动切换
- 假期判断

美股交易时段（东部时间）：
- 盘前 Pre-Market:  04:00 - 09:30 ET
- 正常盘 Regular:    09:30 - 16:00 ET
- 盘后 After-Hours:  16:00 - 20:00 ET
- 闭市 Closed:       20:00 - 04:00 ET (次日)
"""

import logging
from datetime import datetime, date, timedelta, timezone, time
from zoneinfo import ZoneInfo
from typing import Tuple, Optional, Dict, NamedTuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# 市场状态枚举
# ============================================================================
class MarketStatus(Enum):
    """市场状态"""
    CLOSED = 'CLOSED'              # 休市
    PRE_MARKET = 'PRE_MARKET'      # 盘前交易 (04:00-09:30 ET)
    REGULAR_HOURS = 'REGULAR'      # 正常交易时段 (09:30-16:00 ET)
    AFTER_HOURS = 'AFTER_HOURS'    # 盘后交易 (16:00-20:00 ET)
    WEEKEND = 'WEEKEND'            # 周末
    HOLIDAY = 'HOLIDAY'            # 假期


# 市场状态对应的中文描述
STATUS_DESC = {
    MarketStatus.PRE_MARKET:    '盘前交易',
    MarketStatus.REGULAR_HOURS: '正常交易',
    MarketStatus.AFTER_HOURS:   '盘后交易',
    MarketStatus.CLOSED:        '已闭市',
    MarketStatus.WEEKEND:       '周末休市',
    MarketStatus.HOLIDAY:       '假期休市',
}

# 市场状态对应的颜色（终端展示用）
STATUS_COLOR = {
    MarketStatus.PRE_MARKET:    '\033[93m',    # 黄色
    MarketStatus.REGULAR_HOURS: '\033[92m',    # 绿色
    MarketStatus.AFTER_HOURS:   '\033[94m',    # 蓝色
}


# 市场关键时间点（东部时间）
MARKET_TIMES = {
    'pre_market_start':   time(4, 0),     # 04:00
    'regular_start':      time(9, 30),    # 09:30
    'regular_end':        time(16, 0),    # 16:00
    'early_close_end':    time(13, 0),    # 13:00 (Black Friday等早收盘)
    'after_hours_end':    time(20, 0),    # 20:00
}

# 美股假期（2025-2026完整列表）
# 假期数据统一从 constants 导入，避免重复维护 (GUARDRAILS #16)

# ============================================================================
# 市场时钟类
# ============================================================================
class MarketClock:
    """
    美股市场时钟
    
    自动判断当前市场状态，并提供倒计时功能。
    
    时区处理：
    - 美股使用东部时间 (ET)
    - 夏令时 EDT (UTC-4): 3月第二个周日 ~ 11月第一个周日
    - 冬令时 EST (UTC-5): 其余时间
    
    使用示例:
        clock = MarketClock()
        status, desc = clock.get_status()
        if status == MarketStatus.CLOSED:
            h, m, s = clock.countdown_to_next_open()
            print(f"距离开市还有 {h}小时{m}分钟{s}秒")
    """
    
    def __init__(self):
        pass
    
    def _is_dst(self, dt: datetime) -> bool:
        """
        判断是否处于夏令时 (EDT)
        
        规则：3月第二个周日凌晨2:00 到 11月第一个周日凌晨2:00
        
        参数:
            dt: datetime对象（UTC或本地时间均可）
        
        返回:
            是否DST
        """
        # 优先使用 zoneinfo 正确处理时区
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        if dt.tzinfo is not None:
            et_dt = dt.astimezone(et_tz)
        else:
            # naive datetime 假设已是 ET
            et_dt = dt.replace(tzinfo=et_tz)
        # DST期间 UTC offset 为 -4h (EDT), 否则 -5h (EST)
        return et_dt.utcoffset().total_seconds() == -4 * 3600
    
    def get_utc_offset(self, dt: Optional[datetime] = None) -> timedelta:
        """
        获取当前的UTC偏移
        
        参数:
            dt: 参考时间（None则使用当前UTC时间）
        
        返回:
            UTC偏移（EDT=-4h, EST=-5h）
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        # 使用 zoneinfo 正确处理夏令时
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        if dt.tzinfo is not None:
            et_dt = dt.astimezone(et_tz)
        else:
            et_dt = dt.replace(tzinfo=et_tz)
        return et_dt.utcoffset()
    
    def utc_to_et(self, utc_dt: datetime) -> datetime:
        """
        UTC时间转东部时间
        
        参数:
            utc_dt: UTC时间
        
        返回:
            东部时间（naive datetime）
        """
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        if utc_dt.tzinfo is not None:
            et_dt = utc_dt.astimezone(et_tz)
        else:
            et_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(et_tz)
        return et_dt.replace(tzinfo=None)
    
    def et_to_utc(self, et_dt: datetime) -> datetime:
        """
        东部时间转UTC时间
        
        参数:
            et_dt: 东部时间（naive datetime）
        
        返回:
            UTC时间
        """
        offset = self.get_utc_offset(et_dt)
        utc_dt = et_dt - offset
        return utc_dt.replace(tzinfo=timezone.utc)
    
    def now_et(self) -> datetime:
        """获取当前东部时间"""
        return self.utc_to_et(datetime.now(timezone.utc))
    
    def is_holiday(self, d: date) -> bool:
        """判断指定日期是否为美股假期 (委派给 helpers._get_holidays_for_year)"""
        from utils.helpers import _get_holidays_for_year
        return d in _get_holidays_for_year(d.year)
    
    def is_trading_day(self, d: Optional[date] = None) -> bool:
        """
        判断指定日期是否为交易日
        
        规则：周一至周五，且非假期。
        
        参数:
            d: 日期（None为当前东部时间日期）
        """
        if d is None:
            d = self.now_et().date()
        
        # 周末不交易
        if d.weekday() >= 5:  # 5=周六, 6=周日
            return False
        
        # 假期不交易
        if self.is_holiday(d):
            return False
        
        return True
    
    def get_status(self, et_time: Optional[datetime] = None) -> Tuple[MarketStatus, str]:
        """
        获取当前市场状态
        
        判断逻辑：
        1. 先判断是否为交易日
        2. 再判断时段（盘前/正常/盘后/闭市）
        
        参数:
            et_time: 东部时间（None则使用当前时间）
        
        返回:
            (MarketStatus, 中文描述)
        """
        if et_time is None:
            et_time = self.now_et()
        
        today = et_time.date()
        current_time = et_time.time()
        
        # 周末
        if today.weekday() >= 5:
            return MarketStatus.WEEKEND, STATUS_DESC[MarketStatus.WEEKEND]
        
        # 假期
        if self.is_holiday(today):
            return MarketStatus.HOLIDAY, STATUS_DESC[MarketStatus.HOLIDAY]
        
        # 交易日时段判断
        if current_time < MARKET_TIMES['pre_market_start']:
            # 00:00 - 04:00: 闭市
            return MarketStatus.CLOSED, STATUS_DESC[MarketStatus.CLOSED]
        elif current_time < MARKET_TIMES['regular_start']:
            # 04:00 - 09:30: 盘前
            return MarketStatus.PRE_MARKET, STATUS_DESC[MarketStatus.PRE_MARKET]
        elif current_time <= MARKET_TIMES['regular_end']:
            # 09:30 - 16:00: 正常交易 (提前收盘日到13:00)
            if self.is_early_close(today) and current_time > MARKET_TIMES['early_close_end']:
                return MarketStatus.AFTER_HOURS, '提前收盘后'
            return MarketStatus.REGULAR_HOURS, STATUS_DESC[MarketStatus.REGULAR_HOURS]
        elif current_time <= MARKET_TIMES['after_hours_end']:
            # 16:00 - 20:00: 盘后
            return MarketStatus.AFTER_HOURS, STATUS_DESC[MarketStatus.AFTER_HOURS]
        else:
            # 20:00 - 23:59: 闭市
            return MarketStatus.CLOSED, STATUS_DESC[MarketStatus.CLOSED]
    
    def is_early_close(self, d: date = None) -> bool:
        """
        判断是否为提前收盘日（如感恩节后周五13:00收盘）
        
        参数:
            d: 日期对象（None则使用当前日期）
        
        返回:
            是否提前收盘
        """
        if d is None:
            d = self.now_et().date()
        
        # 感恩节后周五 (Black Friday) — 13:00 收盘
        # 感恩节 = 11月第4个周四
        year = d.year
        nov1 = date(year, 11, 1)
        days_to_thu = (3 - nov1.weekday()) % 7
        thanksgiving = nov1 + timedelta(days=days_to_thu + 21)  # 第4个周四
        black_friday = thanksgiving + timedelta(days=1)
        if d == black_friday:
            return True
        
        return False
    
    def is_market_open(self) -> bool:
        """
        当前是否处于正常交易时段
        
        仅 09:30-16:00 ET 返回True。
        """
        status, _ = self.get_status()
        return status == MarketStatus.REGULAR_HOURS
    
    def is_trading_active(self) -> bool:
        """
        当前是否可交易（含盘前盘后）
        """
        status, _ = self.get_status()
        return status in (MarketStatus.PRE_MARKET, MarketStatus.REGULAR_HOURS, MarketStatus.AFTER_HOURS)
    
    def countdown_to_next_open(self) -> Tuple[int, int, int]:
        """
        计算距离开市的倒计时
        
        倒计时目标：下一个交易日的正常开市时间（09:30 ET）。
        如果是周末/假期，计算到下一个交易日的09:30。
        如果已经过了当天的09:30但在休市时段，则返回(0,0,0)。
        
        返回:
            (小时, 分钟, 秒)
        """
        now = self.now_et()
        
        # 找到下一个交易日的09:30
        target_date = now.date()
        
        # 如果当前时间已经过了今天的09:30，目标应该是明天
        if now.time() >= MARKET_TIMES['regular_start']:
            target_date += timedelta(days=1)
        
        # 往前跳到下一个交易日
        while True:
            if target_date.weekday() < 5 and not self.is_holiday(target_date):
                break
            target_date += timedelta(days=1)
        
        # 计算目标时间
        target_dt = datetime.combine(target_date, MARKET_TIMES['regular_start'])
        
        # 计算差值
        diff = target_dt - now
        total_seconds = int(diff.total_seconds())
        
        if total_seconds <= 0:
            return 0, 0, 0
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return hours, minutes, seconds
    
    def countdown_to_close(self) -> Optional[Tuple[int, int, int]]:
        """
        计算距离闭市的倒计时
        
        仅在正常交易时段有效。
        
        返回:
            (小时, 分钟, 秒) 或 None（非交易时段）
        """
        status, _ = self.get_status()
        
        if status != MarketStatus.REGULAR_HOURS:
            return None
        
        now = self.now_et()
        close_dt = datetime.combine(now.date(), MARKET_TIMES['regular_end'])
        
        diff = close_dt - now
        total_seconds = int(diff.total_seconds())
        
        if total_seconds <= 0:
            return None
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return hours, minutes, seconds
    
    def get_trading_session_info(self) -> Dict[str, object]:
        """
        获取完整的交易时段信息（用于面板展示）
        
        返回:
            {
                'status': MarketStatus,
                'description': str,
                'current_et': str,
                'is_open': bool,
                'countdown_to_open': (h, m, s) or None,
                'countdown_to_close': (h, m, s) or None,
                'next_trading_day': str,
            }
        """
        now = self.now_et()
        status, desc = self.get_status(now)
        
        info = {
            'status': status,
            'description': desc,
            'current_et': now.strftime('%Y-%m-%d %H:%M:%S ET'),
            'is_open': status == MarketStatus.REGULAR_HOURS,
            'is_active': self.is_trading_active(),
            'next_trading_day': self._next_trading_day_str(),
        }
        
        # 倒计时（始终提供）
        if status == MarketStatus.REGULAR_HOURS:
            info['countdown_to_open'] = None
            info['countdown_to_close'] = self.countdown_to_close()
        else:
            info['countdown_to_open'] = self.countdown_to_next_open()
            info['countdown_to_close'] = None
        
        return info
    
    def _next_trading_day_str(self) -> str:
        """获取下一个交易日的字符串表示"""
        now = self.now_et()
        d = now.date() + timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d.strftime('%Y-%m-%d')
    
    def format_countdown(self, h: int, m: int, s: int) -> str:
        """
        格式化倒计时显示
        
        参数:
            h, m, s: 时、分、秒
        
        返回:
            格式化的字符串，如 "2h 15m 30s"
        """
        parts = []
        if h > 0:
            parts.append(f"{h}h")
        if m > 0 or h > 0:
            parts.append(f"{m:02d}m")
        parts.append(f"{s:02d}s")
        return ' '.join(parts)


# ============================================================================
# 便捷函数（模块级使用）
# ============================================================================

# 全局时钟实例
_clock = MarketClock()


def get_market_status() -> Tuple[MarketStatus, str]:
    """获取当前市场状态（便捷函数）"""
    return _clock.get_status()


def countdown_to_market() -> str:
    """获取距离开市的倒计时字符串（便捷函数）"""
    status, _ = _clock.get_status()
    
    if status == MarketStatus.REGULAR_HOURS:
        cd = _clock.countdown_to_close()
        if cd:
            return f"距闭市还有: {_clock.format_countdown(*cd)}"
        return "交易中"
    
    if status in (MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS):
        cd = _clock.countdown_to_next_open()
        return f"距正常开盘还有: {_clock.format_countdown(*cd)}"
    
    cd = _clock.countdown_to_next_open()
    return f"距离开市还有: {_clock.format_countdown(*cd)}"
