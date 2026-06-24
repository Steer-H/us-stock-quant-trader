"""US stock market trading session clock with holiday calendar."""

import logging
from datetime import datetime, date, timedelta, timezone, time
from zoneinfo import ZoneInfo
from typing import Tuple, Optional, Dict, NamedTuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# 市場狀態枚舉
# ============================================================================
class MarketStatus(Enum):
    """市場狀態"""
    CLOSED = 'CLOSED'              # 休市
    PRE_MARKET = 'PRE_MARKET'      # 盤前交易 (04:00-09:30 ET)
    REGULAR_HOURS = 'REGULAR'      # 正常交易時段 (09:30-16:00 ET)
    AFTER_HOURS = 'AFTER_HOURS'    # 盤後交易 (16:00-20:00 ET)
    WEEKEND = 'WEEKEND'            # 周末
    HOLIDAY = 'HOLIDAY'            # 假期


# 市場狀態對應的中文描述
STATUS_DESC = {
    MarketStatus.PRE_MARKET:    '盤前交易',
    MarketStatus.REGULAR_HOURS: '正常交易',
    MarketStatus.AFTER_HOURS:   '盤後交易',
    MarketStatus.CLOSED:        '已閉市',
    MarketStatus.WEEKEND:       '周末休市',
    MarketStatus.HOLIDAY:       '假期休市',
}

# 市場狀態對應的顏色（終端展示用）
STATUS_COLOR = {
    MarketStatus.PRE_MARKET:    '\033[93m',    # 黃色
    MarketStatus.REGULAR_HOURS: '\033[92m',    # 綠色
    MarketStatus.AFTER_HOURS:   '\033[94m',    # 藍色
}


# 市場關鍵時間點（東部時間）
MARKET_TIMES = {
    'pre_market_start':   time(4, 0),     # 04:00
    'regular_start':      time(9, 30),    # 09:30
    'regular_end':        time(16, 0),    # 16:00
    'early_close_end':    time(13, 0),    # 13:00 (Black Friday等早收盤)
    'after_hours_end':    time(20, 0),    # 20:00
}

# 美股假期（2025-2026完整列表）
# 假期數據統一從 constants 導入，避免重複維護 (GUARDRAILS #16)

# ============================================================================
# 市場時鐘類
# ============================================================================
class MarketClock:
    """
    美股市場時鐘
    
    自動判斷當前市場狀態，並提供倒計時功能。
    
    時區處理：
    - 美股使用東部時間 (ET)
    - 夏令時 EDT (UTC-4): 3月第二個周日 ~ 11月第一個周日
    - 冬令時 EST (UTC-5): 其餘時間
    
    使用示例:
        clock = MarketClock()
        status, desc = clock.get_status()
        if status == MarketStatus.CLOSED:
            h, m, s = clock.countdown_to_next_open()
            print(f"距離開市還有 {h}小時{m}分鐘{s}秒")
    """
    
    def __init__(self):
        pass
    
    def _is_dst(self, dt: datetime) -> bool:
        """
        判斷是否處於夏令時 (EDT)
        
        規則：3月第二個周日凌晨2:00 到 11月第一個周日凌晨2:00
        
        參數:
            dt: datetime對象（UTC或本地時間均可）
        
        返回:
            是否DST
        """
        # 優先使用 zoneinfo 正確處理時區
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        if dt.tzinfo is not None:
            et_dt = dt.astimezone(et_tz)
        else:
            # naive datetime 假設已是 ET
            et_dt = dt.replace(tzinfo=et_tz)
        # DST期間 UTC offset 為 -4h (EDT), 否則 -5h (EST)
        return et_dt.utcoffset().total_seconds() == -4 * 3600
    
    def get_utc_offset(self, dt: Optional[datetime] = None) -> timedelta:
        """
        獲取當前的UTC偏移
        
        參數:
            dt: 參考時間（None則使用當前UTC時間）
        
        返回:
            UTC偏移（EDT=-4h, EST=-5h）
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        # 使用 zoneinfo 正確處理夏令時
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
        UTC時間轉東部時間
        
        參數:
            utc_dt: UTC時間
        
        返回:
            東部時間（naive datetime）
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
        東部時間轉UTC時間
        
        參數:
            et_dt: 東部時間（naive datetime）
        
        返回:
            UTC時間
        """
        offset = self.get_utc_offset(et_dt)
        utc_dt = et_dt - offset
        return utc_dt.replace(tzinfo=timezone.utc)
    
    def now_et(self) -> datetime:
        """獲取當前東部時間"""
        return self.utc_to_et(datetime.now(timezone.utc))
    
    def is_holiday(self, d: date) -> bool:
        """判斷指定日期是否為美股假期 (委派給 helpers._get_holidays_for_year)"""
        from utils.helpers import _get_holidays_for_year
        return d in _get_holidays_for_year(d.year)
    
    def is_trading_day(self, d: Optional[date] = None) -> bool:
        """
        判斷指定日期是否為交易日
        
        規則：周一至周五，且非假期。
        
        參數:
            d: 日期（None為當前東部時間日期）
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
        獲取當前市場狀態
        
        判斷邏輯：
        1. 先判斷是否為交易日
        2. 再判斷時段（盤前/正常/盤後/閉市）
        
        參數:
            et_time: 東部時間（None則使用當前時間）
        
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
        
        # 交易日時段判斷
        if current_time < MARKET_TIMES['pre_market_start']:
            # 00:00 - 04:00: 閉市
            return MarketStatus.CLOSED, STATUS_DESC[MarketStatus.CLOSED]
        elif current_time < MARKET_TIMES['regular_start']:
            # 04:00 - 09:30: 盤前
            return MarketStatus.PRE_MARKET, STATUS_DESC[MarketStatus.PRE_MARKET]
        elif current_time <= MARKET_TIMES['regular_end']:
            # 09:30 - 16:00: 正常交易 (提前收盤日到13:00)
            if self.is_early_close(today) and current_time > MARKET_TIMES['early_close_end']:
                return MarketStatus.AFTER_HOURS, '提前收盤後'
            return MarketStatus.REGULAR_HOURS, STATUS_DESC[MarketStatus.REGULAR_HOURS]
        elif current_time <= MARKET_TIMES['after_hours_end']:
            # 16:00 - 20:00: 盤後
            return MarketStatus.AFTER_HOURS, STATUS_DESC[MarketStatus.AFTER_HOURS]
        else:
            # 20:00 - 23:59: 閉市
            return MarketStatus.CLOSED, STATUS_DESC[MarketStatus.CLOSED]
    
    def is_early_close(self, d: date = None) -> bool:
        """
        判斷是否為提前收盤日（如感恩節後周五13:00收盤）
        
        參數:
            d: 日期對象（None則使用當前日期）
        
        返回:
            是否提前收盤
        """
        if d is None:
            d = self.now_et().date()
        
        # 感恩節後周五 (Black Friday) — 13:00 收盤
        # 感恩節 = 11月第4個周四
        year = d.year
        nov1 = date(year, 11, 1)
        days_to_thu = (3 - nov1.weekday()) % 7
        thanksgiving = nov1 + timedelta(days=days_to_thu + 21)  # 第4個周四
        black_friday = thanksgiving + timedelta(days=1)
        if d == black_friday:
            return True
        
        return False
    
    def is_market_open(self) -> bool:
        """
        當前是否處於正常交易時段
        
        僅 09:30-16:00 ET 返回True。
        """
        status, _ = self.get_status()
        return status == MarketStatus.REGULAR_HOURS
    
    def is_trading_active(self) -> bool:
        """
        當前是否可交易（含盤前盤後）
        """
        status, _ = self.get_status()
        return status in (MarketStatus.PRE_MARKET, MarketStatus.REGULAR_HOURS, MarketStatus.AFTER_HOURS)
    
    def countdown_to_next_open(self) -> Tuple[int, int, int]:
        """
        計算距離開市的倒計時
        
        倒計時目標：下一個交易日的正常開市時間（09:30 ET）。
        如果是周末/假期，計算到下一個交易日的09:30。
        如果已經過了當天的09:30但在休市時段，則返回(0,0,0)。
        
        返回:
            (小時, 分鐘, 秒)
        """
        now = self.now_et()
        
        # 找到下一個交易日的09:30
        target_date = now.date()
        
        # 如果當前時間已經過了今天的09:30，目標應該是明天
        if now.time() >= MARKET_TIMES['regular_start']:
            target_date += timedelta(days=1)
        
        # 往前跳到下一個交易日
        while True:
            if target_date.weekday() < 5 and not self.is_holiday(target_date):
                break
            target_date += timedelta(days=1)
        
        # 計算目標時間
        target_dt = datetime.combine(target_date, MARKET_TIMES['regular_start'])
        
        # 計算差值
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
        計算距離閉市的倒計時
        
        僅在正常交易時段有效。
        
        返回:
            (小時, 分鐘, 秒) 或 None（非交易時段）
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
        獲取完整的交易時段信息（用於面板展示）
        
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
        
        # 倒計時（始終提供）
        if status == MarketStatus.REGULAR_HOURS:
            info['countdown_to_open'] = None
            info['countdown_to_close'] = self.countdown_to_close()
        else:
            info['countdown_to_open'] = self.countdown_to_next_open()
            info['countdown_to_close'] = None
        
        return info
    
    def _next_trading_day_str(self) -> str:
        """獲取下一個交易日的字符串表示"""
        now = self.now_et()
        d = now.date() + timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d.strftime('%Y-%m-%d')
    
    def format_countdown(self, h: int, m: int, s: int) -> str:
        """
        格式化倒計時顯示
        
        參數:
            h, m, s: 時、分、秒
        
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
# 便捷函數（模塊級使用）
# ============================================================================

# 全局時鐘實例
_clock = MarketClock()


def get_market_status() -> Tuple[MarketStatus, str]:
    """獲取當前市場狀態（便捷函數）"""
    return _clock.get_status()


def countdown_to_market() -> str:
    """獲取距離開市的倒計時字符串（便捷函數）"""
    status, _ = _clock.get_status()
    
    if status == MarketStatus.REGULAR_HOURS:
        cd = _clock.countdown_to_close()
        if cd:
            return f"距閉市還有: {_clock.format_countdown(*cd)}"
        return "交易中"
    
    if status in (MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS):
        cd = _clock.countdown_to_next_open()
        return f"距正常開盤還有: {_clock.format_countdown(*cd)}"
    
    cd = _clock.countdown_to_next_open()
    return f"距離開市還有: {_clock.format_countdown(*cd)}"
