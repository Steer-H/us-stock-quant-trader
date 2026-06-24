"""Technical indicators computation."""

import logging
from typing import List

import numpy as np
import pandas as pd

from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


# ============================================================================
# 技術指標計算器
# ============================================================================
class TechnicalIndicators:
    """
    技術指標計算器
    
    所有方法均為靜態方法或類方法，無內部狀態，
    支持獨立調用或批量計算。
    
    使用示例:
        df = TechnicalIndicators.add_all_indicators(price_df)
        # 或單獨計算
        rsi = TechnicalIndicators.rsi(df['close'], period=14)
    """
    
    # ------------------------------------------------------------------
    # 趨勢類指標
    # ------------------------------------------------------------------
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        簡單移動平均線 (Simple Moving Average)
        
        SMA = (P1 + P2 + ... + Pn) / n
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            series: 價格序列
            period: 計算周期
        
        返回:
            SMA序列（前period-1個值為NaN）
        """
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        指數移動平均線 (Exponential Moving Average)
        
        EMA_t = α * P_t + (1-α) * EMA_{t-1}
        其中 α = 2/(period+1)
        
        EMA對近期價格賦予更高權重，比SMA更靈敏。
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            series: 價格序列
            period: 計算周期
        
        返回:
            EMA序列
        """
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> pd.DataFrame:
        """
        MACD指標 (Moving Average Convergence Divergence)
        
        MACD線 = EMA(fast) - EMA(slow)
        信號線 = EMA(MACD線, signal)
        柱狀圖 = MACD線 - 信號線
        
        用途：
        - MACD線上穿信號線 → 看漲信號
        - MACD線下穿信號線 → 看跌信號
        - 柱狀圖由負轉正 → 動能轉強
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            close: 收盤價序列
            fast: 快線周期（默認12）
            slow: 慢線周期（默認26）
            signal: 信號線周期（默認9）
        
        返回:
            包含 macd, signal, histogram 三列的DataFrame
        """
        ema_fast = TechnicalIndicators.ema(close, fast)
        ema_slow = TechnicalIndicators.ema(close, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }, index=close.index)
    
    # ------------------------------------------------------------------
    # 動量類指標
    # ------------------------------------------------------------------
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        相對強弱指標 (Relative Strength Index)
        
        RSI = 100 - 100/(1 + RS)
        其中 RS = 平均漲幅 / 平均跌幅
        
        RSI範圍在0-100之間：
        - > 70: 超買區域，可能回調
        - < 30: 超賣區域，可能反彈
        - 50: 多空平衡點
        
        Wilder平滑法：使用前值加權而非簡單平均
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            close: 收盤價序列
            period: 計算周期（默認14）
        
        返回:
            RSI序列
        """
        delta = close.diff()
        
        # 分別計算漲幅和跌幅
        gain = delta.clip(lower=0)    # 正值（漲幅）
        loss = -delta.clip(upper=0)   # 跌幅取正
        
        # 使用Wilder平滑法（指數移動平均）
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        # RS = 平均漲幅 / 平均跌幅
        rs = safe_divide(avg_gain, avg_loss, default=0)
        
        # RSI = 100 - 100/(1+RS)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        隨機指標 (Stochastic Oscillator)
        
        %K = 100 * (Close - Lowest_Low) / (Highest_High - Lowest_Low)
        %D = SMA(%K, 3)
        
        - > 80: 超買
        - < 20: 超賣
        - %K上穿%D: 看漲
        - %K下穿%D: 看跌
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            high: 最高價序列
            low: 最低價序列
            close: 收盤價序列
            k_period: %K計算周期
            d_period: %D平滑周期
        
        返回:
            包含 %K 和 %D 的DataFrame
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # %K = 100 * (Close - Lowest) / (Highest - Lowest)
        k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        # %D = 3日簡單移動平均的%K
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({'stoch_k': k, 'stoch_d': d}, index=close.index)
    
    # ------------------------------------------------------------------
    # 波動類指標
    # ------------------------------------------------------------------
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, 
                        num_std: float = 2.0) -> pd.DataFrame:
        """
        布林帶 (Bollinger Bands)
        
        中軌 = SMA(close, period)
        上軌 = 中軌 + num_std * 標準差
        下軌 = 中軌 - num_std * 標準差
        
        用途：
        - 價格觸及上軌：超買，可能回調
        - 價格觸及下軌：超賣，可能反彈
        - 帶寬收窄：即將出現大幅波動（squeeze）
        - 帶寬擴張：趨勢確認
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            close: 收盤價序列
            period: 移動平均周期（默認20）
            num_std: 標準差倍數（默認2）
        
        返回:
            包含 upper, middle, lower 的DataFrame
        """
        middle = TechnicalIndicators.sma(close, period)
        std = close.rolling(window=period, min_periods=1).std()
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        }, index=close.index)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        平均真實波幅 (Average True Range)
        
        True Range = max(
            High - Low,
            |High - Prev_Close|,
            |Low - Prev_Close|
        )
        ATR = EMA(TR, period)
        
        ATR衡量價格波動程度（非方向），值越大表示波動越劇烈。
        常用於設置止損位和倉位大小。
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            high: 最高價序列
            low: 最低價序列
            close: 收盤價序列
            period: ATR計算周期
        
        返回:
            ATR序列
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 使用Wilder平滑法
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    @staticmethod
    def historical_volatility(close: pd.Series, period: int = 20,
                              trading_days: int = 252) -> pd.Series:
        """
        歷史波動率 (Historical Volatility)
        
        計算方式：
        1. 計算對數收益率: r_t = ln(P_t / P_{t-1})
        2. 計算滾動標準差: σ = std(r) * sqrt(trading_days)
        
        年化波動率用於期權定價和風險管理。
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            close: 收盤價序列
            period: 計算窗口
            trading_days: 年化交易日數（美股約252天）
        
        返回:
            年化波動率序列
        """
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.rolling(window=period).std() * np.sqrt(trading_days)
        return vol
    
    # ------------------------------------------------------------------
    # 成交量類指標
    # ------------------------------------------------------------------
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        能量潮 (On-Balance Volume)
        
        規則：
        - 當日收盤 > 前日收盤: OBV += 成交量
        - 當日收盤 < 前日收盤: OBV -= 成交量
        - 當日收盤 = 前日收盤: OBV 不變
        
        OBV通過成交量變化預判價格趨勢：
        - 價格上漲 + OBV上升: 上漲有量能支撐
        - 價格上漲 + OBV下降: 上漲缺乏支撐，可能反轉
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        
        參數:
            close: 收盤價序列
            volume: 成交量序列
        
        返回:
            OBV序列
        """
        price_change = close.diff()
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        obv = (direction * volume).cumsum()
        return pd.Series(obv, index=close.index, name='obv')
    
    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        量比 (Volume Ratio)
        
        量比 = 當日成交量 / N日均量
        
        用途：
        - > 1.5: 放量，關注價格突破方向
        - < 0.5: 縮量，市場觀望
        - ≈ 1.0: 正常成交量
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        """
        avg_volume = volume.rolling(window=period, min_periods=1).mean()
        return safe_divide(volume, avg_volume, default=1.0)
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series) -> pd.Series:
        """
        成交量加權平均價格 (Volume-Weighted Average Price)
        
        VWAP = Σ(典型價格 * 成交量) / Σ成交量
        
        其中 典型價格 = (High + Low + Close) / 3
        
        VWAP是機構交易者常用的基準價格，
        用于衡量成交質量（是否優於市場均價）。
        
        時間複雜度: O(n)
        空間複雜度: O(n)
        """
        typical_price = (high + low + close) / 3.0
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_pv / cumulative_vol
        return vwap
    
    # ------------------------------------------------------------------
    # 複合指標：一次性計算所有常用指標
    # ------------------------------------------------------------------
    
    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        為DataFrame添加所有常用技術指標
        
        這是一個便利方法，一次性計算所有ML模型可能用到的特徵。
        返回一個新的DataFrame，不修改原數據。
        
        時間複雜度: O(n*k)，n為行數，k為指標數量
        空間複雜度: O(n*k)
        
        參數:
            df: 包含 open, high, low, close, volume 的DataFrame
        
        返回:
            添加了所有技術指標的DataFrame
        """
        result = df.copy()
        
        # 確保價格列為float類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in result.columns:
                result[col] = result[col].astype(float)
        
        close = result['close']
        high = result['high']
        low = result['low']
        volume = result['volume']
        open_price = result['open']
        
        # ---- 收益率特徵 ----
        result['returns_1d'] = close.pct_change(1)
        result['returns_5d'] = close.pct_change(5)
        result['returns_20d'] = close.pct_change(20)
        result['log_returns'] = np.log(close / close.shift(1))
        
        # ---- 波動率特徵 ----
        result['volatility_5d'] = cls.historical_volatility(close, 5, 252)
        result['volatility_20d'] = cls.historical_volatility(close, 20, 252)
        
        # ---- 趨勢類指標 ----
        result['sma_5'] = cls.sma(close, 5)
        result['sma_20'] = cls.sma(close, 20)
        result['sma_60'] = cls.sma(close, 60)
        result['ema_12'] = cls.ema(close, 12)
        result['ema_26'] = cls.ema(close, 26)
        
        # 價格相對於均線的偏離度
        result['price_vs_sma20'] = safe_divide(close - result['sma_20'], result['sma_20'], 0)
        result['price_vs_sma60'] = safe_divide(close - result['sma_60'], result['sma_60'], 0)
        
        # ---- MACD ----
        macd_df = cls.macd(close)
        result['macd'] = macd_df['macd']
        result['macd_signal'] = macd_df['macd_signal']
        result['macd_hist'] = macd_df['macd_hist']
        
        # ---- RSI ----
        result['rsi_14'] = cls.rsi(close, 14)
        
        # ---- 布林帶 ----
        bb_df = cls.bollinger_bands(close)
        result['bb_upper'] = bb_df['bb_upper']
        result['bb_middle'] = bb_df['bb_middle']
        result['bb_lower'] = bb_df['bb_lower']
        # 布林帶寬度（歸一化）
        result['bb_width'] = safe_divide(
            result['bb_upper'] - result['bb_lower'],
            result['bb_middle'], 0
        )
        # 價格在布林帶中的位置（%B指標）
        result['bb_pct'] = safe_divide(
            close - result['bb_lower'],
            result['bb_upper'] - result['bb_lower'], 0.5
        )
        
        # ---- ATR ----
        result['atr_14'] = cls.atr(high, low, close, 14)
        # ATR歸一化（相對於價格）
        result['atr_pct'] = safe_divide(result['atr_14'], close, 0)
        
        # ---- 成交量類 ----
        result['volume_ratio'] = cls.volume_ratio(volume, 20)
        result['volume_sma_20'] = cls.sma(volume, 20)
        result['obv'] = cls.obv(close, volume)
        
        # ---- 價格形態特徵 ----
        # 上下影線長度（相對於實體）
        body = (close - open_price).abs()
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        result['upper_shadow_ratio'] = safe_divide(upper_shadow, body + 1e-8, 0)
        result['lower_shadow_ratio'] = safe_divide(lower_shadow, body + 1e-8, 0)
        
        # 日內振幅
        result['daily_range'] = safe_divide(high - low, close, 0)
        
        # ---- 標籤列（用於ML訓練） ----
        # 未來1日和5日的收益率（用於預測）
        result['target_1d'] = close.shift(-1) / close - 1  # 未來1日收益率
        result['target_5d'] = close.shift(-5) / close - 1  # 未來5日收益率
        # 方向標籤: 1=上漲, 0=下跌
        result['target_direction_1d'] = (result['target_1d'] > 0).astype(int)
        result['target_direction_5d'] = (result['target_5d'] > 0).astype(int)
        
        logger.debug(f"已為DataFrame添加技術指標，形狀: {result.shape}")
        
        return result
    
    @classmethod
    def get_feature_columns(cls) -> List[str]:
        """
        獲取ML模型使用的特徵列名列表
        
        返回:
            特徵列名列表
        """
        return [
            'open', 'high', 'low', 'close', 'volume',
            'returns_1d', 'returns_5d', 'returns_20d',
            'volatility_5d', 'volatility_20d',
            'sma_5', 'sma_20', 'sma_60',
            'ema_12', 'ema_26',
            'price_vs_sma20', 'price_vs_sma60',
            'macd', 'macd_signal', 'macd_hist',
            'rsi_14',
            'bb_upper', 'bb_middle', 'bb_lower',
            'bb_width', 'bb_pct',
            'atr_14', 'atr_pct',
            'volume_ratio', 'volume_sma_20',
            'upper_shadow_ratio', 'lower_shadow_ratio',
            'daily_range',
        ]
