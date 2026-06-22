"""
美股量化交易系统 - 技术指标计算模块

提供高效的技术分析指标计算，所有计算均使用向量化操作，
避免Python循环，充分利用numpy/pandas的底层C实现。

支持的技术指标包括：
- 趋势类: SMA, EMA, MACD
- 动量类: RSI, Stochastic, Williams %R
- 波动类: Bollinger Bands, ATR, 历史波动率
- 成交量类: OBV, Volume Ratio, VWAP
- 自定义复合指标

时间复杂度: 所有指标计算为 O(n)，n为数据行数
空间复杂度: 大多数指标为 O(n)，部分为 O(1)

参考文献:
- Murphy, "Technical Analysis of the Financial Markets"
- Kaufman, "Trading Systems and Methods"
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


# ============================================================================
# 技术指标计算器
# ============================================================================
class TechnicalIndicators:
    """
    技术指标计算器
    
    所有方法均为静态方法或类方法，无内部状态，
    支持独立调用或批量计算。
    
    使用示例:
        df = TechnicalIndicators.add_all_indicators(price_df)
        # 或单独计算
        rsi = TechnicalIndicators.rsi(df['close'], period=14)
    """
    
    # ------------------------------------------------------------------
    # 趋势类指标
    # ------------------------------------------------------------------
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        简单移动平均线 (Simple Moving Average)
        
        SMA = (P1 + P2 + ... + Pn) / n
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            series: 价格序列
            period: 计算周期
        
        返回:
            SMA序列（前period-1个值为NaN）
        """
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        指数移动平均线 (Exponential Moving Average)
        
        EMA_t = α * P_t + (1-α) * EMA_{t-1}
        其中 α = 2/(period+1)
        
        EMA对近期价格赋予更高权重，比SMA更灵敏。
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            series: 价格序列
            period: 计算周期
        
        返回:
            EMA序列
        """
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> pd.DataFrame:
        """
        MACD指标 (Moving Average Convergence Divergence)
        
        MACD线 = EMA(fast) - EMA(slow)
        信号线 = EMA(MACD线, signal)
        柱状图 = MACD线 - 信号线
        
        用途：
        - MACD线上穿信号线 → 看涨信号
        - MACD线下穿信号线 → 看跌信号
        - 柱状图由负转正 → 动能转强
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            close: 收盘价序列
            fast: 快线周期（默认12）
            slow: 慢线周期（默认26）
            signal: 信号线周期（默认9）
        
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
    # 动量类指标
    # ------------------------------------------------------------------
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        相对强弱指标 (Relative Strength Index)
        
        RSI = 100 - 100/(1 + RS)
        其中 RS = 平均涨幅 / 平均跌幅
        
        RSI范围在0-100之间：
        - > 70: 超买区域，可能回调
        - < 30: 超卖区域，可能反弹
        - 50: 多空平衡点
        
        Wilder平滑法：使用前值加权而非简单平均
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            close: 收盘价序列
            period: 计算周期（默认14）
        
        返回:
            RSI序列
        """
        delta = close.diff()
        
        # 分别计算涨幅和跌幅
        gain = delta.clip(lower=0)    # 正值（涨幅）
        loss = -delta.clip(upper=0)   # 跌幅取正
        
        # 使用Wilder平滑法（指数移动平均）
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        # RS = 平均涨幅 / 平均跌幅
        rs = safe_divide(avg_gain, avg_loss, default=0)
        
        # RSI = 100 - 100/(1+RS)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        随机指标 (Stochastic Oscillator)
        
        %K = 100 * (Close - Lowest_Low) / (Highest_High - Lowest_Low)
        %D = SMA(%K, 3)
        
        - > 80: 超买
        - < 20: 超卖
        - %K上穿%D: 看涨
        - %K下穿%D: 看跌
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            k_period: %K计算周期
            d_period: %D平滑周期
        
        返回:
            包含 %K 和 %D 的DataFrame
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # %K = 100 * (Close - Lowest) / (Highest - Lowest)
        k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        # %D = 3日简单移动平均的%K
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({'stoch_k': k, 'stoch_d': d}, index=close.index)
    
    # ------------------------------------------------------------------
    # 波动类指标
    # ------------------------------------------------------------------
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, 
                        num_std: float = 2.0) -> pd.DataFrame:
        """
        布林带 (Bollinger Bands)
        
        中轨 = SMA(close, period)
        上轨 = 中轨 + num_std * 标准差
        下轨 = 中轨 - num_std * 标准差
        
        用途：
        - 价格触及上轨：超买，可能回调
        - 价格触及下轨：超卖，可能反弹
        - 带宽收窄：即将出现大幅波动（squeeze）
        - 带宽扩张：趋势确认
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            close: 收盘价序列
            period: 移动平均周期（默认20）
            num_std: 标准差倍数（默认2）
        
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
        平均真实波幅 (Average True Range)
        
        True Range = max(
            High - Low,
            |High - Prev_Close|,
            |Low - Prev_Close|
        )
        ATR = EMA(TR, period)
        
        ATR衡量价格波动程度（非方向），值越大表示波动越剧烈。
        常用于设置止损位和仓位大小。
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: ATR计算周期
        
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
        历史波动率 (Historical Volatility)
        
        计算方式：
        1. 计算对数收益率: r_t = ln(P_t / P_{t-1})
        2. 计算滚动标准差: σ = std(r) * sqrt(trading_days)
        
        年化波动率用于期权定价和风险管理。
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            close: 收盘价序列
            period: 计算窗口
            trading_days: 年化交易日数（美股约252天）
        
        返回:
            年化波动率序列
        """
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.rolling(window=period).std() * np.sqrt(trading_days)
        return vol
    
    # ------------------------------------------------------------------
    # 成交量类指标
    # ------------------------------------------------------------------
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        能量潮 (On-Balance Volume)
        
        规则：
        - 当日收盘 > 前日收盘: OBV += 成交量
        - 当日收盘 < 前日收盘: OBV -= 成交量
        - 当日收盘 = 前日收盘: OBV 不变
        
        OBV通过成交量变化预判价格趋势：
        - 价格上涨 + OBV上升: 上涨有量能支撑
        - 价格上涨 + OBV下降: 上涨缺乏支撑，可能反转
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        
        参数:
            close: 收盘价序列
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
        
        量比 = 当日成交量 / N日均量
        
        用途：
        - > 1.5: 放量，关注价格突破方向
        - < 0.5: 缩量，市场观望
        - ≈ 1.0: 正常成交量
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        """
        avg_volume = volume.rolling(window=period, min_periods=1).mean()
        return safe_divide(volume, avg_volume, default=1.0)
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series) -> pd.Series:
        """
        成交量加权平均价格 (Volume-Weighted Average Price)
        
        VWAP = Σ(典型价格 * 成交量) / Σ成交量
        
        其中 典型价格 = (High + Low + Close) / 3
        
        VWAP是机构交易者常用的基准价格，
        用于衡量成交质量（是否优于市场均价）。
        
        时间复杂度: O(n)
        空间复杂度: O(n)
        """
        typical_price = (high + low + close) / 3.0
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_pv / cumulative_vol
        return vwap
    
    # ------------------------------------------------------------------
    # 复合指标：一次性计算所有常用指标
    # ------------------------------------------------------------------
    
    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        为DataFrame添加所有常用技术指标
        
        这是一个便利方法，一次性计算所有ML模型可能用到的特征。
        返回一个新的DataFrame，不修改原数据。
        
        时间复杂度: O(n*k)，n为行数，k为指标数量
        空间复杂度: O(n*k)
        
        参数:
            df: 包含 open, high, low, close, volume 的DataFrame
        
        返回:
            添加了所有技术指标的DataFrame
        """
        result = df.copy()
        
        # 确保价格列为float类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in result.columns:
                result[col] = result[col].astype(float)
        
        close = result['close']
        high = result['high']
        low = result['low']
        volume = result['volume']
        open_price = result['open']
        
        # ---- 收益率特征 ----
        result['returns_1d'] = close.pct_change(1)
        result['returns_5d'] = close.pct_change(5)
        result['returns_20d'] = close.pct_change(20)
        result['log_returns'] = np.log(close / close.shift(1))
        
        # ---- 波动率特征 ----
        result['volatility_5d'] = cls.historical_volatility(close, 5, 252)
        result['volatility_20d'] = cls.historical_volatility(close, 20, 252)
        
        # ---- 趋势类指标 ----
        result['sma_5'] = cls.sma(close, 5)
        result['sma_20'] = cls.sma(close, 20)
        result['sma_60'] = cls.sma(close, 60)
        result['ema_12'] = cls.ema(close, 12)
        result['ema_26'] = cls.ema(close, 26)
        
        # 价格相对于均线的偏离度
        result['price_vs_sma20'] = safe_divide(close - result['sma_20'], result['sma_20'], 0)
        result['price_vs_sma60'] = safe_divide(close - result['sma_60'], result['sma_60'], 0)
        
        # ---- MACD ----
        macd_df = cls.macd(close)
        result['macd'] = macd_df['macd']
        result['macd_signal'] = macd_df['macd_signal']
        result['macd_hist'] = macd_df['macd_hist']
        
        # ---- RSI ----
        result['rsi_14'] = cls.rsi(close, 14)
        
        # ---- 布林带 ----
        bb_df = cls.bollinger_bands(close)
        result['bb_upper'] = bb_df['bb_upper']
        result['bb_middle'] = bb_df['bb_middle']
        result['bb_lower'] = bb_df['bb_lower']
        # 布林带宽度（归一化）
        result['bb_width'] = safe_divide(
            result['bb_upper'] - result['bb_lower'],
            result['bb_middle'], 0
        )
        # 价格在布林带中的位置（%B指标）
        result['bb_pct'] = safe_divide(
            close - result['bb_lower'],
            result['bb_upper'] - result['bb_lower'], 0.5
        )
        
        # ---- ATR ----
        result['atr_14'] = cls.atr(high, low, close, 14)
        # ATR归一化（相对于价格）
        result['atr_pct'] = safe_divide(result['atr_14'], close, 0)
        
        # ---- 成交量类 ----
        result['volume_ratio'] = cls.volume_ratio(volume, 20)
        result['volume_sma_20'] = cls.sma(volume, 20)
        result['obv'] = cls.obv(close, volume)
        
        # ---- 价格形态特征 ----
        # 上下影线长度（相对于实体）
        body = (close - open_price).abs()
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        result['upper_shadow_ratio'] = safe_divide(upper_shadow, body + 1e-8, 0)
        result['lower_shadow_ratio'] = safe_divide(lower_shadow, body + 1e-8, 0)
        
        # 日内振幅
        result['daily_range'] = safe_divide(high - low, close, 0)
        
        # ---- 标签列（用于ML训练） ----
        # 未来1日和5日的收益率（用于预测）
        result['target_1d'] = close.shift(-1) / close - 1  # 未来1日收益率
        result['target_5d'] = close.shift(-5) / close - 1  # 未来5日收益率
        # 方向标签: 1=上涨, 0=下跌
        result['target_direction_1d'] = (result['target_1d'] > 0).astype(int)
        result['target_direction_5d'] = (result['target_5d'] > 0).astype(int)
        
        logger.debug(f"已为DataFrame添加技术指标，形状: {result.shape}")
        
        return result
    
    @classmethod
    def get_feature_columns(cls) -> List[str]:
        """
        获取ML模型使用的特征列名列表
        
        返回:
            特征列名列表
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
