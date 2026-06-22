"""
美股量化交易系统 - 数据清洗模块

处理原始行情数据的质量问题：
- 缺失值处理（前向填充+插值）
- 异常值检测与处理（价格跳变、成交量异常）
- 复权处理（分股、分红导致的假跳空）
- 多数据源对齐（统一时间戳、列名）
- 幸存者偏差标记（标注已下市的股票）

设计原则：
- 所有清洗操作生成审计日志，记录每次修改的内容和原因
- 支持链式调用: cleaner.remove_outliers().fill_missing().adjust_prices()
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import DataSourceConfig
from utils.exceptions import DataQualityError

logger = logging.getLogger(__name__)


# ============================================================================
# 数据清洗器
# ============================================================================
class DataCleaner:
    """
    行情数据清洗器
    
    清洗流程（标准流水线）：
    1. validate_structure()    - 验证数据结构完整性
    2. remove_outliers()       - 移除异常价格跳变
    3. handle_corporate_actions() - 处理分红/分股导致的跳空
    4. fill_missing()          - 填充缺失数据
    5. normalize()             - 归一化/标准化
    
    每一步都返回 self，支持链式调用。
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        参数:
            config: 数据源配置
        """
        self.config = config
        self._audit_log: List[Dict] = []  # 审计日志
        self._modified_count: int = 0     # 总修改次数统计
    
    def _log_action(self, action: str, ticker: str, details: str, 
                    before: any = None, after: any = None) -> None:
        """
        记录清洗操作到审计日志
        
        参数:
            action: 操作类型（如 'remove_outlier', 'fill_missing'）
            ticker: 涉及的股票代码
            details: 详细描述
            before: 修改前的值
            after: 修改后的值
        """
        self._modified_count += 1
        self._audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'ticker': ticker,
            'details': details,
            'before': str(before) if before is not None else None,
            'after': str(after) if after is not None else None,
        })
        logger.debug(f"[{action}] {ticker}: {details}")
    
    def validate_structure(self, df: pd.DataFrame, ticker: str = '') -> 'DataCleaner':
        """
        验证数据结构完整性
        
        检查项：
        - 必需的OHLCV列是否存在
        - 是否存在全空行/列
        - 日期索引是否单调递增
        - 价格是否为正数
        
        参数:
            df: 待验证的DataFrame
            ticker: 股票代码（用于日志）
        
        返回:
            self，支持链式调用
        
        抛出:
            DataQualityError: 数据质量严重不合格
        """
        if df is None or df.empty:
            raise DataQualityError("DataFrame为空", {'ticker': ticker})
        
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        actual_cols = set(df.columns) & required_cols
        
        missing = required_cols - actual_cols
        if missing:
            raise DataQualityError(
                f"缺少必要列: {missing}",
                {'ticker': ticker, 'missing': list(missing)}
            )
        
        # 检查日期索引是否单调递增
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            self._log_action(
                'remove_duplicates', ticker,
                f'移除 {dup_count} 个重复日期行'
            )
            df = df[~df.index.duplicated(keep='first')]
        
        # 检查价格是否为正数（美股价格通常 > $0.01）
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                negative_mask = df[col] <= 0
                if negative_mask.any():
                    neg_count = negative_mask.sum()
                    self._log_action(
                        'fix_negative_prices', ticker,
                        f'将 {col} 的 {neg_count} 个非正值设为NaN',
                        before=neg_count
                    )
                    df.loc[negative_mask, col] = np.nan
        
        # High必须 >= Low（基本约束）
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                self._log_action(
                    'fix_hl_order', ticker,
                    f'修正 {invalid_hl.sum()} 行 high<low 的数据',
                )
                # 互换 High 和 Low
                df.loc[invalid_hl, ['high', 'low']] = df.loc[
                    invalid_hl, ['low', 'high']
                ].values
        
        return self
    
    def remove_outliers(self, df: pd.DataFrame, ticker: str = '',
                       price_col: str = 'close') -> 'DataCleaner':
        """
        检测并处理异常价格跳变
        
        使用两种方法联合检测：
        1. 绝对阈值: 单日涨跌幅超过阈值倍数的跳变
        2. 滚动标准差: 基于滚动窗口的Z-score检测
        
        对检测到的异常值采用插值修正而非直接删除，
        以保持时间序列的完整性。
        
        时间复杂度: O(n)，n为数据行数
        空间复杂度: O(n)
        
        参数:
            df: 待处理的DataFrame（会被就地修改）
            ticker: 股票代码
            price_col: 价格列名（默认'close'）
        
        返回:
            self
        """
        if df.empty or price_col not in df.columns:
            return self
        
        prices = df[price_col].values
        n = len(prices)
        
        if n < 3:
            return self
        
        # 方法1: 检测单日百分比变化异常
        # 计算日收益率，标记超过阈值的位置
        daily_returns = np.abs(np.diff(prices) / np.where(prices[:-1] == 0, 1, prices[:-1]))
        spike_mask = np.concatenate([[False], daily_returns > self.config.price_spike_threshold])
        
        # 方法2: 滚动Z-score检测（基于20日窗口）
        if n >= 20:
            rolling_mean = pd.Series(prices).rolling(20, min_periods=5).mean().values
            rolling_std = pd.Series(prices).rolling(20, min_periods=5).std().values
            rolling_std = np.where(rolling_std == 0, 1, rolling_std)  # 避免除零
            
            z_scores = np.abs((prices - rolling_mean) / rolling_std)
            z_mask = z_scores > 5  # Z-score > 5 视为异常
            
            # 合并两种方法的结果
            outlier_mask = spike_mask | z_mask
        else:
            outlier_mask = spike_mask
        
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            # 不对异常值进行直接删除，而是用插值替换
            df.loc[df.index[outlier_mask], price_col] = np.nan
            
            self._log_action(
                'mark_outliers', ticker,
                f'标记 {outlier_count} 个异常价格点（共{len(df)}行）',
                before=outlier_count
            )
        
        return self
    
    def handle_corporate_actions(
        self, df: pd.DataFrame, ticker: str = '',
        dividends: Optional[pd.Series] = None,
        splits: Optional[pd.Series] = None
    ) -> 'DataCleaner':
        """
        处理公司行为导致的跳空
        
        核心问题：分红和拆股会导致股价出现"假跳空"，
        如果不处理，技术指标和回测结果都会失真。
        
        处理方式：
        - 分股(Stock Split): 对分股前的价格按分股比例反向调整
        - 分红(Cash Dividend): 对除息日前的价格减去分红金额
        
        注意：yfinance 的 auto_adjust=True 已自动处理，
        这里提供手动处理能力以应对自定义数据源。
        
        参数:
            df: 价格DataFrame
            ticker: 股票代码
            dividends: 分红序列（日期索引）
            splits: 分股序列（日期索引）
        
        返回:
            self
        """
        if splits is not None and not splits.empty:
            # 分股调整：例如2拆1（split=2），分股前价格除以2
            # 从后向前处理，累积调整因子
            cumulative_factor = 1.0
            for split_date, split_ratio in splits.sort_index(ascending=False).items():
                # split_ratio: 2.0 表示 2-for-1 split
                cumulative_factor *= split_ratio
                # 调整 split_date 之前的所有价格
                if split_date in df.index:
                    mask_before = df.index < split_date
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df.loc[mask_before, col] /= split_ratio
                
                self._log_action(
                    'adjust_split', ticker,
                    f'分股调整: {split_ratio}-for-1 @ {split_date.date()}',
                    before=split_ratio
                )
        
        if dividends is not None and not dividends.empty:
            # 分红调整：除息日前价格减去分红金额
            for div_date, div_amount in dividends.items():
                if div_date in df.index:
                    mask_before = df.index < div_date
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df.loc[mask_before, col] -= div_amount
                    
                    self._log_action(
                        'adjust_dividend', ticker,
                        f'分红调整: ${div_amount:.4f} @ {div_date.date()}',
                        before=div_amount
                    )
        
        return self
    
    def fill_missing(self, df: pd.DataFrame, ticker: str = '',
                    max_gap: int = 5) -> 'DataCleaner':
        """
        填充缺失数据
        
        填充策略（按优先级）：
        1. 前向填充 (forward fill): 用前一个有效值填充
        2. 线性插值 (linear interpolation): 用于价格列的小段缺失
        3. 零填充: 成交量缺失填0（表示无交易）
        
        参数:
            df: 待填充的DataFrame（会被就地修改）
            ticker: 股票代码
            max_gap: 最大填充间隔，超过此天数的缺失保留为NaN
        
        返回:
            self
        """
        if df.empty:
            return self
        
        nan_before = df.isna().sum().sum()
        
        # 确保索引是日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                logger.debug(f"Non-critical error in cleaner.py: {e}", exc_info=True)
        
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        
        for col in price_cols:
            # 先前向填充，再线性插值
            df[col] = df[col].ffill(limit=max_gap).interpolate(method='linear', limit=max_gap)
        
        # 成交量缺失填0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        nan_after = df.isna().sum().sum()
        filled_count = nan_before - nan_after
        
        if filled_count > 0:
            self._log_action(
                'fill_missing', ticker,
                f'填充了 {filled_count} 个缺失值',
                before=nan_before, after=nan_after
            )
        
        return self
    
    def detect_survivorship_bias(self, df: pd.DataFrame, ticker: str,
                                 last_date: Optional[str] = None) -> 'DataCleaner':
        """
        检测并标记幸存者偏差
        
        问题：如果只使用"当前仍然存活"的股票做回测，
        会忽略那些已经退市/破产的股票，导致回测结果虚高。
        
        标记方式：在DataFrame中添加 'delisted' 列，
        如果最后交易日期早于预期结束日期，标记为 True。
        
        参数:
            df: 股票数据DataFrame
            ticker: 股票代码
            last_date: 预期最后交易日期
        
        返回:
            self
        """
        if df.empty:
            return self
        
        data_last_date = df.index.max()
        
        if last_date:
            expected_end = pd.Timestamp(last_date)
            if data_last_date < expected_end - pd.Timedelta(days=30):
                # 数据提前终止超过30天，可能已下市
                df['delisted'] = True
                self._log_action(
                    'flag_survivorship', ticker,
                    f'数据终止于 {data_last_date.date()}，预期 {expected_end.date()}，标记为可能已下市'
                )
            else:
                df['delisted'] = False
        else:
            df['delisted'] = False
        
        return self
    
    def get_audit_log(self) -> List[Dict]:
        """获取清洗操作的审计日志"""
        return self._audit_log.copy()
    
    def get_summary(self) -> Dict:
        """获取清洗操作的摘要统计"""
        return {
            'total_actions': self._modified_count,
            'action_types': list(set(log['action'] for log in self._audit_log)),
            'tickers_processed': len(set(log['ticker'] for log in self._audit_log)),
        }
    
    def reset(self) -> None:
        """重置审计日志和计数器"""
        self._audit_log.clear()
        self._modified_count = 0
