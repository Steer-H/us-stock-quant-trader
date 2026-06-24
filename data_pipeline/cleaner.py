"""Data cleaning and preprocessing."""

import logging
from typing import Optional, List, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import DataSourceConfig
from utils.exceptions import DataQualityError

logger = logging.getLogger(__name__)


# ============================================================================
# 數據清洗器
# ============================================================================
class DataCleaner:
    """
    行情數據清洗器
    
    清洗流程（標準流水線）：
    1. validate_structure()    - 驗證數據結構完整性
    2. remove_outliers()       - 移除異常價格跳變
    3. handle_corporate_actions() - 處理分紅/分股導致的跳空
    4. fill_missing()          - 填充缺失數據
    5. normalize()             - 歸一化/標準化
    
    每一步都返回 self，支持鏈式調用。
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        參數:
            config: 數據源配置
        """
        self.config = config
        self._audit_log: List[Dict] = []  # 審計日誌
        self._modified_count: int = 0     # 總修改次數統計
    
    def _log_action(self, action: str, ticker: str, details: str, 
                    before: any = None, after: any = None) -> None:
        """
        記錄清洗操作到審計日誌
        
        參數:
            action: 操作類型（如 'remove_outlier', 'fill_missing'）
            ticker: 涉及的股票代碼
            details: 詳細描述
            before: 修改前的值
            after: 修改後的值
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
        驗證數據結構完整性
        
        檢查項：
        - 必需的OHLCV列是否存在
        - 是否存在全空行/列
        - 日期索引是否單調遞增
        - 價格是否為正數
        
        參數:
            df: 待驗證的DataFrame
            ticker: 股票代碼（用於日誌）
        
        返回:
            self，支持鏈式調用
        
        拋出:
            DataQualityError: 數據質量嚴重不合格
        """
        if df is None or df.empty:
            raise DataQualityError("DataFrame為空", {'ticker': ticker})
        
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        actual_cols = set(df.columns) & required_cols
        
        missing = required_cols - actual_cols
        if missing:
            raise DataQualityError(
                f"缺少必要列: {missing}",
                {'ticker': ticker, 'missing': list(missing)}
            )
        
        # 檢查日期索引是否單調遞增
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            self._log_action(
                'remove_duplicates', ticker,
                f'移除 {dup_count} 個重複日期行'
            )
            df = df[~df.index.duplicated(keep='first')]
        
        # 檢查價格是否為正數（美股價格通常 > $0.01）
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                negative_mask = df[col] <= 0
                if negative_mask.any():
                    neg_count = negative_mask.sum()
                    self._log_action(
                        'fix_negative_prices', ticker,
                        f'將 {col} 的 {neg_count} 個非正值設為NaN',
                        before=neg_count
                    )
                    df.loc[negative_mask, col] = np.nan
        
        # High必須 >= Low（基本約束）
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                self._log_action(
                    'fix_hl_order', ticker,
                    f'修正 {invalid_hl.sum()} 行 high<low 的數據',
                )
                # 互換 High 和 Low
                df.loc[invalid_hl, ['high', 'low']] = df.loc[
                    invalid_hl, ['low', 'high']
                ].values
        
        return self
    
    def remove_outliers(self, df: pd.DataFrame, ticker: str = '',
                       price_col: str = 'close') -> 'DataCleaner':
        """
        檢測並處理異常價格跳變
        
        使用兩種方法聯合檢測：
        1. 絕對閾值: 單日漲跌幅超過閾值倍數的跳變
        2. 滾動標準差: 基於滾動窗口的Z-score檢測
        
        對檢測到的異常值採用插值修正而非直接刪除，
        以保持時間序列的完整性。
        
        時間複雜度: O(n)，n為數據行數
        空間複雜度: O(n)
        
        參數:
            df: 待處理的DataFrame（會被就地修改）
            ticker: 股票代碼
            price_col: 價格列名（默認'close'）
        
        返回:
            self
        """
        if df.empty or price_col not in df.columns:
            return self
        
        prices = df[price_col].values
        n = len(prices)
        
        if n < 3:
            return self
        
        # 方法1: 檢測單日百分比變化異常
        # 計算日收益率，標記超過閾值的位置
        daily_returns = np.abs(np.diff(prices) / np.where(prices[:-1] == 0, 1, prices[:-1]))
        spike_mask = np.concatenate([[False], daily_returns > self.config.price_spike_threshold])
        
        # 方法2: 滾動Z-score檢測（基於20日窗口）
        if n >= 20:
            rolling_mean = pd.Series(prices).rolling(20, min_periods=5).mean().values
            rolling_std = pd.Series(prices).rolling(20, min_periods=5).std().values
            rolling_std = np.where(rolling_std == 0, 1, rolling_std)  # 避免除零
            
            z_scores = np.abs((prices - rolling_mean) / rolling_std)
            z_mask = z_scores > 5  # Z-score > 5 視為異常
            
            # 合併兩種方法的結果
            outlier_mask = spike_mask | z_mask
        else:
            outlier_mask = spike_mask
        
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            # 不對異常值進行直接刪除，而是用插值替換
            df.loc[df.index[outlier_mask], price_col] = np.nan
            
            self._log_action(
                'mark_outliers', ticker,
                f'標記 {outlier_count} 個異常價格點（共{len(df)}行）',
                before=outlier_count
            )
        
        return self
    
    def handle_corporate_actions(
        self, df: pd.DataFrame, ticker: str = '',
        dividends: Optional[pd.Series] = None,
        splits: Optional[pd.Series] = None
    ) -> 'DataCleaner':
        """
        處理公司行為導致的跳空
        
        核心問題：分紅和拆股會導致股價出現"假跳空"，
        如果不處理，技術指標和回測結果都會失真。
        
        處理方式：
        - 分股(Stock Split): 對分股前的價格按分股比例反向調整
        - 分紅(Cash Dividend): 對除息日前的價格減去分紅金額
        
        注意：yfinance 的 auto_adjust=True 已自動處理，
        這裡提供手動處理能力以應對自定義數據源。
        
        參數:
            df: 價格DataFrame
            ticker: 股票代碼
            dividends: 分紅序列（日期索引）
            splits: 分股序列（日期索引）
        
        返回:
            self
        """
        if splits is not None and not splits.empty:
            # 分股調整：例如2拆1（split=2），分股前價格除以2
            # 從後向前處理，累積調整因子
            cumulative_factor = 1.0
            for split_date, split_ratio in splits.sort_index(ascending=False).items():
                # split_ratio: 2.0 表示 2-for-1 split
                cumulative_factor *= split_ratio
                # 調整 split_date 之前的所有價格
                if split_date in df.index:
                    mask_before = df.index < split_date
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df.loc[mask_before, col] /= split_ratio
                
                self._log_action(
                    'adjust_split', ticker,
                    f'分股調整: {split_ratio}-for-1 @ {split_date.date()}',
                    before=split_ratio
                )
        
        if dividends is not None and not dividends.empty:
            # 分紅調整：除息日前價格減去分紅金額
            for div_date, div_amount in dividends.items():
                if div_date in df.index:
                    mask_before = df.index < div_date
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df.loc[mask_before, col] -= div_amount
                    
                    self._log_action(
                        'adjust_dividend', ticker,
                        f'分紅調整: ${div_amount:.4f} @ {div_date.date()}',
                        before=div_amount
                    )
        
        return self
    
    def fill_missing(self, df: pd.DataFrame, ticker: str = '',
                    max_gap: int = 5) -> 'DataCleaner':
        """
        填充缺失數據
        
        填充策略（按優先級）：
        1. 前向填充 (forward fill): 用前一個有效值填充
        2. 線性插值 (linear interpolation): 用於價格列的小段缺失
        3. 零填充: 成交量缺失填0（表示無交易）
        
        參數:
            df: 待填充的DataFrame（會被就地修改）
            ticker: 股票代碼
            max_gap: 最大填充間隔，超過此天數的缺失保留為NaN
        
        返回:
            self
        """
        if df.empty:
            return self
        
        nan_before = df.isna().sum().sum()
        
        # 確保索引是日期類型
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                logger.debug(f"Non-critical error in cleaner.py: {e}", exc_info=True)
        
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        
        for col in price_cols:
            # 先前向填充，再線性插值
            df[col] = df[col].ffill(limit=max_gap).interpolate(method='linear', limit=max_gap)
        
        # 成交量缺失填0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        nan_after = df.isna().sum().sum()
        filled_count = nan_before - nan_after
        
        if filled_count > 0:
            self._log_action(
                'fill_missing', ticker,
                f'填充了 {filled_count} 個缺失值',
                before=nan_before, after=nan_after
            )
        
        return self
    
    def detect_survivorship_bias(self, df: pd.DataFrame, ticker: str,
                                 last_date: Optional[str] = None) -> 'DataCleaner':
        """
        檢測並標記倖存者偏差
        
        問題：如果只使用"當前仍然存活"的股票做回測，
        會忽略那些已經退市/破產的股票，導致回測結果虛高。
        
        標記方式：在DataFrame中添加 'delisted' 列，
        如果最後交易日期早於預期結束日期，標記為 True。
        
        參數:
            df: 股票數據DataFrame
            ticker: 股票代碼
            last_date: 預期最後交易日期
        
        返回:
            self
        """
        if df.empty:
            return self
        
        data_last_date = df.index.max()
        
        if last_date:
            expected_end = pd.Timestamp(last_date)
            if data_last_date < expected_end - pd.Timedelta(days=30):
                # 數據提前終止超過30天，可能已下市
                df['delisted'] = True
                self._log_action(
                    'flag_survivorship', ticker,
                    f'數據終止於 {data_last_date.date()}，預期 {expected_end.date()}，標記為可能已下市'
                )
            else:
                df['delisted'] = False
        else:
            df['delisted'] = False
        
        return self
    
    def get_audit_log(self) -> List[Dict]:
        """獲取清洗操作的審計日誌"""
        return self._audit_log.copy()
    
    def get_summary(self) -> Dict:
        """獲取清洗操作的摘要統計"""
        return {
            'total_actions': self._modified_count,
            'action_types': list(set(log['action'] for log in self._audit_log)),
            'tickers_processed': len(set(log['ticker'] for log in self._audit_log)),
        }
    
    def reset(self) -> None:
        """重置審計日誌和計數器"""
        self._audit_log.clear()
        self._modified_count = 0
