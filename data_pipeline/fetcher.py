"""Yahoo Finance data fetcher with caching."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    DataSourceConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR, system_config
)
from utils.helpers import Timer, validate_ticker
from utils.exceptions import (
    DataSourceError, DataQualityError, ConfigurationError
)

logger = logging.getLogger(__name__)


# ============================================================================
# 抽象數據獲取器基類
# ============================================================================
class DataFetcher(ABC):
    """
    數據獲取器抽象基類
    
    所有數據源實現必須繼承此類並實現其抽象方法。
    子類只需關注數據獲取邏輯，重試、速率限制等通用邏輯由基類處理。
    """
    
    # 子類可覆蓋的元數據
    SOURCE_NAME: str = "abstract"
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0  # 秒
    RATE_LIMIT_CALLS: int = 5     # 每秒最大請求數
    RATE_LIMIT_PERIOD: float = 1.0  # 限速窗口（秒）
    
    def __init__(self, config: DataSourceConfig):
        """
        參數:
            config: 數據源配置對象
        """
        self.config = config
        self._last_call_times: List[float] = []  # 用於速率限制的調用時間戳隊列
        self._session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        創建帶重試機制的HTTP會話
        
        使用指數退避重試策略：
        - 第1次重試: 等待 1s
        - 第2次重試: 等待 2s
        - 第3次重試: 等待 4s
        總共最多重試3次
        
        返回:
            配置好的requests.Session對象
        """
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=1.0,         # 指數退避因子
            status_forcelist=[429, 500, 502, 503, 504],  # 這些狀態碼觸發重試
            allowed_methods=["GET"]      # 僅重試GET請求（冪等安全）
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,     # 連接池大小
            pool_maxsize=20          # 最大連接數
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # 設置User-Agent，避免被某些API拒絕
        session.headers.update({
            'User-Agent': 'QuantTradingSystem/1.0 (Educational Purpose)'
        })
        
        return session
    
    def _rate_limit(self) -> None:
        """
        速率限制：確保不超過API調用頻率限制
        
        使用滑動窗口算法，如果過去 RATE_LIMIT_PERIOD 秒內的調用次數
        達到 RATE_LIMIT_CALLS，則等待到下一個窗口。
        
        時間複雜度: O(k)，k為窗口內調用次數（通常很小）
        空間複雜度: O(k)
        """
        now = time.monotonic()
        window_start = now - self.RATE_LIMIT_PERIOD
        
        # 清理過期的調用記錄（滑動窗口）
        self._last_call_times = [t for t in self._last_call_times if t > window_start]
        
        if len(self._last_call_times) >= self.RATE_LIMIT_CALLS:
            # 需要等待到最早的調用過期
            wait_time = self._last_call_times[0] + self.RATE_LIMIT_PERIOD - now
            if wait_time > 0:
                logger.debug(f"速率限制: 等待 {wait_time:.2f}s")
                time.sleep(wait_time)
        
        self._last_call_times.append(now)
    
    @abstractmethod
    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        獲取股票日線數據（OHLCV）
        
        參數:
            ticker: 股票代碼 (如 'AAPL')
            start_date: 開始日期 'YYYY-MM-DD'
            end_date: 結束日期 'YYYY-MM-DD'
            auto_adjust: 是否自動復權
        
        返回:
            包含以下列的DataFrame:
            - date (索引): 日期
            - open, high, low, close: OHLC價格
            - volume: 成交量
            - adj_close: 復權收盤價（如適用）
        """
        pass
    
    @abstractmethod
    def fetch_intraday_bars(
        self,
        ticker: str,
        interval: str = '5min',
        days_back: int = 5
    ) -> pd.DataFrame:
        """
        獲取日內分時數據
        
        參數:
            ticker: 股票代碼
            interval: K線周期 ('1min', '5min', '15min', '30min', '1hour')
            days_back: 回溯天數
        
        返回:
            日內分時OHLCV DataFrame
        """
        pass
    
    @abstractmethod
    def fetch_reference_data(self, ticker: str) -> Dict[str, Any]:
        """
        獲取股票參考數據（基本面信息）
        
        參數:
            ticker: 股票代碼
        
        返回:
            包含公司名稱、市值、行業、流通股數等信息的字典
        """
        pass
    
    def fetch_batch_daily_bars(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_workers: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量獲取多隻股票的日線數據（並發請求）
        
        使用ThreadPoolExecutor實現並發IO，適用於網絡IO密集型任務。
        注意：需遵守API速率限制，不同數據源的速率限制不同。
        
        時間複雜度: O(n*m/p)，n=股票數，m=單股數據量，p=並發數
        空間複雜度: O(n*m)
        
        參數:
            tickers: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期
            max_workers: 最大並發數（默認使用系統配置）
        
        返回:
            字典: {ticker: DataFrame}
        """
        if max_workers is None:
            max_workers = system_config.max_workers
        
        results: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_ticker = {
                executor.submit(
                    self.fetch_daily_bars, ticker, start_date, end_date
                ): ticker
                for ticker in tickers
            }
            
            # 收集結果
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[ticker] = df
                        logger.debug(f"已獲取 {ticker} 數據: {len(df)} 行")
                    else:
                        failed.append(ticker)
                except Exception as e:
                    logger.warning(f"獲取 {ticker} 數據失敗: {e}")
                    failed.append(ticker)
        
        if failed:
            logger.warning(f"以下股票數據獲取失敗 ({len(failed)}/{len(tickers)}): {failed}")
        
        logger.info(f"批量數據獲取完成: 成功 {len(results)}/{len(tickers)}")
        return results
    
    def __del__(self):
        """析構時關閉HTTP會話"""
        if hasattr(self, '_session'):
            self._session.close()


# ============================================================================
# Yahoo Finance 數據獲取器
# ============================================================================
class YahooFetcher(DataFetcher):
    """
    基於Yahoo Finance的數據獲取器
    
    優點：
    - 免費，無需API密鑰
    - 覆蓋美股歷史數據全面（1970年至今）
    - 自動處理分股和分紅復權
    
    限制：
    - 速率限制嚴格（建議每秒不超過2個請求）
    - 部分已下市股票數據可能不完整
    - 日內數據僅提供最近30天
    
    實現方式：
    使用 yfinance 庫，該庫內部解析Yahoo Finance的公開API端點。
    """
    
    SOURCE_NAME = "YahooFinance"
    RATE_LIMIT_CALLS = 2  # Yahoo Finance速率限制較嚴格
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        # 延遲導入 yfinance，避免非必要依賴
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ConfigurationError(
                "yfinance 未安裝，請執行: pip install yfinance"
            )
    
    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        通過Yahoo Finance獲取日線OHLCV數據
        
        Yahoo Finance自動提供分股和分紅復權數據，
        auto_adjust=True時返回的close即為adj_close。
        
        參數:
            ticker: 股票代碼
            start_date: 開始日期 'YYYY-MM-DD'
            end_date: 結束日期 'YYYY-MM-DD'
            auto_adjust: 是否使用復權價格
        
        返回:
            OHLCV DataFrame，索引為日期
        """
        if not validate_ticker(ticker):
            raise DataSourceError(f"無效的股票代碼: {ticker}", {'ticker': ticker})
        
        self._rate_limit()
        
        try:
            with Timer(f"Yahoo獲取{ticker}日線數據"):
                stock = self.yf.Ticker(ticker)
                
                # 獲取歷史數據
                # auto_adjust=False 返回未復權數據，含dividends和stock splits列
                # auto_adjust=True 返回復權後的OHLC數據
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    actions=True  # 包含分紅和分股信息
                )
                
                if df.empty:
                    raise DataQualityError(
                        f"{ticker} 在 {start_date}~{end_date} 期間無數據",
                        {'ticker': ticker}
                    )
                
                # 統一列名格式（小寫）
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                
                # 確保必要的列存在
                required_cols = {'open', 'high', 'low', 'close', 'volume'}
                missing = required_cols - set(df.columns)
                if missing:
                    raise DataQualityError(
                        f"{ticker} 數據缺少列: {missing}",
                        {'ticker': ticker, 'missing': list(missing)}
                    )
                
                # 添加ticker標記
                df['ticker'] = ticker
                
                logger.info(f"已獲取 {ticker}: {start_date}~{end_date}, 共{len(df)}條記錄")
                return df
                
        except Exception as e:
            if isinstance(e, (DataSourceError, DataQualityError)):
                raise
            raise DataSourceError(
                f"Yahoo獲取{ticker}數據時出錯: {str(e)}",
                {'ticker': ticker, 'error_type': type(e).__name__}
            )
    
    def fetch_intraday_bars(
        self,
        ticker: str,
        interval: str = '5min',
        days_back: int = 5
    ) -> pd.DataFrame:
        """
        獲取日內分時數據
        
        注意：Yahoo Finance免費API的日內數據限制：
        - 1min數據: 最多7天
        - 5min/15min/30min: 最多60天
        - 1hour: 最多730天
        
        參數:
            ticker: 股票代碼
            interval: K線周期
            days_back: 回溯天數
        
        返回:
            日內分時DataFrame
        """
        self._rate_limit()
        
        # Yahoo Finance支持的日內周期映射
        valid_intervals = {'1m': '1m', '5m': '5m', '15m': '15m', 
                          '30m': '30m', '1h': '60m', '60m': '60m'}
        
        yf_interval = valid_intervals.get(interval, '5m')
        
        try:
            stock = self.yf.Ticker(ticker)
            df = stock.history(period=f'{days_back}d', interval=yf_interval)
            
            if df.empty:
                logger.warning(f"{ticker} 無 {days_back}d 日內數據")
                return pd.DataFrame()
            
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            return df
            
        except Exception as e:
            logger.error(f"獲取 {ticker} 日內數據失敗: {e}")
            return pd.DataFrame()
    
    def fetch_reference_data(self, ticker: str) -> Dict[str, Any]:
        """
        獲取股票基本面參考數據
        
        返回欄位：
        - name: 公司名稱
        - sector: 行業分類
        - industry: 細分行業
        - market_cap: 總市值
        - shares_outstanding: 流通股數
        - beta: Beta係數
        - pe_ratio: 市盈率
        - dividend_yield: 股息率
        - exchange: 上市交易所
        
        參數:
            ticker: 股票代碼
        
        返回:
            基本面信息字典
        """
        self._rate_limit()
        
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info
            
            ref_data = {
                'ticker': ticker,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'beta': info.get('beta'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'exchange': info.get('exchange', ''),
                'currency': info.get('currency', 'USD'),
                'short_ratio': info.get('shortRatio'),  # 做空比例
                'short_pct_float': info.get('shortPercentOfFloat'),
            }
            
            return ref_data
            
        except Exception as e:
            logger.warning(f"獲取 {ticker} 參考數據失敗: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def fetch_dividends_and_splits(
        self, ticker: str
    ) -> Tuple[pd.Series, pd.Series]:
        """
        獲取分紅和拆股歷史
        
        這對回測的復權處理至關重要：
        - 分紅(dividends): 現金分紅會直接降低股價
        - 拆股(stock splits): 會成倍改變股價和持倉數量
        
        參數:
            ticker: 股票代碼
        
        返回:
            (dividends_series, splits_series)
        """
        self._rate_limit()
        
        try:
            stock = self.yf.Ticker(ticker)
            dividends = stock.dividends
            splits = stock.splits
            return dividends, splits
        except Exception as e:
            logger.warning(f"獲取 {ticker} 分紅/拆股數據失敗: {e}")
            return pd.Series(), pd.Series()


# ============================================================================
# Polygon.io 數據獲取器（付費API）
# ============================================================================
class PolygonFetcher(DataFetcher):
    """
    基於Polygon.io的數據獲取器
    
    Polygon.io提供高質量的實時和歷史美股數據：
    - 實時Level-1/Level-2行情（WebSocket）
    - 歷史日線和分時數據（REST API）
    - 參考數據、財報、分紅等信息
    
    需要付費API密鑰，免費版有嚴格限制。
    官網: https://polygon.io
    """
    
    SOURCE_NAME = "PolygonIO"
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        if not config.polygon_api_key:
            raise ConfigurationError("Polygon.io API密鑰未設置")
        self._api_key = config.polygon_api_key
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        向Polygon API發送GET請求
        
        參數:
            endpoint: API端點路徑（如 '/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-12-31'）
            params: 額外的查詢參數
        
        返回:
            JSON響應字典
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        if params is None:
            params = {}
        params['apiKey'] = self._api_key
        
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get('status') == 'ERROR':
                raise DataSourceError(
                    f"Polygon API返回錯誤: {data.get('error', '未知錯誤')}",
                    {'endpoint': endpoint}
                )
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise DataSourceError(
                f"Polygon API請求失敗: {str(e)}",
                {'endpoint': endpoint}
            )
    
    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        通過Polygon.io獲取日線OHLCV數據
        
        Polygon使用 adjusted=true 參數返回復權數據。
        復權方式為 forward split adjustment（前向分股復權）。
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': str(auto_adjust).lower(),
            'sort': 'asc',
            'limit': 50000,  # 單次最多50000條
        }
        
        data = self._make_request(endpoint, params)
        
        results = data.get('results', [])
        if not results:
            raise DataQualityError(
                f"Polygon未返回{ticker}在{start_date}~{end_date}的數據",
                {'ticker': ticker}
            )
        
        # 轉換為DataFrame
        df = pd.DataFrame(results)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            't': 'date', 'o': 'open', 'h': 'high',
            'l': 'low', 'c': 'close', 'v': 'volume',
            'n': 'transactions', 'vw': 'vwap'
        })
        df.set_index('date', inplace=True)
        df['ticker'] = ticker
        
        return df
    
    def fetch_intraday_bars(
        self,
        ticker: str,
        interval: str = '5min',
        days_back: int = 5
    ) -> pd.DataFrame:
        """獲取日內分時數據"""
        interval_map = {
            '1min': '1/minute', '5min': '5/minute',
            '15min': '15/minute', '30min': '30/minute',
            '1hour': '1/hour'
        }
        poly_interval = interval_map.get(interval, f'{interval}/minute')
        
        end = datetime.now()
        start = end - pd.Timedelta(days=days_back)
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{poly_interval}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        
        data = self._make_request(endpoint, {'adjusted': 'true', 'limit': 50000})
        
        results = data.get('results', [])
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            't': 'date', 'o': 'open', 'h': 'high',
            'l': 'low', 'c': 'close', 'v': 'volume'
        })
        df.set_index('date', inplace=True)
        return df
    
    def fetch_reference_data(self, ticker: str) -> Dict[str, Any]:
        """獲取股票參考數據"""
        endpoint = f"/v3/reference/tickers/{ticker}"
        data = self._make_request(endpoint)
        result = data.get('results', {})
        
        return {
            'ticker': ticker,
            'name': result.get('name', ''),
            'market_cap': result.get('market_cap'),
            'sector': result.get('sic_description', ''),
            'exchange': result.get('primary_exchange', ''),
            'currency': result.get('currency_name', 'USD'),
            'shares_outstanding': result.get('weighted_shares_outstanding'),
        }


# ============================================================================
# 工廠函數：根據配置創建數據獲取器
# ============================================================================
def create_fetcher(config: DataSourceConfig) -> DataFetcher:
    """
    數據獲取器工廠函數
    
    根據配置選擇數據源，優先級：
    1. 如果配置了Polygon API密鑰且啟用了Polygon → PolygonFetcher
    2. 否則 → YahooFetcher（免費）
    
    參數:
        config: 數據源配置
    
    返回:
        對應的DataFetcher實現
    """
    if config.polygon_enabled and config.polygon_api_key:
        logger.info("使用 Polygon.io 作為數據源")
        return PolygonFetcher(config)
    
    logger.info("使用 Yahoo Finance 作為數據源")
    return YahooFetcher(config)
