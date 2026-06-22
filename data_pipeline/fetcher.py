"""
美股量化交易系统 - 数据获取模块

提供统一的数据获取接口，支持多种数据源：
- Yahoo Finance: 免费历史数据（爬虫/API混合使用）
- Polygon.io: 付费实时+历史行情数据
- Interactive Brokers: 券商提供的实时行情数据

设计模式：策略模式（Strategy Pattern）
- DataFetcher 为抽象基类
- 各数据源实现为具体策略类
- 通过工厂函数 create_fetcher() 根据配置选择策略

数据获取优先级：
1. 历史数据: Yahoo Finance（免费、覆盖全面）→ Polygon（付费、更精确）
2. 实时数据: IBKR（实盘）→ Polygon（模拟/回测）
3. 参考数据: Polygon → Yahoo Finance
"""

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
# 抽象数据获取器基类
# ============================================================================
class DataFetcher(ABC):
    """
    数据获取器抽象基类
    
    所有数据源实现必须继承此类并实现其抽象方法。
    子类只需关注数据获取逻辑，重试、速率限制等通用逻辑由基类处理。
    """
    
    # 子类可覆盖的元数据
    SOURCE_NAME: str = "abstract"
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0  # 秒
    RATE_LIMIT_CALLS: int = 5     # 每秒最大请求数
    RATE_LIMIT_PERIOD: float = 1.0  # 限速窗口（秒）
    
    def __init__(self, config: DataSourceConfig):
        """
        参数:
            config: 数据源配置对象
        """
        self.config = config
        self._last_call_times: List[float] = []  # 用于速率限制的调用时间戳队列
        self._session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        创建带重试机制的HTTP会话
        
        使用指数退避重试策略：
        - 第1次重试: 等待 1s
        - 第2次重试: 等待 2s
        - 第3次重试: 等待 4s
        总共最多重试3次
        
        返回:
            配置好的requests.Session对象
        """
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=1.0,         # 指数退避因子
            status_forcelist=[429, 500, 502, 503, 504],  # 这些状态码触发重试
            allowed_methods=["GET"]      # 仅重试GET请求（幂等安全）
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,     # 连接池大小
            pool_maxsize=20          # 最大连接数
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # 设置User-Agent，避免被某些API拒绝
        session.headers.update({
            'User-Agent': 'QuantTradingSystem/1.0 (Educational Purpose)'
        })
        
        return session
    
    def _rate_limit(self) -> None:
        """
        速率限制：确保不超过API调用频率限制
        
        使用滑动窗口算法，如果过去 RATE_LIMIT_PERIOD 秒内的调用次数
        达到 RATE_LIMIT_CALLS，则等待到下一个窗口。
        
        时间复杂度: O(k)，k为窗口内调用次数（通常很小）
        空间复杂度: O(k)
        """
        now = time.monotonic()
        window_start = now - self.RATE_LIMIT_PERIOD
        
        # 清理过期的调用记录（滑动窗口）
        self._last_call_times = [t for t in self._last_call_times if t > window_start]
        
        if len(self._last_call_times) >= self.RATE_LIMIT_CALLS:
            # 需要等待到最早的调用过期
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
        获取股票日线数据（OHLCV）
        
        参数:
            ticker: 股票代码 (如 'AAPL')
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            auto_adjust: 是否自动复权
        
        返回:
            包含以下列的DataFrame:
            - date (索引): 日期
            - open, high, low, close: OHLC价格
            - volume: 成交量
            - adj_close: 复权收盘价（如适用）
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
        获取日内分时数据
        
        参数:
            ticker: 股票代码
            interval: K线周期 ('1min', '5min', '15min', '30min', '1hour')
            days_back: 回溯天数
        
        返回:
            日内分时OHLCV DataFrame
        """
        pass
    
    @abstractmethod
    def fetch_reference_data(self, ticker: str) -> Dict[str, Any]:
        """
        获取股票参考数据（基本面信息）
        
        参数:
            ticker: 股票代码
        
        返回:
            包含公司名称、市值、行业、流通股数等信息的字典
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
        批量获取多只股票的日线数据（并发请求）
        
        使用ThreadPoolExecutor实现并发IO，适用于网络IO密集型任务。
        注意：需遵守API速率限制，不同数据源的速率限制不同。
        
        时间复杂度: O(n*m/p)，n=股票数，m=单股数据量，p=并发数
        空间复杂度: O(n*m)
        
        参数:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            max_workers: 最大并发数（默认使用系统配置）
        
        返回:
            字典: {ticker: DataFrame}
        """
        if max_workers is None:
            max_workers = system_config.max_workers
        
        results: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_ticker = {
                executor.submit(
                    self.fetch_daily_bars, ticker, start_date, end_date
                ): ticker
                for ticker in tickers
            }
            
            # 收集结果
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[ticker] = df
                        logger.debug(f"已获取 {ticker} 数据: {len(df)} 行")
                    else:
                        failed.append(ticker)
                except Exception as e:
                    logger.warning(f"获取 {ticker} 数据失败: {e}")
                    failed.append(ticker)
        
        if failed:
            logger.warning(f"以下股票数据获取失败 ({len(failed)}/{len(tickers)}): {failed}")
        
        logger.info(f"批量数据获取完成: 成功 {len(results)}/{len(tickers)}")
        return results
    
    def __del__(self):
        """析构时关闭HTTP会话"""
        if hasattr(self, '_session'):
            self._session.close()


# ============================================================================
# Yahoo Finance 数据获取器
# ============================================================================
class YahooFetcher(DataFetcher):
    """
    基于Yahoo Finance的数据获取器
    
    优点：
    - 免费，无需API密钥
    - 覆盖美股历史数据全面（1970年至今）
    - 自动处理分股和分红复权
    
    限制：
    - 速率限制严格（建议每秒不超过2个请求）
    - 部分已下市股票数据可能不完整
    - 日内数据仅提供最近30天
    
    实现方式：
    使用 yfinance 库，该库内部解析Yahoo Finance的公开API端点。
    """
    
    SOURCE_NAME = "YahooFinance"
    RATE_LIMIT_CALLS = 2  # Yahoo Finance速率限制较严格
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        # 延迟导入 yfinance，避免非必要依赖
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ConfigurationError(
                "yfinance 未安装，请执行: pip install yfinance"
            )
    
    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        通过Yahoo Finance获取日线OHLCV数据
        
        Yahoo Finance自动提供分股和分红复权数据，
        auto_adjust=True时返回的close即为adj_close。
        
        参数:
            ticker: 股票代码
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            auto_adjust: 是否使用复权价格
        
        返回:
            OHLCV DataFrame，索引为日期
        """
        if not validate_ticker(ticker):
            raise DataSourceError(f"无效的股票代码: {ticker}", {'ticker': ticker})
        
        self._rate_limit()
        
        try:
            with Timer(f"Yahoo获取{ticker}日线数据"):
                stock = self.yf.Ticker(ticker)
                
                # 获取历史数据
                # auto_adjust=False 返回未复权数据，含dividends和stock splits列
                # auto_adjust=True 返回复权后的OHLC数据
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    actions=True  # 包含分红和分股信息
                )
                
                if df.empty:
                    raise DataQualityError(
                        f"{ticker} 在 {start_date}~{end_date} 期间无数据",
                        {'ticker': ticker}
                    )
                
                # 统一列名格式（小写）
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                
                # 确保必要的列存在
                required_cols = {'open', 'high', 'low', 'close', 'volume'}
                missing = required_cols - set(df.columns)
                if missing:
                    raise DataQualityError(
                        f"{ticker} 数据缺少列: {missing}",
                        {'ticker': ticker, 'missing': list(missing)}
                    )
                
                # 添加ticker标记
                df['ticker'] = ticker
                
                logger.info(f"已获取 {ticker}: {start_date}~{end_date}, 共{len(df)}条记录")
                return df
                
        except Exception as e:
            if isinstance(e, (DataSourceError, DataQualityError)):
                raise
            raise DataSourceError(
                f"Yahoo获取{ticker}数据时出错: {str(e)}",
                {'ticker': ticker, 'error_type': type(e).__name__}
            )
    
    def fetch_intraday_bars(
        self,
        ticker: str,
        interval: str = '5min',
        days_back: int = 5
    ) -> pd.DataFrame:
        """
        获取日内分时数据
        
        注意：Yahoo Finance免费API的日内数据限制：
        - 1min数据: 最多7天
        - 5min/15min/30min: 最多60天
        - 1hour: 最多730天
        
        参数:
            ticker: 股票代码
            interval: K线周期
            days_back: 回溯天数
        
        返回:
            日内分时DataFrame
        """
        self._rate_limit()
        
        # Yahoo Finance支持的日内周期映射
        valid_intervals = {'1m': '1m', '5m': '5m', '15m': '15m', 
                          '30m': '30m', '1h': '60m', '60m': '60m'}
        
        yf_interval = valid_intervals.get(interval, '5m')
        
        try:
            stock = self.yf.Ticker(ticker)
            df = stock.history(period=f'{days_back}d', interval=yf_interval)
            
            if df.empty:
                logger.warning(f"{ticker} 无 {days_back}d 日内数据")
                return pd.DataFrame()
            
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            return df
            
        except Exception as e:
            logger.error(f"获取 {ticker} 日内数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_reference_data(self, ticker: str) -> Dict[str, Any]:
        """
        获取股票基本面参考数据
        
        返回字段：
        - name: 公司名称
        - sector: 行业分类
        - industry: 细分行业
        - market_cap: 总市值
        - shares_outstanding: 流通股数
        - beta: Beta系数
        - pe_ratio: 市盈率
        - dividend_yield: 股息率
        - exchange: 上市交易所
        
        参数:
            ticker: 股票代码
        
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
            logger.warning(f"获取 {ticker} 参考数据失败: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def fetch_dividends_and_splits(
        self, ticker: str
    ) -> Tuple[pd.Series, pd.Series]:
        """
        获取分红和拆股历史
        
        这对回测的复权处理至关重要：
        - 分红(dividends): 现金分红会直接降低股价
        - 拆股(stock splits): 会成倍改变股价和持仓数量
        
        参数:
            ticker: 股票代码
        
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
            logger.warning(f"获取 {ticker} 分红/拆股数据失败: {e}")
            return pd.Series(), pd.Series()


# ============================================================================
# Polygon.io 数据获取器（付费API）
# ============================================================================
class PolygonFetcher(DataFetcher):
    """
    基于Polygon.io的数据获取器
    
    Polygon.io提供高质量的实时和历史美股数据：
    - 实时Level-1/Level-2行情（WebSocket）
    - 历史日线和分时数据（REST API）
    - 参考数据、财报、分红等信息
    
    需要付费API密钥，免费版有严格限制。
    官网: https://polygon.io
    """
    
    SOURCE_NAME = "PolygonIO"
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        if not config.polygon_api_key:
            raise ConfigurationError("Polygon.io API密钥未设置")
        self._api_key = config.polygon_api_key
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        向Polygon API发送GET请求
        
        参数:
            endpoint: API端点路径（如 '/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-12-31'）
            params: 额外的查询参数
        
        返回:
            JSON响应字典
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
                    f"Polygon API返回错误: {data.get('error', '未知错误')}",
                    {'endpoint': endpoint}
                )
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise DataSourceError(
                f"Polygon API请求失败: {str(e)}",
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
        通过Polygon.io获取日线OHLCV数据
        
        Polygon使用 adjusted=true 参数返回复权数据。
        复权方式为 forward split adjustment（前向分股复权）。
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': str(auto_adjust).lower(),
            'sort': 'asc',
            'limit': 50000,  # 单次最多50000条
        }
        
        data = self._make_request(endpoint, params)
        
        results = data.get('results', [])
        if not results:
            raise DataQualityError(
                f"Polygon未返回{ticker}在{start_date}~{end_date}的数据",
                {'ticker': ticker}
            )
        
        # 转换为DataFrame
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
        """获取日内分时数据"""
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
        """获取股票参考数据"""
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
# 工厂函数：根据配置创建数据获取器
# ============================================================================
def create_fetcher(config: DataSourceConfig) -> DataFetcher:
    """
    数据获取器工厂函数
    
    根据配置选择数据源，优先级：
    1. 如果配置了Polygon API密钥且启用了Polygon → PolygonFetcher
    2. 否则 → YahooFetcher（免费）
    
    参数:
        config: 数据源配置
    
    返回:
        对应的DataFetcher实现
    """
    if config.polygon_enabled and config.polygon_api_key:
        logger.info("使用 Polygon.io 作为数据源")
        return PolygonFetcher(config)
    
    logger.info("使用 Yahoo Finance 作为数据源")
    return YahooFetcher(config)
