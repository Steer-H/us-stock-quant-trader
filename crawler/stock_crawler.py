"""Stock data crawler for Yahoo Finance."""

import logging
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

from data_pipeline.fetcher import YahooFetcher, create_fetcher
from data_pipeline.cleaner import DataCleaner
from data_pipeline.storage import RawDataStore, ParquetStorage
from config.settings import (
    DataSourceConfig, system_config, RAW_DATA_DIR
)
from utils.helpers import Timer, validate_ticker, is_trading_day
from utils.exceptions import DataSourceError, DataQualityError, ConfigurationError

logger = logging.getLogger(__name__)


# ============================================================================
# 爬蟲結果數據結構
# ============================================================================
@dataclass
class CrawlResult:
    """
    單次爬取任務的結果
    
    屬性:
        ticker: 股票代碼
        status: 'success', 'failed', 'skipped'(已存在且無需更新)
        rows: 獲取的數據行數
        start_date: 數據開始日期
        end_date: 數據結束日期
        error: 錯誤信息（僅失敗時）
        duration: 爬取耗時（秒）
    """
    ticker: str
    status: str
    rows: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class CrawlSummary:
    """
    批量爬取的匯總統計
    
    屬性:
        total_tickers: 總共目標股票數
        success_count: 成功獲取的數量
        failed_count: 失敗的數量
        skipped_count: 跳過的數量
        total_rows: 總共獲取的數據行數
        total_duration: 總耗時（秒）
        results: 每隻股票的詳細結果列表
    """
    total_tickers: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_rows: int = 0
    total_duration: float = 0.0
    results: List[CrawlResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_tickers == 0:
            return 0.0
        return self.success_count / self.total_tickers
    
    def to_dict(self) -> Dict:
        """轉換為字典格式（便於日誌和報告）"""
        return {
            'total_tickers': self.total_tickers,
            'success_count': self.success_count,
            'failed_count': self.failed_count,
            'skipped_count': self.skipped_count,
            'total_rows': self.total_rows,
            'total_duration_min': round(self.total_duration / 60, 1),
            'success_rate': f"{self.success_rate:.1%}",
            'failed_tickers': [r.ticker for r in self.results if r.status == 'failed'],
        }


# ============================================================================
# 通用美股歷史數據爬蟲
# ============================================================================
class StockCrawler:
    """
    美股歷史數據爬蟲基類
    
    提供通用的爬取邏輯：
    - 單股爬取
    - 批量爬取（並發）
    - 增量更新（僅獲取缺失的日期）
    - 斷點續傳（跳過已有的數據）
    
    子類需要實現 _get_ticker_list() 方法提供股票列表。
    """
    
    def __init__(self, config: DataSourceConfig = None):
        """
        參數:
            config: 數據源配置（None則使用默認配置）
        """
        self.config = config or DataSourceConfig()
        self.fetcher = create_fetcher(self.config)
        self.cleaner = DataCleaner(self.config)
        self.raw_store = RawDataStore()
        self.processed_store = ParquetStorage()
        self._results: List[CrawlResult] = []
    
    def crawl_single(self, ticker: str, 
                     start_date: str = '2010-01-01',
                     end_date: str = '2025-12-31',
                     force_update: bool = False) -> CrawlResult:
        """
        爬取單只股票的完整歷史數據
        
        流程：
        1. 檢查本地是否已有數據（增量更新）
        2. 從Yahoo Finance獲取日線OHLCV數據
        3. 數據清洗（去異常值、填缺失值）
        4. 計算技術指標
        5. 分別保存原始數據和加工數據
        
        時間複雜度: O(n)，n為數據行數
        空間複雜度: O(n)
        
        參數:
            ticker: 股票代碼
            start_date: 數據起始日期
            end_date: 數據結束日期
            force_update: 是否強制更新（忽略已有數據）
        
        返回:
            CrawlResult對象，包含爬取狀態和統計信息
        """
        start = time.perf_counter()
        
        try:
            # 1. 檢查本地緩存
            if not force_update and self.processed_store.exists(f"{ticker}_features"):
                logger.info(f"{ticker} 特徵數據已存在，跳過")
                return CrawlResult(
                    ticker=ticker, status='skipped',
                    duration=time.perf_counter() - start
                )
            
            # 2. 獲取原始數據
            logger.info(f"正在爬取 {ticker} 數據: {start_date} ~ {end_date}")
            
            df = self.fetcher.fetch_daily_bars(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=True
            )
            
            if df is None or df.empty:
                return CrawlResult(
                    ticker=ticker, status='failed',
                    error='無數據返回',
                    duration=time.perf_counter() - start
                )
            
            # 修復時區：Yahoo Finance返回tz-aware，pandas操作需要tz-naive
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 3. 數據清洗
            # 注意：使用鏈式調用的方式，每一步都返回self
            self.cleaner.reset()  # 重置審計日誌
            
            # 驗證結構
            self.cleaner.validate_structure(df, ticker)
            
            # 處理異常值
            self.cleaner.remove_outliers(df, ticker, 'close')
            
            # 填充缺失值
            self.cleaner.fill_missing(df, ticker)
            
            # 檢測倖存者偏差
            self.cleaner.detect_survivorship_bias(df, ticker, end_date)
            
            # 4. 保存原始數據（不可變存檔）
            self.raw_store.save_raw(df.copy(), ticker)
            
            # 5. 計算技術指標並保存加工數據
            from data_pipeline.indicators import TechnicalIndicators
            features_df = TechnicalIndicators.add_all_indicators(df)
            self.processed_store.save(features_df, f"{ticker}_features")
            
            # 6. 構建結果
            elapsed = time.perf_counter() - start
            result = CrawlResult(
                ticker=ticker,
                status='success',
                rows=len(df),
                start_date=str(df.index.min().date()) if not df.empty else None,
                end_date=str(df.index.max().date()) if not df.empty else None,
                duration=elapsed
            )
            
            logger.info(
                f"✓ {ticker}: {len(df)} 行數據, "
                f"{result.start_date} ~ {result.end_date}, "
                f"耗時 {elapsed:.1f}s"
            )
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"✗ {ticker} 爬取失敗: {type(e).__name__}: {e}")
            return CrawlResult(
                ticker=ticker, status='failed',
                error=f"{type(e).__name__}: {str(e)}",
                duration=elapsed
            )
    
    def crawl_batch(self, tickers: List[str],
                    start_date: str = '2010-01-01',
                    end_date: str = '2025-12-31',
                    max_workers: Optional[int] = None,
                    random_delay: bool = True) -> CrawlSummary:
        """
        批量並發爬取多隻股票數據
        
        使用ThreadPoolExecutor實現並發爬取，適用於IO密集型任務。
        每個線程獨立爬取一隻股票，互不幹擾。
        
        注意：為防止觸發Yahoo Finance的速率限制，
        會在每次請求間添加隨機延遲（0.5-2.0秒）。
        
        時間複雜度: O(n*m/p)，n=股票數，m=平均每隻數據量，p=並發數
        空間複雜度: O(n*m)
        
        參數:
            tickers: 股票代碼列表
            start_date: 數據起始日期
            end_date: 數據結束日期
            max_workers: 最大並發線程數（默認使用系統配置值）
            random_delay: 是否在請求間添加隨機延遲
        
        返回:
            CrawlSummary匯總統計
        """
        if max_workers is None:
            # 對於Yahoo Finance，建議不超過3個並發
            max_workers = min(system_config.max_workers, 3)
        
        logger.info(
            f"開始批量爬取: {len(tickers)} 只股票, "
            f"{start_date} ~ {end_date}, "
            f"並發數: {max_workers}"
        )
        
        overall_start = time.perf_counter()
        self._results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_ticker = {}
            for i, ticker in enumerate(tickers):
                # 錯開請求時間，避免同時發出大量請求觸發速率限制
                if i > 0 and random_delay:
                    delay = random.uniform(0.3, 1.5)
                    time.sleep(delay)
                
                future = executor.submit(
                    self.crawl_single, ticker, start_date, end_date
                )
                future_to_ticker[future] = ticker
            
            # 收集結果
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    self._results.append(result)
                except Exception as e:
                    logger.error(f"{ticker} 線程異常: {e}")
                    self._results.append(CrawlResult(
                        ticker=ticker, status='failed',
                        error=f"Thread異常: {str(e)}"
                    ))
        
        # 構建匯總統計
        overall_duration = time.perf_counter() - overall_start
        
        summary = CrawlSummary(
            total_tickers=len(tickers),
            success_count=sum(1 for r in self._results if r.status == 'success'),
            failed_count=sum(1 for r in self._results if r.status == 'failed'),
            skipped_count=sum(1 for r in self._results if r.status == 'skipped'),
            total_rows=sum(r.rows for r in self._results if r.status == 'success'),
            total_duration=overall_duration,
            results=self._results
        )
        
        logger.info(
            f"批量爬取完成: "
            f"成功 {summary.success_count}, "
            f"失敗 {summary.failed_count}, "
            f"跳過 {summary.skipped_count}, "
            f"總計 {summary.total_rows} 行數據, "
            f"總耗時 {overall_duration/60:.1f}min"
        )
        
        return summary
    
    def incremental_update(self, tickers: List[str]) -> CrawlSummary:
        """
        增量更新已有數據（僅獲取最新的交易日數據）
        
        適用於日常更新場景，避免重新爬取全部歷史數據。
        
        參數:
            tickers: 需要更新的股票列表
        
        返回:
            CrawlSummary匯總統計
        """
        latest_dates = {}
        
        for ticker in tickers:
            try:
                if self.processed_store.exists(f"{ticker}_features"):
                    df = self.processed_store.load(f"{ticker}_features")
                    latest_dates[ticker] = str(df.index.max().date())
                else:
                    logger.info(f"{ticker} 無歷史數據，將全量爬取")
            except Exception:
                logger.debug(f"Non-critical error in stock_crawler.py: {e}", exc_info=True)
        
        # 對每隻股票執行增量爬取
        results = []
        for ticker in tickers:
            if ticker in latest_dates:
                last_date = latest_dates[ticker]
                today = date.today().isoformat()
                # 如果最後數據日期和今天不同（考慮了周末）
                if last_date < today:
                    result = self.crawl_single(
                        ticker, start_date=last_date, end_date=today
                    )
                    results.append(result)
                else:
                    results.append(CrawlResult(ticker=ticker, status='skipped'))
            else:
                result = self.crawl_single(ticker)
                results.append(result)
        
        # 構建匯總（簡化版）
        return CrawlSummary(
            total_tickers=len(tickers),
            success_count=sum(1 for r in results if r.status == 'success'),
            failed_count=sum(1 for r in results if r.status == 'failed'),
            skipped_count=sum(1 for r in results if r.status == 'skipped'),
            total_rows=sum(r.rows for r in results if r.status == 'success'),
            results=results
        )
    
    def get_results(self) -> List[CrawlResult]:
        """獲取最近一次爬取的結果列表"""
        return self._results.copy()
    
    def get_failed_tickers(self) -> List[str]:
        """獲取爬取失敗的股票列表（用於重試）"""
        return [r.ticker for r in self._results if r.status == 'failed']


# ============================================================================
# S&P 500 成分股爬蟲
# ============================================================================
class SP500Crawler(StockCrawler):
    """
    S&P 500 成分股專用爬蟲
    
    從Wikipedia獲取S&P 500當前成分股列表，
    然後批量爬取歷史數據。
    
    S&P 500成分股列表來源：
    - Wikipedia: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    - 該頁面每日更新，包含股票代碼、公司名稱、GICS行業分類
    
    注意事項：
    - S&P 500成分股定期調整（季度rebalance），歷史成分股可能與當前不同
    - 已移除的股票不在此列表中（倖存者偏差問題）
    """
    
    # Wikipedia S&P 500 成分股頁面URL
    SP500_WIKI_URL = (
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    )
    
    def get_sp500_tickers(self, force_refresh: bool = False) -> List[str]:
        """
        從Wikipedia獲取S&P 500當前成分股列表
        
        使用緩存機制：首次獲取後緩存到本地CSV文件，
        避免每次運行都請求Wikipedia。
        
        Wikipedia頁面結構：
        - 第一個表格（class='wikitable sortable'）包含成分股信息
        - 列包括：Symbol, Security, GICS Sector, GICS Sub-Industry, etc.
        
        時間複雜度: O(n)，n為成分股數量（約500）
        空間複雜度: O(n)
        
        參數:
            force_refresh: 是否強制刷新（忽略緩存）
        
        返回:
            股票代碼列表（大寫，已排序）
        """
        cache_path = RAW_DATA_DIR / 'sp500_tickers.csv'
        
        # 1. 嘗試從緩存加載
        if not force_refresh and cache_path.exists():
            # 檢查緩存是否在24小時內
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < 86400:  # 24小時 = 86400秒
                tickers = pd.read_csv(cache_path)['Symbol'].tolist()
                logger.info(f"從緩存加載S&P 500成分股: {len(tickers)} 只")
                return tickers
        
        # 2. 從Wikipedia抓取
        logger.info("從Wikipedia獲取S&P 500成分股列表...")
        
        try:
            # 發送HTTP GET請求
            resp = requests.get(
                self.SP500_WIKI_URL,
                headers={'User-Agent': 'QuantResearch/1.0 (Educational)'},
                timeout=30
            )
            resp.raise_for_status()
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 定位成分股表格（頁面第一個wikitable）
            table = soup.find('table', {'class': 'wikitable sortable'})
            if not table:
                raise DataSourceError("未找到S&P 500成分股表格")
            
            # 解析表格數據
            rows = table.find_all('tr')[1:]  # 跳過表頭
            tickers = []
            
            for row in rows:
                cols = row.find_all('td')
                if cols:
                    # 第一列通常是Symbol
                    symbol = cols[0].text.strip()
                    # 過濾有效的美股ticker（1-5個大寫字母）
                    # 注意：部分股票代碼含點號（如BRK.B），需要特殊處理
                    if symbol and len(symbol) <= 6:
                        # 將點號替換為短橫線（Yahoo Finance使用短橫線）
                        symbol = symbol.replace('.', '-')
                        if validate_ticker(symbol.replace('-', '')):
                            tickers.append(symbol)
            
            if not tickers:
                raise DataSourceError("解析S&P 500列表為空")
            
            # 3. 保存到緩存
            pd.DataFrame({'Symbol': tickers}).to_csv(cache_path, index=False)
            
            logger.info(f"S&P 500成分股獲取完成: {len(tickers)} 只")
            return sorted(tickers)
            
        except requests.RequestException as e:
            logger.error(f"Wikipedia請求失敗: {e}")
            
            # 降級：如果緩存存在（即使過期），也使用緩存
            if cache_path.exists():
                tickers = pd.read_csv(cache_path)['Symbol'].tolist()
                logger.warning(f"降級使用過期緩存: {len(tickers)} 只")
                return tickers
            
            raise DataSourceError(f"無法獲取S&P 500成分股列表: {e}")
    
    def get_sp500_with_sectors(self) -> pd.DataFrame:
        """
        獲取S&P 500成分股及其行業分類
        
        返回包含以下列的DataFrame:
        - Symbol: 股票代碼
        - Security: 公司名稱
        - GICS_Sector: GICS行業分類（如 Information Technology）
        - GICS_Sub_Industry: GICS子行業分類
        - Headquarters: 總部所在地
        - Date_Added: 加入S&P 500的日期
        - CIK: SEC中央索引鍵
        
        返回:
            成分股信息DataFrame
        """
        try:
            resp = requests.get(
                self.SP500_WIKI_URL,
                headers={'User-Agent': 'QuantResearch/1.0 (Educational)'},
                timeout=30
            )
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable sortable'})
            
            if not table:
                raise DataSourceError("未找到S&P 500成分股表格")
            
            # 使用pandas的read_html更可靠
            dfs = pd.read_html(str(table))
            df = dfs[0]
            
            # 標準化列名
            df.columns = [c.replace(' ', '_') for c in df.columns]
            
            # 處理Symbol中的點號
            if 'Symbol' in df.columns:
                df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
            
            logger.info(f"S&P 500成分股+行業分類獲取完成: {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"獲取S&P 500詳細信息失敗: {e}")
            return pd.DataFrame()
    
    def crawl_sp500_historical(
        self,
        start_date: str = '2010-01-01',
        end_date: str = '2025-12-31',
        max_workers: int = 3
    ) -> CrawlSummary:
        """
        爬取全部S&P 500成分股的歷史數據（2010-2025）
        
        這是主要的入口方法，一鍵完成：
        1. 獲取S&P 500成分股列表
        2. 批量並發爬取歷史日線數據
        3. 數據清洗+特徵工程
        4. 生成爬取報告
        
        參數:
            start_date: 數據起始日期
            end_date: 數據結束日期
            max_workers: 最大並發數
        
        返回:
            CrawlSummary匯總統計
        """
        # 1. 獲取股票列表
        with Timer("獲取S&P 500成分股列表"):
            tickers = self.get_sp500_tickers()
        
        if not tickers:
            raise DataSourceError("無法獲取S&P 500成分股列表")
        
        logger.info(
            f"準備爬取 {len(tickers)} 只S&P 500成分股的歷史數據 "
            f"({start_date} ~ {end_date})"
        )
        
        # 2. 批量爬取
        summary = self.crawl_batch(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers
        )
        
        # 3. 重試失敗的單只股票（最多重試1次）
        failed = self.get_failed_tickers()
        if failed:
            logger.info(f"重試 {len(failed)} 只失敗的股票...")
            time.sleep(5)  # 等待冷卻
            
            retry_results = []
            for ticker in failed:
                time.sleep(random.uniform(1.0, 3.0))  # 逐個重試，增加間隔
                result = self.crawl_single(ticker, start_date, end_date, force_update=True)
                retry_results.append(result)
            
            # 更新匯總統計
            retry_success = sum(1 for r in retry_results if r.status == 'success')
            summary.success_count += retry_success
            summary.failed_count -= retry_success
            summary.total_rows += sum(r.rows for r in retry_results if r.status == 'success')
            summary.results.extend(retry_results)
            
            logger.info(f"重試完成: 新增成功 {retry_success}/{len(failed)}")
        
        # 4. 列印最終報告
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: CrawlSummary) -> None:
        """列印爬取匯總報告"""
        print("\n" + "=" * 60)
        print("          S&P 500 歷史數據爬取報告")
        print("=" * 60)
        print(f"  目標股票數:     {summary.total_tickers}")
        print(f"  成功獲取:       {summary.success_count}")
        print(f"  失敗:           {summary.failed_count}")
        print(f"  跳過(已存在):    {summary.skipped_count}")
        print(f"  總數據行數:     {summary.total_rows:,}")
        print(f"  成功率:         {summary.success_rate:.1%}")
        print(f"  總耗時:         {summary.total_duration/60:.1f} 分鐘")
        
        if summary.failed_count > 0:
            failed_tickers = [r.ticker for r in summary.results if r.status == 'failed']
            print(f"\n  失敗股票 ({len(failed_tickers)}):")
            for i, ticker in enumerate(failed_tickers):
                if i < 20:  # 最多顯示20個
                    error = next(
                        (r.error for r in summary.results 
                         if r.ticker == ticker and r.error), '未知錯誤'
                    )
                    print(f"    - {ticker}: {error[:80]}")
                elif i == 20:
                    print(f"    ... 還有 {len(failed_tickers) - 20} 只")
                    break
        
        print("=" * 60 + "\n")


# ============================================================================
# 便捷函數：快速爬取S&P 500數據
# ============================================================================
def crawl_sp500_data(
    start_date: str = '2010-01-01',
    end_date: str = '2025-12-31',
    max_workers: int = 3
) -> CrawlSummary:
    """
    一鍵爬取S&P 500全部歷史數據
    
    這是對外暴露的便捷函數，隱藏了內部實現細節。
    
    使用示例:
        from crawler import crawl_sp500_data
        summary = crawl_sp500_data('2010-01-01', '2025-12-31')
    
    參數:
        start_date: 起始日期
        end_date: 結束日期
        max_workers: 並發數
    
    返回:
        CrawlSummary匯總統計
    """
    crawler = SP500Crawler()
    return crawler.crawl_sp500_historical(start_date, end_date, max_workers)
