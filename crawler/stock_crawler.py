"""
美股量化交易系统 - 美股历史数据爬虫模块

负责从公开数据源抓取美股历史交易数据（2010-2025），
为ML模型提供训练数据。

核心功能：
- 获取S&P 500成分股列表（含历史变更）
- 批量抓取日线OHLCV数据
- 自动处理网络异常、速率限制和重试
- 数据完整性校验（行数、日期连续性等）
- 支持断点续传（增量更新）

数据源：
- 主要: Yahoo Finance (免费，覆盖全面)
- 备用: Alpha Vantage (API密钥可选)
- 参考: Wikipedia (S&P 500成分股列表)

注意事项：
- 爬虫行为遵循 robots.txt 和 API 使用条款
- 速率限制：每秒不超过2个请求（Yahoo Finance限制）
- 仅用于个人学习和研究目的
"""

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
# 爬虫结果数据结构
# ============================================================================
@dataclass
class CrawlResult:
    """
    单次爬取任务的结果
    
    属性:
        ticker: 股票代码
        status: 'success', 'failed', 'skipped'(已存在且无需更新)
        rows: 获取的数据行数
        start_date: 数据开始日期
        end_date: 数据结束日期
        error: 错误信息（仅失败时）
        duration: 爬取耗时（秒）
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
    批量爬取的汇总统计
    
    属性:
        total_tickers: 总共目标股票数
        success_count: 成功获取的数量
        failed_count: 失败的数量
        skipped_count: 跳过的数量
        total_rows: 总共获取的数据行数
        total_duration: 总耗时（秒）
        results: 每只股票的详细结果列表
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
        """转换为字典格式（便于日志和报告）"""
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
# 通用美股历史数据爬虫
# ============================================================================
class StockCrawler:
    """
    美股历史数据爬虫基类
    
    提供通用的爬取逻辑：
    - 单股爬取
    - 批量爬取（并发）
    - 增量更新（仅获取缺失的日期）
    - 断点续传（跳过已有的数据）
    
    子类需要实现 _get_ticker_list() 方法提供股票列表。
    """
    
    def __init__(self, config: DataSourceConfig = None):
        """
        参数:
            config: 数据源配置（None则使用默认配置）
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
        爬取单只股票的完整历史数据
        
        流程：
        1. 检查本地是否已有数据（增量更新）
        2. 从Yahoo Finance获取日线OHLCV数据
        3. 数据清洗（去异常值、填缺失值）
        4. 计算技术指标
        5. 分别保存原始数据和加工数据
        
        时间复杂度: O(n)，n为数据行数
        空间复杂度: O(n)
        
        参数:
            ticker: 股票代码
            start_date: 数据起始日期
            end_date: 数据结束日期
            force_update: 是否强制更新（忽略已有数据）
        
        返回:
            CrawlResult对象，包含爬取状态和统计信息
        """
        start = time.perf_counter()
        
        try:
            # 1. 检查本地缓存
            if not force_update and self.processed_store.exists(f"{ticker}_features"):
                logger.info(f"{ticker} 特征数据已存在，跳过")
                return CrawlResult(
                    ticker=ticker, status='skipped',
                    duration=time.perf_counter() - start
                )
            
            # 2. 获取原始数据
            logger.info(f"正在爬取 {ticker} 数据: {start_date} ~ {end_date}")
            
            df = self.fetcher.fetch_daily_bars(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=True
            )
            
            if df is None or df.empty:
                return CrawlResult(
                    ticker=ticker, status='failed',
                    error='无数据返回',
                    duration=time.perf_counter() - start
                )
            
            # 修复时区：Yahoo Finance返回tz-aware，pandas操作需要tz-naive
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 3. 数据清洗
            # 注意：使用链式调用的方式，每一步都返回self
            self.cleaner.reset()  # 重置审计日志
            
            # 验证结构
            self.cleaner.validate_structure(df, ticker)
            
            # 处理异常值
            self.cleaner.remove_outliers(df, ticker, 'close')
            
            # 填充缺失值
            self.cleaner.fill_missing(df, ticker)
            
            # 检测幸存者偏差
            self.cleaner.detect_survivorship_bias(df, ticker, end_date)
            
            # 4. 保存原始数据（不可变存档）
            self.raw_store.save_raw(df.copy(), ticker)
            
            # 5. 计算技术指标并保存加工数据
            from data_pipeline.indicators import TechnicalIndicators
            features_df = TechnicalIndicators.add_all_indicators(df)
            self.processed_store.save(features_df, f"{ticker}_features")
            
            # 6. 构建结果
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
                f"✓ {ticker}: {len(df)} 行数据, "
                f"{result.start_date} ~ {result.end_date}, "
                f"耗时 {elapsed:.1f}s"
            )
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"✗ {ticker} 爬取失败: {type(e).__name__}: {e}")
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
        批量并发爬取多只股票数据
        
        使用ThreadPoolExecutor实现并发爬取，适用于IO密集型任务。
        每个线程独立爬取一只股票，互不干扰。
        
        注意：为防止触发Yahoo Finance的速率限制，
        会在每次请求间添加随机延迟（0.5-2.0秒）。
        
        时间复杂度: O(n*m/p)，n=股票数，m=平均每只数据量，p=并发数
        空间复杂度: O(n*m)
        
        参数:
            tickers: 股票代码列表
            start_date: 数据起始日期
            end_date: 数据结束日期
            max_workers: 最大并发线程数（默认使用系统配置值）
            random_delay: 是否在请求间添加随机延迟
        
        返回:
            CrawlSummary汇总统计
        """
        if max_workers is None:
            # 对于Yahoo Finance，建议不超过3个并发
            max_workers = min(system_config.max_workers, 3)
        
        logger.info(
            f"开始批量爬取: {len(tickers)} 只股票, "
            f"{start_date} ~ {end_date}, "
            f"并发数: {max_workers}"
        )
        
        overall_start = time.perf_counter()
        self._results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_ticker = {}
            for i, ticker in enumerate(tickers):
                # 错开请求时间，避免同时发出大量请求触发速率限制
                if i > 0 and random_delay:
                    delay = random.uniform(0.3, 1.5)
                    time.sleep(delay)
                
                future = executor.submit(
                    self.crawl_single, ticker, start_date, end_date
                )
                future_to_ticker[future] = ticker
            
            # 收集结果
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    self._results.append(result)
                except Exception as e:
                    logger.error(f"{ticker} 线程异常: {e}")
                    self._results.append(CrawlResult(
                        ticker=ticker, status='failed',
                        error=f"Thread异常: {str(e)}"
                    ))
        
        # 构建汇总统计
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
            f"失败 {summary.failed_count}, "
            f"跳过 {summary.skipped_count}, "
            f"总计 {summary.total_rows} 行数据, "
            f"总耗时 {overall_duration/60:.1f}min"
        )
        
        return summary
    
    def incremental_update(self, tickers: List[str]) -> CrawlSummary:
        """
        增量更新已有数据（仅获取最新的交易日数据）
        
        适用于日常更新场景，避免重新爬取全部历史数据。
        
        参数:
            tickers: 需要更新的股票列表
        
        返回:
            CrawlSummary汇总统计
        """
        latest_dates = {}
        
        for ticker in tickers:
            try:
                if self.processed_store.exists(f"{ticker}_features"):
                    df = self.processed_store.load(f"{ticker}_features")
                    latest_dates[ticker] = str(df.index.max().date())
                else:
                    logger.info(f"{ticker} 无历史数据，将全量爬取")
            except Exception:
                logger.debug(f"Non-critical error in stock_crawler.py: {e}", exc_info=True)
        
        # 对每只股票执行增量爬取
        results = []
        for ticker in tickers:
            if ticker in latest_dates:
                last_date = latest_dates[ticker]
                today = date.today().isoformat()
                # 如果最后数据日期和今天不同（考虑了周末）
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
        
        # 构建汇总（简化版）
        return CrawlSummary(
            total_tickers=len(tickers),
            success_count=sum(1 for r in results if r.status == 'success'),
            failed_count=sum(1 for r in results if r.status == 'failed'),
            skipped_count=sum(1 for r in results if r.status == 'skipped'),
            total_rows=sum(r.rows for r in results if r.status == 'success'),
            results=results
        )
    
    def get_results(self) -> List[CrawlResult]:
        """获取最近一次爬取的结果列表"""
        return self._results.copy()
    
    def get_failed_tickers(self) -> List[str]:
        """获取爬取失败的股票列表（用于重试）"""
        return [r.ticker for r in self._results if r.status == 'failed']


# ============================================================================
# S&P 500 成分股爬虫
# ============================================================================
class SP500Crawler(StockCrawler):
    """
    S&P 500 成分股专用爬虫
    
    从Wikipedia获取S&P 500当前成分股列表，
    然后批量爬取历史数据。
    
    S&P 500成分股列表来源：
    - Wikipedia: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    - 该页面每日更新，包含股票代码、公司名称、GICS行业分类
    
    注意事项：
    - S&P 500成分股定期调整（季度rebalance），历史成分股可能与当前不同
    - 已移除的股票不在此列表中（幸存者偏差问题）
    """
    
    # Wikipedia S&P 500 成分股页面URL
    SP500_WIKI_URL = (
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    )
    
    def get_sp500_tickers(self, force_refresh: bool = False) -> List[str]:
        """
        从Wikipedia获取S&P 500当前成分股列表
        
        使用缓存机制：首次获取后缓存到本地CSV文件，
        避免每次运行都请求Wikipedia。
        
        Wikipedia页面结构：
        - 第一个表格（class='wikitable sortable'）包含成分股信息
        - 列包括：Symbol, Security, GICS Sector, GICS Sub-Industry, etc.
        
        时间复杂度: O(n)，n为成分股数量（约500）
        空间复杂度: O(n)
        
        参数:
            force_refresh: 是否强制刷新（忽略缓存）
        
        返回:
            股票代码列表（大写，已排序）
        """
        cache_path = RAW_DATA_DIR / 'sp500_tickers.csv'
        
        # 1. 尝试从缓存加载
        if not force_refresh and cache_path.exists():
            # 检查缓存是否在24小时内
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < 86400:  # 24小时 = 86400秒
                tickers = pd.read_csv(cache_path)['Symbol'].tolist()
                logger.info(f"从缓存加载S&P 500成分股: {len(tickers)} 只")
                return tickers
        
        # 2. 从Wikipedia抓取
        logger.info("从Wikipedia获取S&P 500成分股列表...")
        
        try:
            # 发送HTTP GET请求
            resp = requests.get(
                self.SP500_WIKI_URL,
                headers={'User-Agent': 'QuantResearch/1.0 (Educational)'},
                timeout=30
            )
            resp.raise_for_status()
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 定位成分股表格（页面第一个wikitable）
            table = soup.find('table', {'class': 'wikitable sortable'})
            if not table:
                raise DataSourceError("未找到S&P 500成分股表格")
            
            # 解析表格数据
            rows = table.find_all('tr')[1:]  # 跳过表头
            tickers = []
            
            for row in rows:
                cols = row.find_all('td')
                if cols:
                    # 第一列通常是Symbol
                    symbol = cols[0].text.strip()
                    # 过滤有效的美股ticker（1-5个大写字母）
                    # 注意：部分股票代码含点号（如BRK.B），需要特殊处理
                    if symbol and len(symbol) <= 6:
                        # 将点号替换为短横线（Yahoo Finance使用短横线）
                        symbol = symbol.replace('.', '-')
                        if validate_ticker(symbol.replace('-', '')):
                            tickers.append(symbol)
            
            if not tickers:
                raise DataSourceError("解析S&P 500列表为空")
            
            # 3. 保存到缓存
            pd.DataFrame({'Symbol': tickers}).to_csv(cache_path, index=False)
            
            logger.info(f"S&P 500成分股获取完成: {len(tickers)} 只")
            return sorted(tickers)
            
        except requests.RequestException as e:
            logger.error(f"Wikipedia请求失败: {e}")
            
            # 降级：如果缓存存在（即使过期），也使用缓存
            if cache_path.exists():
                tickers = pd.read_csv(cache_path)['Symbol'].tolist()
                logger.warning(f"降级使用过期缓存: {len(tickers)} 只")
                return tickers
            
            raise DataSourceError(f"无法获取S&P 500成分股列表: {e}")
    
    def get_sp500_with_sectors(self) -> pd.DataFrame:
        """
        获取S&P 500成分股及其行业分类
        
        返回包含以下列的DataFrame:
        - Symbol: 股票代码
        - Security: 公司名称
        - GICS_Sector: GICS行业分类（如 Information Technology）
        - GICS_Sub_Industry: GICS子行业分类
        - Headquarters: 总部所在地
        - Date_Added: 加入S&P 500的日期
        - CIK: SEC中央索引键
        
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
            
            # 标准化列名
            df.columns = [c.replace(' ', '_') for c in df.columns]
            
            # 处理Symbol中的点号
            if 'Symbol' in df.columns:
                df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
            
            logger.info(f"S&P 500成分股+行业分类获取完成: {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"获取S&P 500详细信息失败: {e}")
            return pd.DataFrame()
    
    def crawl_sp500_historical(
        self,
        start_date: str = '2010-01-01',
        end_date: str = '2025-12-31',
        max_workers: int = 3
    ) -> CrawlSummary:
        """
        爬取全部S&P 500成分股的历史数据（2010-2025）
        
        这是主要的入口方法，一键完成：
        1. 获取S&P 500成分股列表
        2. 批量并发爬取历史日线数据
        3. 数据清洗+特征工程
        4. 生成爬取报告
        
        参数:
            start_date: 数据起始日期
            end_date: 数据结束日期
            max_workers: 最大并发数
        
        返回:
            CrawlSummary汇总统计
        """
        # 1. 获取股票列表
        with Timer("获取S&P 500成分股列表"):
            tickers = self.get_sp500_tickers()
        
        if not tickers:
            raise DataSourceError("无法获取S&P 500成分股列表")
        
        logger.info(
            f"准备爬取 {len(tickers)} 只S&P 500成分股的历史数据 "
            f"({start_date} ~ {end_date})"
        )
        
        # 2. 批量爬取
        summary = self.crawl_batch(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers
        )
        
        # 3. 重试失败的单只股票（最多重试1次）
        failed = self.get_failed_tickers()
        if failed:
            logger.info(f"重试 {len(failed)} 只失败的股票...")
            time.sleep(5)  # 等待冷却
            
            retry_results = []
            for ticker in failed:
                time.sleep(random.uniform(1.0, 3.0))  # 逐个重试，增加间隔
                result = self.crawl_single(ticker, start_date, end_date, force_update=True)
                retry_results.append(result)
            
            # 更新汇总统计
            retry_success = sum(1 for r in retry_results if r.status == 'success')
            summary.success_count += retry_success
            summary.failed_count -= retry_success
            summary.total_rows += sum(r.rows for r in retry_results if r.status == 'success')
            summary.results.extend(retry_results)
            
            logger.info(f"重试完成: 新增成功 {retry_success}/{len(failed)}")
        
        # 4. 打印最终报告
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: CrawlSummary) -> None:
        """打印爬取汇总报告"""
        print("\n" + "=" * 60)
        print("          S&P 500 历史数据爬取报告")
        print("=" * 60)
        print(f"  目标股票数:     {summary.total_tickers}")
        print(f"  成功获取:       {summary.success_count}")
        print(f"  失败:           {summary.failed_count}")
        print(f"  跳过(已存在):    {summary.skipped_count}")
        print(f"  总数据行数:     {summary.total_rows:,}")
        print(f"  成功率:         {summary.success_rate:.1%}")
        print(f"  总耗时:         {summary.total_duration/60:.1f} 分钟")
        
        if summary.failed_count > 0:
            failed_tickers = [r.ticker for r in summary.results if r.status == 'failed']
            print(f"\n  失败股票 ({len(failed_tickers)}):")
            for i, ticker in enumerate(failed_tickers):
                if i < 20:  # 最多显示20个
                    error = next(
                        (r.error for r in summary.results 
                         if r.ticker == ticker and r.error), '未知错误'
                    )
                    print(f"    - {ticker}: {error[:80]}")
                elif i == 20:
                    print(f"    ... 还有 {len(failed_tickers) - 20} 只")
                    break
        
        print("=" * 60 + "\n")


# ============================================================================
# 便捷函数：快速爬取S&P 500数据
# ============================================================================
def crawl_sp500_data(
    start_date: str = '2010-01-01',
    end_date: str = '2025-12-31',
    max_workers: int = 3
) -> CrawlSummary:
    """
    一键爬取S&P 500全部历史数据
    
    这是对外暴露的便捷函数，隐藏了内部实现细节。
    
    使用示例:
        from crawler import crawl_sp500_data
        summary = crawl_sp500_data('2010-01-01', '2025-12-31')
    
    参数:
        start_date: 起始日期
        end_date: 结束日期
        max_workers: 并发数
    
    返回:
        CrawlSummary汇总统计
    """
    crawler = SP500Crawler()
    return crawler.crawl_sp500_historical(start_date, end_date, max_workers)
