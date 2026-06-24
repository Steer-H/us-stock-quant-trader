"""Financial news scraper."""

import logging
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from data_pipeline.storage import ParquetStorage
from config.settings import PROCESSED_DATA_DIR
from crawler.news_sentiment import (
    POSITIVE_WORDS, NEGATIVE_WORDS, EARNINGS_WORDS,
    NewsSentimentAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """單條新聞"""
    ticker: str
    date: date
    headline: str
    source: str
    sentiment_score: float = 0.0
    is_earnings: bool = False


class NewsScraper:
    """
    多源RSS新聞抓取器
    
    從多個RSS源抓取股票相關新聞，進行情感分析。
    """
    
    # RSS源配置
    SOURCES = {
        'yahoo': 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US',
        'google': 'https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en',
    }
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or (PROCESSED_DATA_DIR / 'sentiment')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = NewsSentimentAnalyzer()
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def _fetch_rss(self, url: str, source_name: str) -> List[Tuple[str, str]]:
        """
        從RSS源抓取標題
        
        返回: [(headline, pub_date_str), ...]
        """
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode('utf-8', errors='replace')
        except urllib.error.URLError as e:
            logger.debug(f"RSS {source_name} 抓取失敗: {e}")
            return []
        except Exception as e:
            logger.debug(f"RSS {source_name} 解析失敗: {e}")
            return []
        
        # 解析RSS XML
        items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
        results = []
        
        for item in items:
            title_match = re.search(r'<title>(.*?)</title>', item)
            date_match = re.search(r'<pubDate>(.*?)</pubDate>', item)
            
            if title_match:
                title = title_match.group(1).strip()
                # 清理HTML實體
                title = title.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                title = title.replace('&quot;', '"').replace('&#39;', "'")
                title = title.replace('&apos;', "'")
                
                pub_date = date_match.group(1) if date_match else ''
                results.append((title, pub_date))
        
        logger.debug(f"RSS {source_name}: {len(results)} 條標題")
        return results
    
    def _parse_rss_date(self, date_str: str) -> Optional[date]:
        """解析RSS日期字符串"""
        if not date_str:
            return None
        
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.date()
            except ValueError:
                continue
        
        return None
    
    def fetch_ticker_news(
        self, ticker: str, days: int = 7, sources: List[str] = None
    ) -> List[NewsItem]:
        """
        抓取單只股票的新聞
        
        參數:
            ticker: 股票代碼
            days: 回溯天數
            sources: RSS源列表（默認全部）
        
        返回:
            NewsItem列表
        """
        if sources is None:
            sources = list(self.SOURCES.keys())
        
        cutoff = date.today() - timedelta(days=days)
        all_items: List[NewsItem] = []
        seen_headlines = set()
        
        for source_name in sources:
            url_template = self.SOURCES.get(source_name)
            if not url_template:
                continue
            
            url = url_template.format(ticker=ticker)
            
            # 抓取
            items = self._fetch_rss(url, source_name)
            
            for headline, date_str in items:
                # 去重
                normalized = headline.lower().strip()
                if normalized in seen_headlines:
                    continue
                seen_headlines.add(normalized)
                
                # 情感分析
                sentiment, is_earnings = self.analyzer.analyze_headline(headline)
                
                # 日期解析
                pub_date = self._parse_rss_date(date_str)
                if pub_date is None:
                    pub_date = date.today()
                
                if pub_date < cutoff:
                    continue
                
                all_items.append(NewsItem(
                    ticker=ticker,
                    date=pub_date,
                    headline=headline,
                    source=source_name,
                    sentiment_score=sentiment,
                    is_earnings=is_earnings,
                ))
            
            # 避免請求過快
            if len(sources) > 1:
                time.sleep(0.5)
        
        logger.info(
            f"{ticker}: 抓取 {len(all_items)} 條新聞 "
            f"({len(sources)} 源, {days}天)"
        )
        return all_items
    
    def aggregate_to_dataframe(
        self, items: List[NewsItem]
    ) -> pd.DataFrame:
        """
        將新聞列表聚合為每日情感DataFrame
        
        返回:
            DataFrame with columns:
            - news_sentiment_daily: 當日平均情感得分
            - news_count: 新聞數量
            - positive_ratio: 正面新聞佔比
            - earnings_news_count: 財報新聞數
        """
        if not items:
            return pd.DataFrame()
        
        records = []
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for item in items:
            grouped[item.date].append(item)
        
        for d, day_items in sorted(grouped.items()):
            scores = [it.sentiment_score for it in day_items]
            pos_count = sum(1 for s in scores if s > 0)
            earn_count = sum(1 for it in day_items if it.is_earnings)
            
            records.append({
                'date': d,
                'news_sentiment_daily': float(np.mean(scores)),
                'news_count': len(day_items),
                'positive_ratio': pos_count / len(day_items) if day_items else 0,
                'earnings_news_count': earn_count,
                'has_earnings_news': 1 if earn_count > 0 else 0,
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()
    
    def build_sentiment_features(
        self, tickers: List[str], lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        為多隻股票構建新聞情感特徵
        
        參數:
            tickers: 股票列表
            lookback_days: 回溯天數
        
        返回:
            MultiIndex DataFrame (date, ticker) with sentiment columns
        """
        all_features = []
        
        for ticker in tickers:
            news_items = self.fetch_ticker_news(ticker, days=lookback_days)
            if news_items:
                daily = self.aggregate_to_dataframe(news_items)
            else:
                daily = pd.DataFrame()
            
            if not daily.empty:
                daily['ticker'] = ticker
                daily = daily.reset_index().set_index(['date', 'ticker'])
                all_features.append(daily)
        
        if not all_features:
            logger.warning("無新聞數據")
            return pd.DataFrame()
        
        result = pd.concat(all_features)
        logger.info(
            f"新聞情感特徵構建完成: {len(result)} 行"
        )
        return result
    
    def merge_with_existing(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        將抓取的新聞情感合併到現有特徵DataFrame中
        
        只填充新聞相關列，不影響其他欄位。
        
        參數:
            df: 現有特徵DataFrame (單只股票, DatetimeIndex)
            ticker: 股票代碼
        
        返回:
            更新後的DataFrame
        """
        news_items = self.fetch_ticker_news(ticker, days=90)
        if not news_items:
            return df
        
        daily = self.aggregate_to_dataframe(news_items)
        if daily.empty:
            return df
        
        # 確保索引可比較
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)
        
        # 前向填充：新聞情緒影響持續數日
        daily['news_sentiment_3d'] = daily['news_sentiment_daily'].rolling(
            3, min_periods=1
        ).mean()
        daily['news_sentiment_7d'] = daily['news_sentiment_daily'].rolling(
            7, min_periods=1
        ).mean()
        
        # 合併到df
        for col in ['news_sentiment_3d', 'news_sentiment_7d']:
            if col in daily.columns:
                common_idx = df.index.intersection(daily.index)
                if len(common_idx) > 0:
                    df.loc[common_idx, col] = daily.loc[common_idx, col].values
        
        # 對於沒有新聞覆蓋的日期，保持之前的代理值（如財報代理）
        # 不做額外覆蓋
        
        return df


def scrape_recent_news_for_tickers(
    tickers: List[str],
    lookback_days: int = 30,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    便捷函數：為股票列表抓取最近新聞情感
    
    參數:
        tickers: 股票代碼列表
        lookback_days: 回溯天數
        output_dir: 輸出目錄
    
    返回:
        MultiIndex DataFrame
    """
    scraper = NewsScraper()
    result = scraper.build_sentiment_features(tickers, lookback_days)
    
    if output_dir and not result.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"news_sentiment_{date.today().isoformat()}.parquet"
        result.to_parquet(path)
        logger.info(f"已保存: {path}")
    
    return result


if __name__ == '__main__':
    # 快速測試
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    
    scraper = NewsScraper()
    test_tickers = ['AAPL', 'NVDA', 'MSFT']
    
    print(f"\n{'='*60}")
    print(f"測試新聞抓取: {test_tickers}")
    print(f"{'='*60}")
    
    for ticker in test_tickers:
        items = scraper.fetch_ticker_news(ticker, days=7)
        daily = scraper.aggregate_to_dataframe(items)
        if not daily.empty:
            avg_sent = daily['news_sentiment_daily'].mean()
            print(f"\n{ticker}: {len(items)} 條新聞, 平均情感={avg_sent:.3f}")
            print(f"  日期範圍: {daily.index.min().date()} ~ {daily.index.max().date()}")
            # 顯示幾條樣例
            for item in items[:3]:
                print(f"  [{item.sentiment_score:+.2f}] {item.headline[:80]}")
    
    print(f"\n{'='*60}")
    print("測試完成")
