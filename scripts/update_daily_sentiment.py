#!/usr/bin/env python3
"""
每日新闻情感更新脚本

从Google News RSS抓取最新新闻，更新parquet文件中的情感特征。
建议通过cron每天运行一次（美股收盘后）。

运行方式:
    PYTHONPATH=. python3 scripts/update_daily_sentiment.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(message)s',
)
logger = logging.getLogger('daily_sentiment')

from crawler.news_scraper import NewsScraper
from data_pipeline.storage import ParquetStorage
from config.settings import PROCESSED_DATA_DIR


def main():
    storage = ParquetStorage(PROCESSED_DATA_DIR)
    tickers = sorted([
        k.replace('_features', '')
        for k in storage.list_keys()
        if k.endswith('_features')
    ])
    
    scraper = NewsScraper()
    today = date.today()
    
    logger.info(f"开始每日情感更新: {today}, {len(tickers)} 只股票")
    
    updated = 0
    total_news = 0
    
    for ticker in tickers:
        try:
            df = storage.load(f'{ticker}_features')
            if df is None or len(df) == 0:
                continue
            
            # 抓取最近新闻
            news_items = scraper.fetch_ticker_news(
                ticker, days=7, sources=['google']
            )
            if not news_items:
                continue
            
            total_news += len(news_items)
            daily = scraper.aggregate_to_dataframe(news_items)
            if daily.empty:
                continue
            
            # 归一化时区
            if daily.index.tz is not None:
                daily.index = daily.index.tz_localize(None)
            
            # 滚动平均
            daily['news_sentiment_3d'] = daily['news_sentiment_daily'].rolling(
                3, min_periods=1
            ).mean()
            daily['news_sentiment_7d'] = daily['news_sentiment_daily'].rolling(
                7, min_periods=1
            ).mean()
            
            # 更新最近日期的情感值
            recent_cutoff = df.index.max() - pd.Timedelta(days=14)
            recent_idx = df.index[df.index >= recent_cutoff]
            overlap = recent_idx.intersection(daily.index)
            
            if len(overlap) > 0:
                for col in ['news_sentiment_3d', 'news_sentiment_7d']:
                    if col in daily.columns:
                        df.loc[overlap, col] = daily.loc[overlap, col].values
                
                storage.save(df, f'{ticker}_features')
                updated += 1
                
        except Exception as e:
            logger.warning(f"  {ticker}: {e}")
    
    logger.info(
        f"每日更新完成: {updated}/{len(tickers)} 只股票更新, "
        f"共 {total_news} 条新闻"
    )


if __name__ == '__main__':
    main()
