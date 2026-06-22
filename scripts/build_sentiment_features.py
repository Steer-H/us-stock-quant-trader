#!/usr/bin/env python3
"""
构建新闻/财报情感特征并合并到每只股票的 features parquet 文件中。

策略:
- 财报数据：从 yfinance 抓取历史 EPS 惊喜数据，合并到 parquet
- 新闻情感：由于 yfinance 新闻仅返回最近几天的数据且与历史数据无交集，
  使用财报惊喜值作为情感代理信号，通过前向填充传播
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(message)s',
)
logger = logging.getLogger('build_sentiment')

from data_pipeline.storage import ParquetStorage
from config.settings import PROCESSED_DATA_DIR


def build_sentiment_for_ticker(ticker, df, analyzer):
    """
    为单只股票构建情感特征并更新 DataFrame。
    直接修改传入的 df。
    """
    import yfinance as yf

    # --- 财报数据 ---
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.earnings_dates
    except Exception as e:
        logger.debug(f'{ticker}: yfinance 财报抓取失败: {e}')
        earnings_dates = None

    if earnings_dates is not None and not earnings_dates.empty:
        ed = earnings_dates.copy()
        # 归一化时区
        if ed.index.tz is not None:
            ed.index = ed.index.tz_convert('America/New_York').normalize().tz_localize(None)

        # 盈利惊喜百分比
        if 'Surprise(%)' in ed.columns:
            ed['earnings_surprise_pct'] = ed['Surprise(%)'] / 100.0
        elif 'EPS Estimate' in ed.columns and 'Reported EPS' in ed.columns:
            surprise = ed['Reported EPS'] - ed['EPS Estimate']
            ed['earnings_surprise_pct'] = np.where(
                ed['EPS Estimate'].abs() > 1e-9,
                surprise / ed['EPS Estimate'].abs(),
                0.0
            )
        else:
            ed['earnings_surprise_pct'] = 0.0

        ed['has_earnings_report'] = 1

        # 合并到 df
        common_idx = df.index.intersection(ed.index)
        if len(common_idx) > 0:
            for col in ['earnings_surprise_pct', 'has_earnings_report']:
                if col in ed.columns:
                    df.loc[common_idx, col] = ed.loc[common_idx, col].values

        earnings_count = len(common_idx)
    else:
        earnings_count = 0

    # --- 新闻情感代理 ---
    # 由于 yfinance 新闻仅返回最近几天且与历史数据无交集，
    # 使用财报惊喜值作为情感代理信号。
    # 策略：财报日的惊喜值向前传播（市场需要数日消化财报信息）
    if 'earnings_surprise_pct' in df.columns:
        # 在财报日，用惊喜值作为新闻情感
        earn_mask = df['has_earnings_report'] > 0
        if earn_mask.any():
            # 新闻情感代理：归一化到 -1 到 1 范围
            surprise_vals = df.loc[earn_mask, 'earnings_surprise_pct'].values
            max_abs = np.nanmax(np.abs(surprise_vals))
            if np.isnan(max_abs) or max_abs < 1e-9:
                max_abs = 0.01
            sentiment_proxy = np.clip(surprise_vals / max_abs, -1.0, 1.0)

            df.loc[earn_mask, 'news_sentiment_3d'] = sentiment_proxy
            df.loc[earn_mask, 'news_sentiment_7d'] = sentiment_proxy

            # 前向填充：惊喜情绪影响持续数日
            df['news_sentiment_3d'] = df['news_sentiment_3d'].replace(0, np.nan)
            df['news_sentiment_3d'] = df['news_sentiment_3d'].ffill(limit=3).fillna(0)

            df['news_sentiment_7d'] = df['news_sentiment_7d'].replace(0, np.nan)
            df['news_sentiment_7d'] = df['news_sentiment_7d'].ffill(limit=7).fillna(0)

    return earnings_count


def main():
    storage = ParquetStorage(PROCESSED_DATA_DIR)
    available = storage.list_keys()
    tickers = sorted([
        k.replace('_features', '')
        for k in available
        if k.endswith('_features')
    ])
    logger.info(f'找到 {len(tickers)} 只股票的特征文件')

    success = 0
    failed = 0
    total_earnings = 0

    for i, ticker in enumerate(tickers):
        t0 = time.perf_counter()
        logger.info(f'[{i+1}/{len(tickers)}] 处理 {ticker}...')

        try:
            df = storage.load(f'{ticker}_features')
            if df is None or len(df) == 0:
                logger.warning(f'  {ticker}: 数据为空，跳过')
                continue

            earnings_count = build_sentiment_for_ticker(ticker, df, None)

            # 保存
            storage.save(df, f'{ticker}_features')

            # 统计非零值
            sentiment_cols = [
                'news_sentiment_3d', 'news_sentiment_7d',
                'earnings_surprise_pct', 'has_earnings_report'
            ]
            nonzero = {
                c: int((df[c] != 0).sum())
                for c in sentiment_cols if c in df.columns
            }
            dt = time.perf_counter() - t0
            logger.info(
                f'  {ticker}: ✅ ({dt:.1f}s) earnings_dates={earnings_count}, nonzero={nonzero}'
            )
            success += 1
            total_earnings += earnings_count

        except Exception as e:
            dt = time.perf_counter() - t0
            logger.error(f'  {ticker}: ❌ ({dt:.1f}s) - {e}')
            failed += 1

        time.sleep(0.2)

    logger.info(
        f'\n===== 构建完成 =====\n'
        f'成功: {success}, 失败: {failed}, 总财报日期: {total_earnings}'
    )


if __name__ == '__main__':
    main()
