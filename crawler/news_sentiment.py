"""
新闻情感与财报数据抓取模块

从 Yahoo Finance 抓取股票相关新闻和财报数据，
进行情感分析后作为ML模型的辅助特征。

核心功能：
- 批量抓取历史新闻标题（通过 yfinance）
- 轻量级情感分析（关键词+规则，无重量级NLP依赖）
- 财报日期和盈利惊喜数据
- 生成每日情感得分和财报特征

设计原则：
- 小权重特征：情感得分仅作为辅助信号
- 轻量实现：无transformers等重量级依赖
- 可缓存：结果存parquet避免重复抓取

数据源：
- Yahoo Finance (via yfinance .news / .earnings)
- 财报日历 (via yfinance .calendar / .earnings_dates)
"""

import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, date, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from data_pipeline.storage import ParquetStorage
from config.settings import PROCESSED_DATA_DIR
from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


# ============================================================================
# 情感词典
# ============================================================================
# 正面词汇 (利好)
POSITIVE_WORDS = {
    'beat', 'beats', 'beat estimates', 'exceed', 'exceeds', 'exceeded',
    'raise', 'raises', 'raised', 'raising', 'upgrade', 'upgrades', 'upgraded',
    'outperform', 'outperforms', 'strong', 'strength', 'growth', 'growing',
    'record', 'breakthrough', 'surge', 'surges', 'surged', 'soar', 'soars',
    'jump', 'jumps', 'jumped', 'rally', 'rallies', 'rallied',
    'profit', 'profitable', 'revenue growth', 'earnings growth',
    'buyback', 'buybacks', 'dividend', 'dividends', 'increase', 'increases',
    'positive', 'optimistic', 'bullish', 'momentum', 'recovery',
    'approval', 'approved', 'launch', 'launches', 'partnership',
    'better than expected', 'above expectations', 'top estimates',
    'guidance raised', 'revenue beat', 'earnings beat', 'double-digit growth',
    'new product', 'expansion', 'expanding', 'market share gain',
    'innovation', 'innovative', 'disruptive', 'leading',
    'undervalued', 'upside', 'catalyst', 'catalysts',
    'bull',
    'bulls',
    'bullish momentum',
    'buy signal',
    'buy rating',
    'outperform rating',
    'overweight',
    'accumulate',
    'add',
    'buy',
    'target raised',
    'price target',
    'raised target',
    'upside potential',
    'top pick',
    'favorite',
    'best idea',
    'compelling',
    'attractive',
    'opportunity',
    'tailwind',
    'tailwinds',
    'accelerating',
    'accelerate',
    'tops',
    'topped',
    'topping',
    'blowout',
    'stellar',
    'impressive',
    'robust',
    'resilient',
    'rebound',
    'rebounds',
    'turnaround',
    'gaining',
    'winner',
    'winning',
    'dominant',
    'dominance',
    'leader',
    'leadership',
    'moat',
    'competitive edge',
    'differentiated',
    'AI boom',
    'AI demand',
    'AI chip',
    'chip demand',
    'data center',
    'cloud growth',
    'IPO',
    'spin-off',
    'spin off',
    'undervalued gem',
    'hidden gem',
    'bargain',
    'cheap',
    'discount',
    'double',
    'triple',
    'skyrocket',
    'all-time high',
    'new high',
    'record high',
    'breakout',
    'breaking out',
    'golden cross',
    'bull flag',
    'momentum play',
    'growth stock',
    'strong buy',
    'promising',
    'bright',
    'optimistic outlook',
    'outpacing',
    'free cash flow',
    'cash machine',
    'capital return',
    'share repurchase',
    'increased dividend',
    'raised dividend',
}

# 负面词汇 (利空)
NEGATIVE_WORDS = {
    'miss', 'misses', 'missed', 'miss estimates', 'below', 'below expectations',
    'cut', 'cuts', 'cutting', 'downgrade', 'downgrades', 'downgraded',
    'underperform', 'underperforms', 'weak', 'weakness', 'decline', 'declining',
    'drop', 'drops', 'dropped', 'plunge', 'plunges', 'plunged', 'fall', 'falls',
    'loss', 'losses', 'losing', 'layoff', 'layoffs', 'restructuring',
    'investigation', 'probe', 'lawsuit', 'litigation', 'fine', 'fined',
    'negative', 'pessimistic', 'bearish', 'concern', 'concerns',
    'worse than expected', 'below estimates', 'guidance cut', 'guidance lowered',
    'revenue miss', 'earnings miss', 'delayed', 'delay', 'suspension',
    'warning', 'warns', 'risk', 'risks', 'risky', 'uncertainty',
    'overvalued', 'bubble', 'correction', 'sell-off', 'selloff',
    'debt', 'bankruptcy', 'default', 'dilution',
    'competition', 'competitive pressure', 'market share loss',
    'bear',
    'bears',
    'bearish',
    'sell signal',
    'sell rating',
    'underperform rating',
    'underweight',
    'reduce',
    'avoid',
    'target cut',
    'target lowered',
    'lowered target',
    'downside risk',
    'least favorite',
    'worst idea',
    'expensive',
    'headwind',
    'headwinds',
    'decelerating',
    'decelerate',
    'slowdown',
    'bottom line miss',
    'disappointing',
    'disappoints',
    'disappointed',
    'weak quarter',
    'soft quarter',
    'tough quarter',
    'challenging',
    'struggling',
    'struggles',
    'losing momentum',
    'fading',
    'loser',
    'disrupted',
    'disruption',
    'commoditized',
    'AI threat',
    'AI disruption',
    'overhyped',
    'bubble territory',
    'peak',
    'peaking',
    'topping out',
    'rolling over',
    'death cross',
    'bear flag',
    'breakdown',
    'breaking down',
    'crashing',
    'crash',
    'meltdown',
    'free fall',
    'strong sell',
    'dump',
    'dumping',
    'dismal',
    'grim',
    'bleak',
    'gloomy',
    'pessimistic',
    'losing share',
    'shrinking',
    'contraction',
    'eroding',
    'cash burn',
    'burning cash',
    'unprofitable',
    'no profit',
    'dividend cut',
    'suspending dividend',
    'paused buyback',
    'restructuring charges',
    'impairment',
    'write-down',
    'write down',
    'china risk',
    'export control',
    'export ban',
    'sanction',
    'tariff',
    'tariffs',
    'trade war',
    'geopolitical risk',
    'recession',
    'recession fear',
    'economic slowdown',
    'inflation',
    'margin pressure',
    'margin compression',
    'inventory glut',
    'oversupply',
    'demand weakness',
    'soft demand',
    'guided lower',
    'reduced guidance',
    'cut forecast',
}

# 财报相关关键词
EARNINGS_WORDS = {
    'earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'fiscal',
    'results', 'report', 'reports', 'reported', 'reporting',
    'eps', 'revenue', 'income', 'profit', 'margin',
    'guidance', 'outlook', 'forecast',
    'quarterly results',
    'quarter results',
    'Q1 results',
    'Q2 results',
    'Q3 results',
    'Q4 results',
    'fiscal year',
    'FY',
    'annual results',
    'earnings call',
    'earnings release',
    'earnings report',
    'financial results',
    'quarterly report',
    'annual report',
    '10-K',
    '10-Q',
    '8-K',
    'SEC filing',
    'pre-announcement',
    'preannouncement',
    'earnings preview',
    'earnings season',
    'reporting season',
}


@dataclass
class SentimentResult:
    """单条新闻的情感分析结果"""
    ticker: str
    date: date
    headline: str
    sentiment_score: float  # -1.0 (极空) 到 +1.0 (极多)
    is_earnings: bool       # 是否财报相关
    source: str              # 新闻来源


@dataclass
class DailySentiment:
    """单日汇总情感"""
    date: date
    ticker: str
    avg_sentiment: float           # 平均情感 (-1到1)
    news_count: int                # 新闻数量
    positive_ratio: float          # 正面新闻占比
    earnings_news_count: int       # 财报相关新闻数
    has_earnings_report: bool      # 当日是否有财报发布


class NewsSentimentAnalyzer:
    """
    新闻情感分析器
    
    从Yahoo Finance抓取新闻，使用关键词规则进行轻量级情感分析。
    无需transformers或大型NLP模型，依赖简单可靠。
    
    使用示例:
        analyzer = NewsSentimentAnalyzer()
        scores = analyzer.fetch_batch(['AAPL', 'MSFT'], lookback_days=30)
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or (PROCESSED_DATA_DIR / 'sentiment')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sentiment_cache: Dict[str, pd.DataFrame] = {}
    
    def analyze_headline(self, headline: str) -> Tuple[float, bool]:
        """
        分析单条新闻标题的情感
        
        返回:
            (sentiment_score, is_earnings)
            sentiment_score: -1.0到1.0
            is_earnings: 是否财报相关
        """
        if not headline or not isinstance(headline, str):
            return 0.0, False
        
        text = headline.lower()
        words = set(text.split())
        
        # 计数正负关键词
        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)
        
        # 检查子串匹配（短语）
        for phrase in POSITIVE_WORDS:
            if ' ' in phrase and phrase in text:
                pos_count += 1
        for phrase in NEGATIVE_WORDS:
            if ' ' in phrase and phrase in text:
                neg_count += 1
        
        # 情感得分
        total = pos_count + neg_count
        if total == 0:
            sentiment = 0.0
        else:
            sentiment = (pos_count - neg_count) / total
        
        # 财报检测
        is_earnings = bool(words & EARNINGS_WORDS)
        # 也检测财报关键词子串
        for word in EARNINGS_WORDS:
            if len(word) <= 2:
                continue
            if word in text:
                is_earnings = True
                break
        
        return round(sentiment, 3), is_earnings
    
    def fetch_news_for_ticker(
        self, ticker: str, lookback_days: int = 30
    ) -> List[SentimentResult]:
        """
        抓取单只股票的历史新闻并分析情感
        
        参数:
            ticker: 股票代码
            lookback_days: 回溯天数
        
        返回:
            SentimentResult列表
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance 未安装，跳过新闻抓取")
            return []
        
        results = []
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                logger.debug(f"{ticker}: 无新闻数据")
                return []
            
            cutoff = date.today() - timedelta(days=lookback_days)
            
            for item in news:
                try:
                    # 解析发布时间
                    pub_time = item.get('content', {}).get('pubDate', '')
                    if pub_time:
                        try:
                            pub_date = datetime.strptime(
                                pub_time[:10], '%Y-%m-%d'
                            ).date()
                        except ValueError:
                            pub_date = date.today()
                    else:
                        provider_pub = item.get('content', {}).get(
                            'providerPublishTime', 0
                        )
                        if provider_pub:
                            pub_date = datetime.fromtimestamp(
                                provider_pub
                            ).date()
                        else:
                            pub_date = date.today()
                    
                    if pub_date < cutoff:
                        continue
                    
                    headline = (
                        item.get('content', {}).get('title', '') or
                        item.get('title', '')
                    )
                    
                    if not headline:
                        continue
                    
                    sentiment, is_earnings = self.analyze_headline(headline)
                    
                    results.append(SentimentResult(
                        ticker=ticker,
                        date=pub_date,
                        headline=headline,
                        sentiment_score=sentiment,
                        is_earnings=is_earnings,
                        source=item.get('content', {}).get('provider', 'yahoo')
                    ))
                    
                except Exception:
                    continue
            
            logger.info(
                f"{ticker}: 抓取 {len(results)} 条新闻 "
                f"(回溯{lookback_days}天)"
            )
            
        except Exception as e:
            logger.warning(f"{ticker} 新闻抓取失败: {e}")
        
        return results
    
    def aggregate_daily(
        self, results: List[SentimentResult]
    ) -> pd.DataFrame:
        """
        将新闻情感结果聚合为每日指标
        
        返回:
            DataFrame with columns:
            - date, ticker
            - sentiment_score, news_count, positive_ratio
            - earnings_news_count, has_earnings_report
        """
        if not results:
            return pd.DataFrame()
        
        records = []
        grouped = defaultdict(list)
        for r in results:
            grouped[(r.date, r.ticker)].append(r)
        
        for (d, ticker), items in grouped.items():
            scores = [it.sentiment_score for it in items]
            positive_count = sum(1 for s in scores if s > 0)
            earnings_count = sum(1 for it in items if it.is_earnings)
            
            records.append({
                'date': d,
                'ticker': ticker,
                'news_sentiment': np.mean(scores),
                'news_count': len(items),
                'news_positive_ratio': positive_count / len(items) if items else 0,
                'earnings_news_count': earnings_count,
                'has_earnings_news': 1 if earnings_count > 0 else 0,
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()
    
    def fetch_earnings_data(
        self, ticker: str
    ) -> Optional[pd.DataFrame]:
        """
        抓取财报历史数据（盈利惊喜等）
        
        返回:
            DataFrame with earnings surprise info
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            # 财报日期
            earnings_dates = stock.earnings_dates
            if earnings_dates is None or earnings_dates.empty:
                return None
            
            df = earnings_dates.copy()
            
            # Normalize timezone-aware index to naive dates for merging
            if df.index.tz is not None:
                df.index = df.index.tz_convert('America/New_York').normalize().tz_localize(None)
            
            # 计算盈利惊喜
            if 'EPS Estimate' in df.columns and 'Reported EPS' in df.columns:
                df['earnings_surprise'] = (
                    df['Reported EPS'] - df['EPS Estimate']
                )
                df['earnings_surprise_pct'] = safe_divide(
                    df['earnings_surprise'],
                    df['EPS Estimate'].abs()
                )
            else:
                df['earnings_surprise'] = 0.0
                df['earnings_surprise_pct'] = 0.0
            
            df['ticker'] = ticker
            df['has_earnings_report'] = 1
            
            # Surprise ratio
            if 'Surprise(%)' in df.columns:
                df['earnings_surprise_pct'] = df['Surprise(%)'] / 100.0
            
            logger.info(
                f"{ticker}: 抓到 {len(df)} 条财报记录"
            )
            
            return df
            
        except Exception as e:
            logger.debug(f"{ticker} 财报抓取失败: {e}")
            return None
    
    def build_sentiment_features(
        self,
        tickers: List[str],
        target_dates: pd.DatetimeIndex,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        为给定日期范围构建情感特征矩阵
        
        参数:
            tickers: 股票列表
            target_dates: 目标日期范围
            lookback_days: 回溯天数
        
        返回:
            DataFrame with sentiment features per date per ticker
        """
        all_features = []
        
        for ticker in tickers:
            # 尝试加载缓存
            cache_key = f"{ticker}_sentiment_{lookback_days}"
            if cache_key in self._sentiment_cache:
                features = self._sentiment_cache[cache_key]
            else:
                # 抓取新闻
                news = self.fetch_news_for_ticker(ticker, lookback_days)
                if news:
                    daily = self.aggregate_daily(news)
                else:
                    daily = pd.DataFrame()
                
                # 抓取财报
                earnings = self.fetch_earnings_data(ticker)
                
                # 构建特征
                features = self._build_feature_df(
                    ticker, daily, earnings, target_dates
                )
                
                self._sentiment_cache[cache_key] = features
            
            if not features.empty:
                all_features.append(features)
        
        if not all_features:
            logger.warning("所有股票均无新闻/财报数据")
            return pd.DataFrame()
        
        result = pd.concat(all_features)
        logger.info(
            f"情感特征构建完成: {len(result)} 行, "
            f"{len(result.columns)} 列"
        )
        return result
    
    def _build_feature_df(
        self,
        ticker: str,
        daily_sentiment: pd.DataFrame,
        earnings: Optional[pd.DataFrame],
        target_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """构建单只股票的情感特征DataFrame"""
        idx = pd.DatetimeIndex(target_dates)
        df = pd.DataFrame(index=idx)
        df['ticker'] = ticker
        
        # 默认值
        df['news_sentiment'] = 0.0
        df['news_count'] = 0
        df['news_positive_ratio'] = 0.5
        df['earnings_news_count'] = 0
        df['has_earnings_news'] = 0
        df['earnings_surprise'] = 0.0
        df['earnings_surprise_pct'] = 0.0
        df['has_earnings_report'] = 0
        
        # 填充新闻情感
        if not daily_sentiment.empty:
            for col in ['news_sentiment', 'news_count', 'news_positive_ratio',
                        'earnings_news_count', 'has_earnings_news']:
                if col in daily_sentiment.columns:
                    common_idx = df.index.intersection(daily_sentiment.index)
                    if len(common_idx) > 0:
                        df.loc[common_idx, col] = daily_sentiment.loc[
                            common_idx, col
                        ].values
        
        # 填充财报数据
        if earnings is not None and not earnings.empty:
            for col in ['earnings_surprise', 'earnings_surprise_pct']:
                if col in earnings.columns:
                    earn_idx = earnings.index.intersection(df.index)
                    if len(earn_idx) > 0:
                        df.loc[earn_idx, col] = earnings.loc[earn_idx, col]
            
            # 财报发布日标记
            if 'has_earnings_report' in earnings.columns:
                earn_idx = earnings.index.intersection(df.index)
                if len(earn_idx) > 0:
                    df.loc[earn_idx, 'has_earnings_report'] = 1
        
        # 前向填充（新闻情感影响持续数日）
        df['news_sentiment'] = df['news_sentiment'].replace(0, np.nan)
        df['news_sentiment'] = df['news_sentiment'].ffill(limit=3).fillna(0)
        
        df['news_positive_ratio'] = df['news_positive_ratio'].replace(0.5, np.nan)
        df['news_positive_ratio'] = df['news_positive_ratio'].ffill(limit=3).fillna(0.5)
        
        # 滚动平均（3日平滑）
        df['news_sentiment_3d'] = df['news_sentiment'].rolling(3, min_periods=1).mean()
        df['news_sentiment_7d'] = df['news_sentiment'].rolling(7, min_periods=1).mean()
        
        return df


def build_news_features_for_pipeline(
    tickers: List[str],
    existing_df: pd.DataFrame,
    lookback_days: int = 90
) -> pd.DataFrame:
    """
    为data_pipeline添加新闻情感特征
    
    便捷函数，供数据处理管线调用。
    
    参数:
        tickers: 股票列表
        existing_df: 现有特征DataFrame（需含MultiIndex [date, ticker]）
        lookback_days: 回溯天数
    
    返回:
        添加了新闻特征的DataFrame
    """
    analyzer = NewsSentimentAnalyzer()
    
    # 获取日期范围
    if isinstance(existing_df.index, pd.MultiIndex):
        dates = existing_df.index.get_level_values(0).unique()
    else:
        dates = existing_df.index
    
    # 构建情感特征
    sentiment_df = analyzer.build_sentiment_features(
        tickers, dates, lookback_days
    )
    
    if sentiment_df.empty:
        logger.warning("无新闻特征可用，返回原始数据")
        return existing_df
    
    # 合并到现有DataFrame
    if isinstance(existing_df.index, pd.MultiIndex):
        existing_df = existing_df.reset_index()
        sentiment_df = sentiment_df.reset_index()
        
        merge_keys = ['date', 'ticker'] if 'date' in sentiment_df.columns else ['index', 'ticker']
        date_col = 'date' if 'date' in sentiment_df.columns else 'index'
        
        existing_df[date_col] = pd.to_datetime(existing_df[date_col])
        sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
        
        merged = existing_df.merge(
            sentiment_df[['date', 'ticker', 'news_sentiment_3d', 
                         'news_sentiment_7d', 'earnings_surprise_pct',
                         'has_earnings_report']],
            on=['date', 'ticker'],
            how='left'
        )
        
        # 填充缺失值
        for col in ['news_sentiment_3d', 'news_sentiment_7d', 
                     'earnings_surprise_pct']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        if 'has_earnings_report' in merged.columns:
            merged['has_earnings_report'] = merged['has_earnings_report'].fillna(0)
        
        return merged.set_index(['date', 'ticker'])
    
    return existing_df
