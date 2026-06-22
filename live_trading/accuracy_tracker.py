"""
在线模拟交易系统 - 模型预测准确率追踪模块

实时追踪ML模型的预测表现：
- 方向预测准确率（涨/跌判断正确率）
- 收益率预测误差（RMSE）
- 滚动窗口准确率（检测性能退化）
- 按股票分组的准确率
- 模型漂移检测

每笔预测都记录在案，用于后续分析和模型改进。
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import deque, defaultdict

import numpy as np
import pandas as pd

from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class PredictionRecord:
    """单次预测记录"""
    id: int
    ticker: str
    timestamp: str
    
    # 预测
    predicted_return: float     # 预测收益率
    predicted_direction: int    # 预测方向 (1=涨, 0=跌)
    confidence: float           # 预测置信度
    
    # 实际结果（在结果确认后填写）
    actual_return: Optional[float] = None
    actual_direction: Optional[int] = None
    is_correct: Optional[bool] = None  # 方向是否正确
    error: Optional[float] = None       # 预测误差
    
    # 状态
    status: str = 'pending'     # pending, confirmed, expired
    confirmed_at: Optional[str] = None


@dataclass
class AccuracySnapshot:
    """准确率快照"""
    timestamp: str = ''
    
    # 总体指标
    total_predictions: int = 0
    confirmed_predictions: int = 0
    direction_accuracy: float = 0.0      # 方向准确率
    recent_accuracy_50: float = 0.0      # 最近50次准确率
    rmse: float = 0.0                    # 均方根误差
    mae: float = 0.0                     # 平均绝对误差
    
    # 分类统计
    correct_long: int = 0   # 正确预测上涨
    correct_short: int = 0  # 正确预测下跌
    total_long: int = 0
    total_short: int = 0
    
    # 滚动指标
    accuracy_trend: str = 'stable'  # stable, improving, degrading
    
    # 按股票
    per_ticker_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # 模型状态
    is_acceptable: bool = True  # 是否可接受（方向准确率 >= 55%）


class AccuracyTracker:
    """
    模型预测准确率追踪器
    
    记录每次预测并持续追踪准确率。
    
    功能：
    - 记录预测和实际结果
    - 计算方向准确率和RMSE
    - 滚动窗口准确率（检测性能退化）
    - 模型漂移预警
    
    使用示例:
        tracker = AccuracyTracker()
        pred_id = tracker.record_prediction('AAPL', 0.012, 1, 0.85)
        # ... 确认结果后 ...
        tracker.confirm_prediction(pred_id, 0.015, 1)
        snapshot = tracker.get_snapshot()
    """
    
    def __init__(self, rolling_window: int = 50):
        """
        参数:
            rolling_window: 滚动窗口大小（用于检测准确率趋势）
        """
        self.rolling_window = rolling_window
        self.predictions: Dict[int, PredictionRecord] = {}
        self._id_counter: int = 0
        
        # 历史准确率序列
        self.accuracy_history: deque = deque(maxlen=200)
        
        # 按股票统计
        self.ticker_stats: Dict[str, Dict] = defaultdict(
            lambda: {'correct': 0, 'total': 0}
        )
        
        # 性能退化检测参数
        self.degradation_threshold: float = 0.05  # 准确率下降5%视为退化
        
        logger.info(f"准确率追踪器初始化, 滚动窗口: {rolling_window}")
    
    def record_prediction(
        self,
        ticker: str,
        predicted_return: float,
        predicted_direction: int,
        confidence: float = 0.5
    ) -> int:
        """
        记录一次预测
        
        参数:
            ticker: 股票代码
            predicted_return: 预测收益率
            predicted_direction: 预测方向 (1=涨, 0=跌)
            confidence: 预测置信度
        
        返回:
            预测记录ID
        """
        self._id_counter += 1
        
        record = PredictionRecord(
            id=self._id_counter,
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat(),
            predicted_return=predicted_return,
            predicted_direction=predicted_direction,
            confidence=confidence,
            status='pending'
        )
        
        self.predictions[self._id_counter] = record
        
        return self._id_counter
    
    def confirm_prediction(
        self,
        prediction_id: int,
        actual_return: float,
        actual_direction: int
    ) -> bool:
        """
        确认预测结果
        
        当真实结果出来后，更新预测记录并重新计算准确率。
        
        参数:
            prediction_id: 预测记录ID
            actual_return: 实际收益率
            actual_direction: 实际方向 (1=涨, 0=跌)
        
        返回:
            是否确认成功
        """
        record = self.predictions.get(prediction_id)
        if not record:
            logger.warning(f"未找到预测记录: {prediction_id}")
            return False
        
        # 更新记录
        record.actual_return = actual_return
        record.actual_direction = actual_direction
        record.is_correct = (record.predicted_direction == actual_direction)
        record.error = abs(record.predicted_return - actual_return)
        record.status = 'confirmed'
        record.confirmed_at = datetime.now().isoformat()
        
        # 更新统计
        self.accuracy_history.append(1 if record.is_correct else 0)
        
        ticker = record.ticker
        self.ticker_stats[ticker]['total'] += 1
        if record.is_correct:
            self.ticker_stats[ticker]['correct'] += 1
        
        logger.debug(
            f"预测确认 #{prediction_id}: {ticker} "
            f"预测{'涨' if record.predicted_direction else '跌'}, "
            f"实际{'涨' if actual_direction else '跌'}, "
            f"{'✓' if record.is_correct else '✗'}"
        )
        
        return True
    
    def get_direction_accuracy(self) -> float:
        """
        获取总体方向准确率
        
        返回:
            准确率 (0.0 ~ 1.0)
        """
        confirmed = [r for r in self.predictions.values() if r.status == 'confirmed']
        if not confirmed:
            return 0.0
        
        correct = sum(1 for r in confirmed if r.is_correct)
        return correct / len(confirmed)
    
    def get_recent_accuracy(self, n: int = 50) -> float:
        """
        获取最近N次预测的准确率
        
        用于检测模型性能是否在退化。
        
        参数:
            n: 最近N次
        
        返回:
            准确率
        """
        recent = list(self.accuracy_history)[-n:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)
    
    def get_rmse(self) -> float:
        """
        获取预测收益率的RMSE
        
        返回:
            均方根误差
        """
        confirmed = [
            r for r in self.predictions.values() 
            if r.status == 'confirmed' and r.error is not None
        ]
        if not confirmed:
            return 0.0
        
        squared_errors = [r.error ** 2 for r in confirmed]
        return float(np.sqrt(np.mean(squared_errors)))
    
    def get_mae(self) -> float:
        """获取平均绝对误差"""
        confirmed = [
            r for r in self.predictions.values() 
            if r.status == 'confirmed' and r.error is not None
        ]
        if not confirmed:
            return 0.0
        
        errors = [r.error for r in confirmed]
        return float(np.mean(errors))
    
    def check_degradation(self) -> Tuple[bool, str]:
        """
        检查模型性能是否退化
        
        比较最近窗口和历史准确率，检测下降趋势。
        
        返回:
            (是否退化, 描述)
        """
        if len(self.accuracy_history) < self.rolling_window * 2:
            return False, '数据不足，无法检测退化'
        
        # 取两段窗口的准确率
        recent = list(self.accuracy_history)[-self.rolling_window:]
        older = list(self.accuracy_history)[-2*self.rolling_window:-self.rolling_window]
        
        recent_acc = sum(recent) / len(recent) if recent else 0
        older_acc = sum(older) / len(older) if older else 0
        
        diff = recent_acc - older_acc
        
        if diff < -self.degradation_threshold:
            return True, (
                f"⚠️ 模型性能退化: 最近{self.rolling_window}次准确率 {recent_acc:.1%}, "
                f"之前{self.rolling_window}次 {older_acc:.1%}, 下降 {abs(diff):.1%}"
            )
        
        return False, f'模型性能稳定 (变化 {diff:+.1%})'
    
    def get_snapshot(self) -> AccuracySnapshot:
        """
        获取准确率快照
        
        返回:
            AccuracySnapshot对象
        """
        confirmed = [r for r in self.predictions.values() if r.status == 'confirmed']
        pending = [r for r in self.predictions.values() if r.status == 'pending']
        
        snap = AccuracySnapshot(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_predictions=len(self.predictions),
            confirmed_predictions=len(confirmed),
        )
        
        if confirmed:
            snap.direction_accuracy = self.get_direction_accuracy()
            snap.recent_accuracy_50 = self.get_recent_accuracy(50)
            snap.rmse = self.get_rmse()
            snap.mae = self.get_mae()
            
            # 分类统计
            for r in confirmed:
                if r.predicted_direction == 1:
                    snap.total_long += 1
                    if r.is_correct:
                        snap.correct_long += 1
                else:
                    snap.total_short += 1
                    if r.is_correct:
                        snap.correct_short += 1
            
            # 是否可接受
            snap.is_acceptable = snap.direction_accuracy >= 0.55
        
        # 按股票准确率
        for ticker, stats in self.ticker_stats.items():
            if stats['total'] > 0:
                snap.per_ticker_accuracy[ticker] = stats['correct'] / stats['total']
        
        # 趋势判断
        _, trend = self.check_degradation()
        snap.accuracy_trend = 'degrading' if '退化' in trend else 'stable'
        
        return snap
    
    def get_recent_predictions(self, n: int = 10) -> pd.DataFrame:
        """
        获取最近N次预测记录
        
        参数:
            n: 记录数量
        
        返回:
            DataFrame
        """
        recent_ids = sorted(self.predictions.keys(), reverse=True)[:n]
        
        records = []
        for pid in recent_ids:
            r = self.predictions[pid]
            records.append({
                'ID': r.id,
                'Ticker': r.ticker,
                '预测收益率': f"{r.predicted_return:+.4%}",
                '预测方向': '📈涨' if r.predicted_direction else '📉跌',
                '置信度': f"{r.confidence:.1%}",
                '实际方向': '📈涨' if r.actual_direction else '📉跌' if r.actual_direction is not None else '⏳待定',
                '结果': '✓正确' if r.is_correct else '✗错误' if r.is_correct is not None else '⏳',
                '时间': r.timestamp[:19],
            })
        
        return pd.DataFrame(records)
    
    def get_accuracy_summary(self) -> Dict[str, str]:
        """
        获取准确率摘要（格式化字符串）
        
        返回:
            {指标: 格式化值}
        """
        snap = self.get_snapshot()
        
        status_icon = '✅' if snap.is_acceptable else '⚠️'
        
        return {
            '总预测次数': str(snap.total_predictions),
            '已确认次数': str(snap.confirmed_predictions),
            '方向准确率': f"{snap.direction_accuracy:.1%}",
            '最近50次': f"{snap.recent_accuracy_50:.1%}",
            '涨预测正确': f"{snap.correct_long}/{snap.total_long}",
            '跌预测正确': f"{snap.correct_short}/{snap.total_short}",
            'RMSE': f"{snap.rmse:.6f}",
            'MAE': f"{snap.mae:.6f}",
            '准确率趋势': snap.accuracy_trend,
            '模型状态': f"{status_icon} {'可接受' if snap.is_acceptable else '需调优'}",
        }
